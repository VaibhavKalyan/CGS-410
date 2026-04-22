import torch
import numpy as np

def compute_attention_entropy(attention_probs):
    eps = 1e-12
    return -torch.sum(attention_probs * torch.log2(attention_probs + eps), dim=-1)

def compute_offset_profile(attention_weights, offsets=range(-5, 6)):
    _, num_heads, seq_len, _ = attention_weights.shape
    profiles = {}
    for offset in offsets:
        if seq_len <= 1:
            profiles[offset] = np.zeros(num_heads)
            continue
        q_idx = torch.arange(0, max(1, seq_len - offset)) if offset >= 0 else torch.arange(-offset, seq_len)
        k_idx = q_idx + offset
        valid = (k_idx >= 0) & (k_idx < seq_len) & (q_idx < seq_len)
        q_idx, k_idx = q_idx[valid], k_idx[valid]
        if len(q_idx) == 0:
            profiles[offset] = np.zeros(num_heads)
            continue
        attn_at_offset = attention_weights[:, :, q_idx, k_idx]
        profiles[offset] = attn_at_offset.mean(dim=(0, 2)).numpy()
    return profiles

def compute_syntax_match_rate(attention_weights, dep_metadata):
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    match_counts = np.zeros(num_heads)
    total = 0
    for b in range(batch_size):
        meta = dep_metadata[b] if b < len(dep_metadata) else None
        if meta is None or meta.get('subject_idx') is None: continue
        subj, verb = meta['subject_idx'], meta['verb_idx']
        if subj >= seq_len or verb >= seq_len: continue
        top_key = attention_weights[b, :, subj, :].argmax(dim=-1).numpy()
        match_counts += (top_key == verb).astype(float)
        total += 1
    return match_counts / total if total > 0 else np.zeros(num_heads)

def compute_attention_to_target(attention_weights, dep_metadata):
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    total, count = np.zeros(num_heads), 0
    for b in range(batch_size):
        meta = dep_metadata[b] if b < len(dep_metadata) else None
        if meta is None or meta.get('subject_idx') is None: continue
        subj, verb = meta['subject_idx'], meta['verb_idx']
        if subj >= seq_len or verb >= seq_len: continue
        total += attention_weights[b, :, subj, verb].numpy()
        count += 1
    return total / count if count > 0 else np.zeros(num_heads)

def compute_delimiter_attention(attention_weights, attention_mask):
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    total = np.zeros(num_heads)
    for b in range(batch_size):
        seq_len_b = int(attention_mask[b].sum().item())
        if seq_len_b < seq_len:
            total += attention_weights[b, :, :, seq_len_b:].sum(dim=-1).mean(dim=-1).numpy()
        last_idx = max(0, seq_len_b - 1)
        total += attention_weights[b, :, :, last_idx].mean(dim=-1).numpy()
    return total / batch_size

def extract_head_features(attention_weights, seq_lengths, dep_metadata=None, attention_mask=None):
    num_layers = len(attention_weights)
    batch_size, num_heads, seq_len, _ = attention_weights[0].shape
    features = []

    for layer_idx in range(num_layers):
        # Convert float16 back to float32 for stable mathematical operations
        aw_cpu = attention_weights[layer_idx].float()

        entropy = compute_attention_entropy(aw_cpu).mean(dim=(0, 2)).numpy()
        
        if seq_len >= 2:
            q_idx, k_idx = torch.arange(1, seq_len), torch.arange(0, seq_len - 1)
            prev_attn = aw_cpu[:, :, q_idx, k_idx].mean(dim=(0, 2)).numpy()
        else:
            prev_attn = np.zeros(num_heads)

        diag_idx = torch.arange(seq_len)
        self_attn = aw_cpu[:, :, diag_idx, diag_idx].mean(dim=(0, 2)).numpy()
        offset_profile = compute_offset_profile(aw_cpu)
        syntax_match = compute_syntax_match_rate(aw_cpu, dep_metadata) if dep_metadata is not None else np.zeros(num_heads)
        attn_to_target = compute_attention_to_target(aw_cpu, dep_metadata) if dep_metadata is not None else np.zeros(num_heads)
        delim_attn = compute_delimiter_attention(aw_cpu, attention_mask) if attention_mask is not None else np.zeros(num_heads)

        for h in range(num_heads):
            head_dict = {
                'layer': layer_idx, 'head': h, 'avg_entropy': float(entropy[h]),
                'prev_attn': float(prev_attn[h]), 'self_attn': float(self_attn[h]),
                'syntax_match': float(syntax_match[h]), 'attn_to_target': float(attn_to_target[h]),
                'delimiter_attn': float(delim_attn[h]),
            }
            for offset, vals in offset_profile.items(): head_dict[f'offset_{offset}'] = float(vals[h])
            features.append(head_dict)

    return features

def features_to_matrix(features):
    exclude = {'layer', 'head'}
    all_keys = sorted(k for k in features[0].keys() if k not in exclude)
    matrix = np.array([[f[k] for k in all_keys] for f in features])
    head_metadata = [(f['layer'], f['head']) for f in features]
    return matrix, all_keys, head_metadata
