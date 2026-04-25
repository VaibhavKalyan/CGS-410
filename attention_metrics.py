import os
import torch
import numpy as np
 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Shared style constants
_TEXT  = '#2C3E50'
_SPINE = '#BDC3C7'
_GRID  = '#E8ECF0'
_BG    = '#FAFAFA'

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

def _get_token_positions(meta):
    """Return (subject_token_idx, verb_token_idx), preferring aligned token indices."""
    subj = meta.get('subject_token_idx')
    verb = meta.get('verb_token_idx')
    if subj is None:
        subj = meta.get('subject_idx')
    if verb is None:
        verb = meta.get('verb_idx')
    return subj, verb

def compute_syntax_match_rate(attention_weights, dep_metadata):
    """Check if VERB attends to SUBJECT (causal direction for GPT-2)."""
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    match_counts = np.zeros(num_heads)
    total = 0
    for b in range(batch_size):
        meta = dep_metadata[b] if b < len(dep_metadata) else None
        if meta is None or meta.get('subject_idx') is None: continue
        subj, verb = _get_token_positions(meta)
        if subj is None or verb is None: continue
        if subj >= seq_len or verb >= seq_len: continue
        if verb <= subj: continue  # verb must come after subject for causal attn
        # VERB attending to SUBJECT (causal: verb looks back)
        top_key = attention_weights[b, :, verb, :].argmax(dim=-1).numpy()
        match_counts += (top_key == subj).astype(float)
        total += 1
    return match_counts / total if total > 0 else np.zeros(num_heads)

def compute_attention_to_target(attention_weights, dep_metadata):
    """Attention mass from VERB to SUBJECT (causal direction for GPT-2)."""
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    total, count = np.zeros(num_heads), 0
    for b in range(batch_size):
        meta = dep_metadata[b] if b < len(dep_metadata) else None
        if meta is None or meta.get('subject_idx') is None: continue
        subj, verb = _get_token_positions(meta)
        if subj is None or verb is None: continue
        if subj >= seq_len or verb >= seq_len: continue
        if verb <= subj: continue
        # VERB→SUBJECT attention mass
        total += attention_weights[b, :, verb, subj].numpy()
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


# ── RQ1 Plot ──────────────────────────────────────────────────────────────────

def plot_rq1_resilience(df, output_dir='outputs', lang='en'):
    """
    Two-panel line chart answering RQ1:
      Left  — Entropy vs. pruning level  (specialization signal: entropy should drop)
      Right — Syntax-match rate vs. pruning level  (grammar retention: should stay high)
    Individual expert-head traces shown in translucent grey; bold blue = mean ± 1 SD band.
    """
    os.makedirs(output_dir, exist_ok=True)

    active = df[df['status'] == 'ACTIVE'].copy()
    if active.empty:
        print("    [RQ1] No ACTIVE head data to plot.")
        return None

    milestones = sorted(active['milestone'].unique(), reverse=True)  # 48 → 5

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')
    fig.patch.set_facecolor('white')
    fig.suptitle(
        f'[{lang.upper()}]  RQ1 — Functional Resilience of Expert Heads',
        fontsize=14, fontweight='bold', color=_TEXT, y=1.02,
    )

    specs = [
        ('entropy',      'Attention Entropy (bits)',  'Entropy  —  Specialization Under Pruning'),
        ('syntax_match', 'Syntax-Match Rate',         'Syntax Match  —  Grammar Retention'),
    ]

    for ax, (metric, ylabel, title) in zip(axes, specs):
        ax.set_facecolor(_BG)

        # Thin individual head traces
        for (_, _), grp in active.groupby(['layer', 'head']):
            grp_s = grp.sort_values('milestone', ascending=False)
            ax.plot(range(len(grp_s)), grp_s[metric].values,
                    color='#AABCD4', linewidth=0.8, alpha=0.40, zorder=2)

        # Mean ± 1 SD band
        summary = (active.groupby('milestone')[metric]
                         .agg(['mean', 'std'])
                         .reindex(milestones)
                         .reset_index())
        xs = range(len(milestones))

        ax.fill_between(xs,
                        summary['mean'] - summary['std'].fillna(0),
                        summary['mean'] + summary['std'].fillna(0),
                        color='#4C72B0', alpha=0.15, zorder=3)
        ax.plot(xs, summary['mean'],
                color='#4C72B0', linewidth=2.6, marker='o', markersize=7,
                markeredgecolor='white', markeredgewidth=1.3,
                zorder=4, label='Mean ± 1 SD')

        # Value labels above each point
        for x, (_, row) in zip(xs, summary.iterrows()):
            ax.annotate(f'{row["mean"]:.2f}',
                        xy=(x, row['mean']),
                        xytext=(0, 11), textcoords='offset points',
                        ha='center', fontsize=8.5, color=_TEXT, fontweight='bold')

        ax.set_title(title, fontsize=12, fontweight='bold', color=_TEXT, pad=12)
        ax.set_xlabel('Active Heads  (right = least pruned)', fontsize=11,
                      color=_TEXT, labelpad=6)
        ax.set_ylabel(ylabel, fontsize=11, color=_TEXT, labelpad=6)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([str(m) for m in milestones], fontsize=9, color=_TEXT)
        ax.grid(True, color=_GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(_SPINE)
        ax.spines['bottom'].set_color(_SPINE)
        ax.tick_params(colors=_TEXT, labelsize=9)
        ax.legend(fontsize=9, framealpha=0.92, edgecolor=_SPINE)

    # Direction arrow annotation on both axes
    for ax in axes:
        ax.annotate('← More pruned          Less pruned →',
                    xy=(0.5, -0.16), xycoords='axes fraction',
                    ha='center', fontsize=8.5, color='#888',
                    style='italic')

    plt.tight_layout(pad=2.5)
    fname = os.path.join(output_dir, f'rq1_resilience_plot_{lang}.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {fname}")
    return fname
