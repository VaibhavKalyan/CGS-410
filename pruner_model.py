"""
pruner_model.py — Pruning engine + head importance attribution.

Uses ABLATION-BASED importance: for each head, measure how much the
language-model loss increases when that head is zeroed out.

Key design decisions:
  1. Loss computed ONLY on real tokens (padding masked out via attention_mask)
  2. Loss computed in float64 for precision (single-head effects are tiny)
  3. Pruning via register_forward_hook (not head_mask, which newer
     transformers may silently ignore)
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel


def _get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HardConcreteHeadMask(nn.Module):
    """
    Stores a binary (num_layers, num_heads) mask via log_alpha.
    log_alpha = +10 → head ON;  log_alpha = -10 → head OFF.
    """
    def __init__(self, num_layers, num_heads):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full((num_layers, num_heads), 10.0))

    def forward(self, training=False):
        return (self.log_alpha > 0).float()

    def get_active_heads(self, threshold=0.5):
        return (self.log_alpha > 0).sum().item()


class GPT2PrunerWrapper(nn.Module):
    def __init__(self, model_name='gpt2-medium'):
        super().__init__()
        self.device = _get_device()
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name, attn_implementation='eager'
        ).to(self.device)
        self.config = self.model.config
        self.num_layers = self.config.n_layer
        self.num_heads  = self.config.n_head
        self.head_dim   = self.config.n_embd // self.config.n_head
        self.head_mask_module = HardConcreteHeadMask(
            self.num_layers, self.num_heads
        ).to(self.device)
        self._hooks = []

    def apply_pruning(self):
        """Apply pruning via forward hooks — works regardless of head_mask support."""
        self.remove_pruning()
        mask = self.head_mask_module(training=False)  # (L, H) binary
        for layer_idx in range(self.num_layers):
            layer_mask = mask[layer_idx]  # (H,)
            if layer_mask.all():
                continue  # all heads ON, no hook needed
            attn_module = self.model.transformer.h[layer_idx].attn
            hook = attn_module.register_forward_hook(
                self._make_pruning_hook(layer_mask)
            )
            self._hooks.append(hook)

    def remove_pruning(self):
        """Remove all pruning hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @staticmethod
    def _make_pruning_hook(layer_mask):
        """Create a hook that zeros out pruned heads' attention outputs + weights."""
        def hook(module, args, output):
            # output = (attn_output, present[, attn_weights])
            attn_output = output[0]  # (B, S, n_embd) after c_proj
            num_heads = layer_mask.numel()
            head_dim = attn_output.size(-1) // num_heads

            # Zero the attention output for pruned heads
            B, S, D = attn_output.shape
            reshaped = attn_output.view(B, S, num_heads, head_dim)
            reshaped = reshaped * layer_mask.view(1, 1, num_heads, 1)
            attn_output = reshaped.view(B, S, D)

            new_output = (attn_output, output[1])

            # Also zero the attention weights if returned
            if len(output) > 2 and output[2] is not None:
                attn_weights = output[2]  # (B, H, S, S)
                attn_weights = attn_weights * layer_mask.view(1, num_heads, 1, 1)
                new_output = new_output + (attn_weights,)

            return new_output
        return hook

    def forward(self, input_ids, attention_mask=None,
                output_attentions=True, output_hidden_states=True):
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )


# ── Head Importance via Ablation ──────────────────────────────────────────────

def compute_head_importance(model, inputs, batch_size=8):
    """
    Compute each head's importance via loss-increase-on-ablation.

    Key design:
      - Loss computed ONLY on real tokens (padding ignored via attention_mask)
      - Loss in float64 for precision
      - Uses hook-based masking (not head_mask arg, which may be ignored)
    """
    model.eval()
    model.remove_pruning()
    device = model.device
    num_layers = model.num_layers
    num_heads  = model.num_heads
    n_samples  = inputs['input_ids'].size(0)
    attn_mask  = inputs['attention_mask']

    def _get_loss():
        """LM loss on real tokens only, in float64."""
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                b_ids  = inputs['input_ids'][i:i+batch_size].to(device)
                b_mask = attn_mask[i:i+batch_size].to(device)

                out = model.model(
                    input_ids=b_ids, attention_mask=b_mask,
                    output_attentions=False, output_hidden_states=False,
                    return_dict=True,
                )

                logits = out.logits[:, :-1, :].contiguous().double()
                labels = b_ids[:, 1:].contiguous().clone()
                # MASK OUT PADDING: set padding labels to -100 (ignored by CE)
                label_mask = b_mask[:, 1:]
                labels[label_mask == 0] = -100

                loss = nn.CrossEntropyLoss(reduction='sum')(
                    logits.view(-1, logits.size(-1)), labels.view(-1)
                )
                n_real = (labels != -100).sum().item()
                total_loss += loss.item()
                total_tokens += n_real
        return total_loss / max(total_tokens, 1)

    # Baseline: all heads ON, no hooks
    baseline = _get_loss()
    print(f"    Baseline loss (real tokens only): {baseline:.6f}")

    importance = torch.zeros(num_layers, num_heads, device=device)

    for layer in range(num_layers):
        for head in range(num_heads):
            # Set mask: all ON except this head
            model.head_mask_module.log_alpha.data.fill_(10.0)
            model.head_mask_module.log_alpha.data[layer, head] = -10.0
            model.apply_pruning()

            ablated_loss = _get_loss()
            importance[layer, head] = ablated_loss - baseline

            model.remove_pruning()

        best = importance[layer].max().item()
        print(f"    Layer {layer:>2} done  (max Δloss: {best:.6f})")

    # Restore all heads ON
    model.head_mask_module.log_alpha.data.fill_(10.0)
    return importance


# ── Importance-Based Pruning ──────────────────────────────────────────────────

def prune_to_target(model, importance, target_active):
    """
    Set log_alpha so exactly `target_active` heads are ON,
    then install pruning hooks.
    """
    flat = importance.view(-1)
    total = flat.numel()
    target_active = min(target_active, total)

    _, sorted_idx = torch.sort(flat, descending=True)
    keep_set = set(sorted_idx[:target_active].tolist())

    new_log_alpha = torch.full_like(model.head_mask_module.log_alpha, -10.0)
    for idx in keep_set:
        layer = idx // importance.size(1)
        head  = idx %  importance.size(1)
        new_log_alpha[layer, head] = 10.0

    model.head_mask_module.log_alpha.data.copy_(new_log_alpha)
    model.apply_pruning()
    model.eval()


# ── Expert Head Selection ─────────────────────────────────────────────────────

def get_expert_heads(importance, top_k=10):
    """
    Select top-K heads with per-layer normalisation for layer diversity.
    """
    imp = importance.abs()
    layer_max = imp.max(dim=1, keepdim=True).values.clamp(min=1e-12)
    normalised = imp / layer_max

    flat = normalised.view(-1)
    k = min(top_k, flat.numel())
    _, topk_idx = torch.topk(flat, k)

    return [
        (idx.item() // importance.size(1), idx.item() % importance.size(1))
        for idx in topk_idx
    ]
