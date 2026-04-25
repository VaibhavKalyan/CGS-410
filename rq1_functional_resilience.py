"""
RQ1 — Functional Resilience of Expert Heads Under Pruning
=========================================================

Do "expert" attention heads — those that perform syntactic
subject-verb linking — retain their specialisation as other
heads are progressively pruned?

Methodology:
  1. Run the UNPRUNED model to identify expert heads (top-10 by
     verb→subject attention alignment, with per-layer diversity).
  2. Compute per-head ABLATION IMPORTANCE (loss increase when
     head is zeroed out; padding-aware, float64).
  3. At each pruning milestone (48→30→20→10→5 active heads),
     keep the most important heads, prune the rest via hooks.
  4. Track each expert head's entropy and syntax-match rate.

Key design decisions:
  - verb→subject direction (GPT-2 is causal: verb comes after
    subject, so verb CAN attend back but subject CANNOT attend fwd)
  - Hook-based pruning (head_mask arg may be silently ignored)
  - Padding tokens excluded from loss computation

Usage:
    python rq1_functional_resilience.py --lang en --max-sentences 200
    python rq1_functional_resilience.py --dry-run
"""

import os, sys, gc, argparse, warnings
from collections import Counter

import torch
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from data_loader import load_agreement_dataset, get_quick_test_data
from pruner_model import (
    GPT2PrunerWrapper, compute_head_importance,
    prune_to_target, get_expert_heads,
)

MILESTONES = [300, 250, 200, 150, 100, 50]
OUTPUT_DIR = 'results_visualizes'
N_EXPERTS  = 10

_TEXT  = '#2C3E50'
_SPINE = '#BDC3C7'
_GRID  = '#E8ECF0'
_BG    = '#FAFAFA'


# ── Head Feature Computation ─────────────────────────────────────────────────

def compute_entropy(aw):
    """Per-head entropy averaged over sentences and query positions."""
    eps = 1e-12
    ent = -torch.sum(aw * torch.log2(aw + eps), dim=-1)  # (B, H, S)
    return ent.mean(dim=(0, 2)).numpy()  # (H,)


def compute_syntax_match(aw, dep_metadata, seq_len):
    """Verb→subject syntax-match rate (causal direction)."""
    batch_size, num_heads = aw.shape[0], aw.shape[1]
    match = np.zeros(num_heads)
    total = 0
    for b in range(batch_size):
        meta = dep_metadata[b] if b < len(dep_metadata) else None
        if meta is None:
            continue
        subj = meta.get('subject_token_idx', meta.get('subject_idx'))
        verb = meta.get('verb_token_idx', meta.get('verb_idx'))
        if subj is None or verb is None:
            continue
        if subj >= seq_len or verb >= seq_len or verb <= subj:
            continue
        top_key = aw[b, :, verb, :].argmax(dim=-1).numpy()
        match += (top_key == subj).astype(float)
        total += 1
    return match / total if total > 0 else np.zeros(num_heads), total


def identify_experts_by_syntax(all_attentions, dep_metadata, top_k=10):
    """Identify expert heads by verb→subject alignment score, with layer diversity."""
    num_layers = len(all_attentions)
    batch_size, num_heads, seq_len, _ = all_attentions[0].shape
    scores = np.zeros((num_layers, num_heads))
    total = 0
    for b in range(batch_size):
        meta = dep_metadata[b] if b < len(dep_metadata) else None
        if meta is None:
            continue
        subj = meta.get('subject_token_idx', meta.get('subject_idx'))
        verb = meta.get('verb_token_idx', meta.get('verb_idx'))
        if subj is None or verb is None or subj >= seq_len or verb >= seq_len or verb <= subj:
            continue
        total += 1
        for l in range(num_layers):
            top_key = all_attentions[l][b].float()[:, verb, :].argmax(dim=-1).numpy()
            scores[l] += (top_key == subj).astype(float)
    if total > 0:
        scores /= total

    # Per-layer normalisation for diversity
    scores_t = torch.tensor(scores)
    layer_max = scores_t.max(dim=1, keepdim=True).values.clamp(min=1e-12)
    normed = scores_t / layer_max
    _, topk_idx = torch.topk(normed.view(-1), min(top_k, normed.numel()))
    experts = [(idx.item() // num_heads, idx.item() % num_heads) for idx in topk_idx]
    return experts, scores


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_rq1(df, output_dir, lang):
    """Two-panel line chart: entropy and syntax-match vs milestone."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')
    fig.suptitle(f'[{lang.upper()}]  RQ1 — Functional Resilience Under Pruning',
                 fontsize=14, fontweight='bold', color=_TEXT, y=1.02)

    for ax, metric, ylabel, title in [
        (axes[0], 'entropy', 'Attention Entropy (bits)',
         'Specialisation  (↓ = more focused)'),
        (axes[1], 'syntax_match', 'Verb→Subject Hit Rate',
         'Syntax Alignment  (↑ = head still does syntax)'),
    ]:
        ax.set_facecolor(_BG)
        milestones = sorted(df['milestone'].unique())
        heads = df.groupby(['layer', 'head']).first().index.tolist()

        # Individual head traces (grey)
        for (l, h) in heads:
            sub = df[(df['layer'] == l) & (df['head'] == h)].sort_values('milestone')
            ax.plot(sub['milestone'], sub[metric], color='#BDC3C7',
                    linewidth=0.8, alpha=0.5, zorder=1)

        # Mean ± std
        grouped = df.groupby('milestone')[metric]
        means = grouped.mean()
        stds  = grouped.std().fillna(0)
        ax.plot(means.index, means.values, 'o-', color='#2980B9',
                linewidth=2.5, markersize=7, zorder=3, label='Mean')
        ax.fill_between(means.index, means - stds, means + stds,
                        alpha=0.2, color='#2980B9', zorder=2)

        ax.set_xlabel('Active Heads', color=_TEXT)
        ax.set_ylabel(ylabel, color=_TEXT)
        ax.set_title(title, fontsize=12, fontweight='bold', color=_TEXT, pad=10)
        ax.invert_xaxis()
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, color=_GRID)
        for s in ax.spines.values():
            s.set_color(_SPINE)

    plt.tight_layout(pad=2.5)
    path = os.path.join(output_dir, f'rq1_resilience_plot_{lang}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_rq1(lang, inputs, labels, dep_metadata, sentence_types,
            model, milestones, dry_run=False):
    lang_u = lang.upper()
    device = model.device
    batch_size = 8 if not dry_run else 2
    n_experts  = 5 if dry_run else N_EXPERTS
    n_samples  = inputs['input_ids'].size(0)
    seq_len    = inputs['input_ids'].size(1)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  RQ1 — Functional Resilience  [{lang_u}]  (device: {device})")
    print(f"{'='*70}")

    # ── Step 1: Single forward pass (unpruned) to get all attention ───
    print(f"\n  Step 1: Forward pass (unpruned, {n_samples} sentences)...")
    model.eval()
    model.remove_pruning()
    all_attentions = [
        torch.empty(n_samples, model.num_heads, seq_len, seq_len, dtype=torch.float16)
        for _ in range(model.num_layers)
    ]
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            b_ids  = inputs['input_ids'][i:i+batch_size].to(device)
            b_mask = inputs['attention_mask'][i:i+batch_size].to(device)
            out = model.model(
                input_ids=b_ids, attention_mask=b_mask,
                output_attentions=True, output_hidden_states=False, return_dict=True,
            )
            for l, attn in enumerate(out.attentions):
                all_attentions[l][i:i+batch_size] = attn.cpu().half()
            del out
            torch.cuda.empty_cache()

    # ── Step 2: Identify expert heads by syntax alignment ─────────────
    print("\n  Step 2: Identifying expert heads (verb→subject attention)...")
    expert_heads, syntax_scores = identify_experts_by_syntax(
        all_attentions, dep_metadata, top_k=n_experts)
    print(f"  Expert Heads ({n_experts}): " +
          ", ".join(f"L{l}H{h} ({syntax_scores[l,h]:.3f})" for l, h in expert_heads))
    layer_dist = Counter(l for l, _ in expert_heads)
    print(f"  Layer Diversity: { {f'L{k}': v for k, v in sorted(layer_dist.items())} }")

    # Compute baseline features for expert heads (unpruned)
    print("\n  Step 2b: Baseline features (unpruned)...")
    all_results = []
    for (el, eh) in expert_heads:
        aw_layer = all_attentions[el].float()
        ent = compute_entropy(aw_layer)
        sm, _ = compute_syntax_match(aw_layer, dep_metadata, seq_len)
        all_results.append({
            'lang': lang, 'milestone': model.num_layers * model.num_heads,
            'layer': el, 'head': eh, 'status': 'ACTIVE',
            'entropy': float(ent[eh]), 'syntax_match': float(sm[eh]),
        })
        print(f"    L{el}H{eh} | Entropy: {ent[eh]:.4f} | Syntax Match: {sm[eh]:.4f}")

    del all_attentions
    gc.collect()
    torch.cuda.empty_cache()

    # ── Step 3: Compute head importance (ablation-based) ──────────────
    print("\n  Step 3: Computing head importance (ablation, ~10 min)...")
    importance = compute_head_importance(model, inputs, batch_size=batch_size)

    # ── Step 4: Pruning milestones ────────────────────────────────────
    for milestone in milestones:
        print(f"\n{'─'*60}")
        print(f"  [{lang_u}] Milestone: {milestone} Active Heads")
        print(f"{'─'*60}")

        prune_to_target(model, importance, target_active=milestone)
        n_active = model.head_mask_module.get_active_heads()
        mask = model.head_mask_module(training=False)
        print(f"  Active heads: {n_active} / {model.num_layers * model.num_heads}")

        # Forward pass WITH pruning hooks active
        print("  Extracting attention weights (pruned model)...")
        pruned_attentions = [
            torch.empty(n_samples, model.num_heads, seq_len, seq_len, dtype=torch.float16)
            for _ in range(model.num_layers)
        ]
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                b_ids  = inputs['input_ids'][i:i+batch_size].to(device)
                b_mask = inputs['attention_mask'][i:i+batch_size].to(device)
                out = model.model(
                    input_ids=b_ids, attention_mask=b_mask,
                    output_attentions=True, output_hidden_states=False, return_dict=True,
                )
                for l, attn in enumerate(out.attentions):
                    pruned_attentions[l][i:i+batch_size] = attn.cpu().half()
                del out
                torch.cuda.empty_cache()

        # Compute features for expert heads
        print(f"\n  Expert Head Status at M={milestone}:")
        for (el, eh) in expert_heads:
            aw_layer = pruned_attentions[el].float()
            ent = compute_entropy(aw_layer)
            sm, _ = compute_syntax_match(aw_layer, dep_metadata, seq_len)
            status = "ACTIVE" if mask[el, eh].item() > 0.5 else "PRUNED"

            all_results.append({
                'lang': lang, 'milestone': milestone,
                'layer': el, 'head': eh, 'status': status,
                'entropy': float(ent[eh]), 'syntax_match': float(sm[eh]),
            })
            print(f"    L{el}H{eh} [{status}] | Entropy: {ent[eh]:.4f} "
                  f"| Syntax Match: {sm[eh]:.4f}")

        del pruned_attentions
        gc.collect()
        torch.cuda.empty_cache()

    model.remove_pruning()

    # ── Save & plot ───────────────────────────────────────────────────
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(OUTPUT_DIR, f'rq1_resilience_{lang}.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved CSV: {csv_path}")
        plot_rq1(df, OUTPUT_DIR, lang)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    if any('ipykernel' in a or 'json' in a for a in sys.argv[1:]):
        sys.argv = sys.argv[:1]
    parser = argparse.ArgumentParser(description='RQ1: Functional Resilience')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--lang', type=str, default='en', choices=['en', 'ru'])
    parser.add_argument('--milestones', type=str, default=None)
    parser.add_argument('--max-sentences', type=int, default=200)
    args = parser.parse_args()
    milestones = ([int(x.strip()) for x in args.milestones.split(',')]
                  if args.milestones else MILESTONES)
    model = GPT2PrunerWrapper('gpt2-medium')
    model.eval()
    if args.dry_run:
        inputs, labels, dep_metadata, sentence_types = get_quick_test_data()
        milestones = [milestones[0], milestones[-1]]
    else:
        inputs, labels, dep_metadata, sentence_types = load_agreement_dataset(
            lang=args.lang, max_sentences=args.max_sentences)
    run_rq1(args.lang, inputs, labels, dep_metadata, sentence_types,
            model, milestones, dry_run=args.dry_run)

if __name__ == '__main__':
    main()
