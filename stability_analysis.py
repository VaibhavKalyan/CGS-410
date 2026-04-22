"""
stability_analysis.py — RQ4: Behavioral Stability.

Stability scoring, bootstrap confidence intervals, comparative analysis
across sentence types, and visualization.

COLAB FIXES:
  - attention_weights may be on GPU — explicit .cpu() before .item()/.numpy()
  - bootstrap_ci: handles n=0 and n=1 edge cases
  - comparative_analysis: handles empty groups gracefully
  - plot_stability: explicit fig close; fallback when no comparative data
  - print_stability_summary: handles missing comparative key safely
"""

import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Per-sentence stability score ──────────────────────────────────────────────

def compute_stability_score(attention_weights, dep_metadata, layer, head):
    """
    For a given (layer, head), return per-sentence 0/1 indicators of whether
    the head's top attention from the subject token lands on the verb token.

    Args:
        attention_weights : list of tensors indexed by layer,
                            each (batch, num_heads, seq, seq).  May be on GPU.
        dep_metadata      : list of dicts with 'subject_idx', 'verb_idx'.
        layer             : int
        head              : int
    Returns:
        scores : numpy array of 0/1, one entry per valid sentence.
    """
    aw        = attention_weights[layer]   # (B, H, S, S)
    batch_size = aw.size(0)
    seq_len    = aw.size(-1)
    scores     = []

    for b in range(batch_size):
        meta = dep_metadata[b] if b < len(dep_metadata) else None
        if meta is None or meta.get('subject_idx') is None:
            continue
        subj = meta['subject_idx']
        verb = meta['verb_idx']
        if subj >= seq_len or verb >= seq_len:
            continue

        top_key = aw[b, head, subj, :].argmax().cpu().item()
        scores.append(1.0 if top_key == verb else 0.0)

    return np.array(scores, dtype=float)


# ── Bootstrap confidence intervals ────────────────────────────────────────────

def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    """
    Compute bootstrap confidence interval for the mean of scores.

    Returns:
        dict with 'mean', 'variance', 'ci_lower', 'ci_upper'.
    """
    n = len(scores)
    if n == 0:
        return {'mean': 0.0, 'variance': 0.0,
                'ci_lower': 0.0, 'ci_upper': 0.0}
    if n == 1:
        v = float(scores[0])
        return {'mean': v, 'variance': 0.0, 'ci_lower': v, 'ci_upper': v}

    rng        = np.random.RandomState(42)
    boot_means = np.array([
        rng.choice(scores, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1.0 - ci) / 2.0
    return {
        'mean':     float(scores.mean()),
        'variance': float(scores.var()),
        'ci_lower': float(np.percentile(boot_means, alpha * 100)),
        'ci_upper': float(np.percentile(boot_means, (1.0 - alpha) * 100)),
    }


# ── Comparative analysis across sentence types ────────────────────────────────

def comparative_analysis(scores, sentence_types):
    """
    Split scores by sentence type and compute bootstrap stats per group.

    Args:
        scores         : numpy array of per-sentence stability scores.
        sentence_types : list of dicts with 'length' and 'complexity' keys.
    Returns:
        results : dict  group_name → bootstrap stats dict.
    """
    results = {}

    for length_tag in ['short', 'long']:
        indices = [
            i for i, st in enumerate(sentence_types)
            if st.get('length') == length_tag and i < len(scores)
        ]
        if indices:
            results[f'length={length_tag}'] = bootstrap_ci(scores[indices])

    for cplx_tag in ['simple', 'complex']:
        indices = [
            i for i, st in enumerate(sentence_types)
            if st.get('complexity') == cplx_tag and i < len(scores)
        ]
        if indices:
            results[f'complexity={cplx_tag}'] = bootstrap_ci(scores[indices])

    return results


# ── Full stability analysis ───────────────────────────────────────────────────

def run_stability_analysis(attention_weights, dep_metadata, sentence_types,
                           expert_heads, n_bootstrap=1000):
    """
    Run full stability analysis for a list of expert heads.

    Args:
        attention_weights : list of attention tensors per layer.
        dep_metadata      : list of dicts.
        sentence_types    : list of dicts.
        expert_heads      : list of (layer, head) tuples.
        n_bootstrap       : number of bootstrap samples.
    Returns:
        all_results : list of dicts.
    """
    all_results = []
    for (layer, head) in expert_heads:
        scores      = compute_stability_score(
            attention_weights, dep_metadata, layer, head
        )
        overall     = bootstrap_ci(scores, n_bootstrap=n_bootstrap)
        comparative = comparative_analysis(scores, sentence_types)

        all_results.append({
            'layer':       layer,
            'head':        head,
            'overall':     overall,
            'comparative': comparative,
            'n_sentences': len(scores),
        })

    return all_results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_stability(results, output_dir='outputs', suffix='', title_prefix=''):
    """Bar chart of per-head stability with 95% bootstrap CI error bars."""
    os.makedirs(output_dir, exist_ok=True)

    if not results:
        print("    No stability results to plot.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left: overall stability per expert head ---
    head_labels  = [f'L{r["layer"]}H{r["head"]}' for r in results]
    means        = [r['overall']['mean']      for r in results]
    err_lower    = [r['overall']['mean'] - r['overall']['ci_lower'] for r in results]
    err_upper    = [r['overall']['ci_upper'] - r['overall']['mean'] for r in results]

    x = np.arange(len(results))
    axes[0].bar(x, means, yerr=[err_lower, err_upper],
                capsize=4, color='steelblue', alpha=0.8, edgecolor='navy')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(head_labels, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('Stability Score')
    axes[0].set_title(f'{title_prefix}Expert Head Stability (95% CI)')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, axis='y', alpha=0.3)

    # --- Right: comparative breakdown for top expert ---
    comp = results[0].get('comparative', {}) if results else {}
    if comp:
        groups     = list(comp.keys())
        c_means    = [comp[g]['mean']      for g in groups]
        c_err_lo   = [comp[g]['mean'] - comp[g]['ci_lower'] for g in groups]
        c_err_hi   = [comp[g]['ci_upper'] - comp[g]['mean'] for g in groups]

        colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']
        x2 = np.arange(len(groups))
        axes[1].bar(x2, c_means, yerr=[c_err_lo, c_err_hi],
                    capsize=4, color=colors[:len(groups)],
                    alpha=0.8, edgecolor='k', linewidth=0.5)
        axes[1].set_xticks(x2)
        axes[1].set_xticklabels(groups, fontsize=9)
        axes[1].set_ylabel('Stability Score')
        head_lbl = f'L{results[0]["layer"]}H{results[0]["head"]}'
        axes[1].set_title(f'{title_prefix}Comparative Stability ({head_lbl})')
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, axis='y', alpha=0.3)
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(output_dir, f'stability_{suffix}.png')
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"    Saved: {fname}")
    return fname


# ── Print summary ─────────────────────────────────────────────────────────────

def print_stability_summary(results):
    """Pretty-print stability results table."""
    print(f"\n    {'Head':<10} {'Mean':>6} {'Var':>8} {'95% CI':>18} {'N':>5}")
    print("    " + "-" * 55)
    for r in results:
        o      = r['overall']
        ci_str = f"[{o['ci_lower']:.3f}, {o['ci_upper']:.3f}]"
        print(
            f"    L{r['layer']}H{r['head']:<6} "
            f"{o['mean']:>6.3f} {o['variance']:>8.5f} "
            f"{ci_str:>18} {r['n_sentences']:>5}"
        )

    if results and results[0].get('comparative'):
        print(
            f"\n    Comparative breakdown for "
            f"L{results[0]['layer']}H{results[0]['head']}:"
        )
        for group, stats in results[0]['comparative'].items():
            ci_str = f"[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]"
            print(f"      {group:<22} mean={stats['mean']:.3f}  CI={ci_str}")
