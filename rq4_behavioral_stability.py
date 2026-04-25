"""
RQ4 — Behavioral Stability & Transition Analysis of Attention Heads
====================================================================

Question A: Are specialized attention behaviors stable across sentences?
            Does a "syntactic" head behave consistently, or only on average?

Question B: Can attention be interpreted as a stochastic transition matrix?
            Do some heads resemble simple movement rules (nearest-neighbor,
            random-walk, syntax-jump)?

NO PRUNING. This script analyses the UNPRUNED model only.

IMPORTANT: GPT-2 is a CAUSAL model — each token can only attend to
tokens BEFORE it. Since the verb typically comes AFTER the subject,
we measure verb→subject attention (does the verb look back at the
subject?), not subject→verb.

Usage:
    python rq4_behavioral_stability.py --lang en --max-sentences 200
    python rq4_behavioral_stability.py --dry-run
"""

import os, sys, gc, argparse, warnings
from collections import Counter

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from data_loader import load_agreement_dataset, get_quick_test_data
from pruner_model import GPT2PrunerWrapper

OUTPUT_DIR = 'results_visualizes'
N_EXPERTS  = 10

_TEXT  = '#2C3E50'
_SPINE = '#BDC3C7'
_GRID  = '#E8ECF0'
_BG    = '#FAFAFA'


def _style_ax(ax, title):
    ax.set_facecolor(_BG)
    ax.set_title(title, fontsize=12, fontweight='bold', color=_TEXT, pad=10)
    ax.tick_params(colors=_TEXT, labelsize=9)
    for s in ax.spines.values():
        s.set_color(_SPINE)
    ax.grid(True, alpha=0.3, color=_GRID)


# ══════════════════════════════════════════════════════════════════════════════
#  PART A — Stability Analysis
# ══════════════════════════════════════════════════════════════════════════════

def identify_expert_heads(all_attentions, dep_metadata, top_k=10):
    """
    Find top-K heads by syntax alignment score.

    Because GPT-2 is CAUSAL, the verb (which comes after the subject)
    CAN attend back to the subject, but the subject CANNOT attend forward
    to the verb. So we check: does argmax(attn[verb, :]) == subject?
    """
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
        if subj is None or verb is None or subj >= seq_len or verb >= seq_len:
            continue
        # verb must come AFTER subject for causal attention to work
        if verb <= subj:
            continue
        total += 1
        for l in range(num_layers):
            aw = all_attentions[l][b].float()  # (H, S, S)
            # VERB attending to SUBJECT (causal direction)
            top_key = aw[:, verb, :].argmax(dim=-1).numpy()  # (H,)
            scores[l] += (top_key == subj).astype(float)

    if total > 0:
        scores /= total
    print(f"    {total} valid subject→verb pairs (verb after subject)")

    # Top-K with layer diversity
    scores_t = torch.tensor(scores)
    layer_max = scores_t.max(dim=1, keepdim=True).values.clamp(min=1e-12)
    normed = scores_t / layer_max
    flat = normed.view(-1)
    _, topk_idx = torch.topk(flat, min(top_k, flat.numel()))
    experts = [(idx.item() // num_heads, idx.item() % num_heads) for idx in topk_idx]

    return experts, scores


def compute_per_sentence_scores(all_attentions, dep_metadata, expert_heads):
    """
    Per expert head, per sentence:
      - syntax_hit: 1 if argmax(attn[verb,:]) == subject, else 0
      - dep_mass:   attention weight attn[verb, subject]
    """
    batch_size = all_attentions[0].shape[0]
    seq_len    = all_attentions[0].shape[2]
    results = []

    for b in range(batch_size):
        meta = dep_metadata[b] if b < len(dep_metadata) else None
        if meta is None:
            continue
        subj = meta.get('subject_token_idx', meta.get('subject_idx'))
        verb = meta.get('verb_token_idx', meta.get('verb_idx'))
        if subj is None or verb is None or subj >= seq_len or verb >= seq_len:
            continue
        if verb <= subj:
            continue

        for (el, eh) in expert_heads:
            aw = all_attentions[el][b, eh].float()  # (S, S)
            top_key = aw[verb, :].argmax().item()
            hit = 1.0 if top_key == subj else 0.0
            mass = aw[verb, subj].item()

            results.append({
                'sentence': b, 'layer': el, 'head': eh,
                'syntax_hit': hit, 'dep_mass': mass,
            })

    return results


def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    scores = np.array(scores)
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        means.append(sample.mean())
    means = np.sort(means)
    lo = means[int((1 - ci) / 2 * n_bootstrap)]
    hi = means[int((1 + ci) / 2 * n_bootstrap)]
    return scores.mean(), lo, hi


def stability_analysis(per_sentence, expert_heads, sentence_types,
                       dep_metadata, n_bootstrap=1000):
    """Bootstrap CI + sentence-type breakdown."""
    df = pd.DataFrame(per_sentence)
    if df.empty:
        print("    [WARNING] No valid sentences for stability analysis.")
        return df

    print(f"\n  {'Head':<8} | {'Mean':>6} | {'95% CI':>16} | {'Var':>8} | {'N':>4}")
    print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*16}-+-{'-'*8}-+-{'-'*4}")

    for (el, eh) in expert_heads:
        sub = df[(df['layer'] == el) & (df['head'] == eh)]
        if sub.empty:
            continue
        hits = sub['syntax_hit'].values
        mean, lo, hi = bootstrap_ci(hits, n_bootstrap)
        var = np.var(hits)
        print(f"  L{el}H{eh:<4} | {mean:>6.3f} | [{lo:.3f}, {hi:.3f}] | {var:>8.4f} | {len(hits):>4}")

    # ── Sentence-type breakdown (short/long, simple/complex) ──────────
    if sentence_types and len(sentence_types) > 0:
        # sentence_types is a list of dicts: {'length': 'short'/'long', 'complexity': 'simple'/'complex'}
        max_sent = int(df['sentence'].max()) + 1

        # By length
        print(f"\n  Breakdown by sentence LENGTH:")
        for length_tag in ['short', 'long']:
            valid_sents = set()
            for s in range(min(max_sent, len(sentence_types))):
                st = sentence_types[s]
                if isinstance(st, dict) and st.get('length') == length_tag:
                    valid_sents.add(s)
            if not valid_sents:
                continue
            sub = df[df['sentence'].isin(valid_sents)]
            if sub.empty:
                continue
            mean_hit = sub['syntax_hit'].mean()
            mean_mass = sub['dep_mass'].mean()
            print(f"    {length_tag:<10}: mean syntax-hit = {mean_hit:.3f}, "
                  f"mean dep-mass = {mean_mass:.4f}  (n={len(sub)})")

        # By complexity
        print(f"\n  Breakdown by sentence COMPLEXITY:")
        for comp_tag in ['simple', 'complex']:
            valid_sents = set()
            for s in range(min(max_sent, len(sentence_types))):
                st = sentence_types[s]
                if isinstance(st, dict) and st.get('complexity') == comp_tag:
                    valid_sents.add(s)
            if not valid_sents:
                continue
            sub = df[df['sentence'].isin(valid_sents)]
            if sub.empty:
                continue
            mean_hit = sub['syntax_hit'].mean()
            mean_mass = sub['dep_mass'].mean()
            print(f"    {comp_tag:<10}: mean syntax-hit = {mean_hit:.3f}, "
                  f"mean dep-mass = {mean_mass:.4f}  (n={len(sub)})")

    return df


def plot_stability(per_sentence, expert_heads, output_dir):
    """Bar chart with 95% CI error bars."""
    df = pd.DataFrame(per_sentence)
    if df.empty:
        return

    means, los, his, labels = [], [], [], []
    for (el, eh) in expert_heads:
        sub = df[(df['layer'] == el) & (df['head'] == eh)]
        if sub.empty:
            continue
        hits = sub['syntax_hit'].values
        m, lo, hi = bootstrap_ci(hits)
        means.append(m)
        los.append(m - lo)
        his.append(hi - m)
        labels.append(f'L{el}H{eh}')

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=[los, his], capsize=4, color='#3498DB',
           edgecolor='white', linewidth=0.5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Syntax-Hit Rate (verb→subject)', color=_TEXT)
    ax.set_ylim(0, max(max(means) * 1.3, 0.1) if means else 0.1)
    _style_ax(ax, 'RQ4A: Expert Head Stability (Bootstrap 95% CI)')

    plt.tight_layout()
    path = os.path.join(output_dir, 'rq4_stability.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  PART B — Transition Matrix Analysis
# ══════════════════════════════════════════════════════════════════════════════

def compute_transition_features(all_attentions, attention_mask):
    """
    Treat each head's attention as a transition matrix.
    Compute per-head (averaged over sentences):
      - expected_jump:  E[|q - k|]
      - self_loop:      P(attending to self)
      - move_right:     P(k > q)  — impossible in causal model (will be ~0)
      - move_left:      P(k < q)  — backward attention (dominant in GPT-2)
      - concentration:  1 - entropy/log2(seq_len)
    """
    num_layers = len(all_attentions)
    batch_size, num_heads, seq_len, _ = all_attentions[0].shape
    features = []

    pos = torch.arange(seq_len, dtype=torch.float32)
    dist_mat = (pos.unsqueeze(0) - pos.unsqueeze(1)).float()
    abs_dist = dist_mat.abs()
    # In causal GPT-2: upper triangle is masked, only lower triangle + diagonal
    lower = (dist_mat < 0).float()  # move left (attend to earlier tokens)
    diag_mask = torch.eye(seq_len)
    max_entropy = np.log2(max(seq_len, 2))

    for l in range(num_layers):
        aw = all_attentions[l].float()  # (B, H, S, S)

        # Mask out padding
        if attention_mask is not None:
            real_mask = attention_mask.float()
            mask_2d = real_mask.unsqueeze(1).unsqueeze(-1) * real_mask.unsqueeze(1).unsqueeze(2)
            aw = aw * mask_2d
            row_sum = aw.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            aw = aw / row_sum

        exp_jump = (aw * abs_dist.unsqueeze(0).unsqueeze(0)).sum(-1).mean(dim=(0, 2)).numpy()
        self_loop = (aw * diag_mask.unsqueeze(0).unsqueeze(0)).sum(-1).mean(dim=(0, 2)).numpy()
        left = (aw * lower.unsqueeze(0).unsqueeze(0)).sum(-1).mean(dim=(0, 2)).numpy()

        eps = 1e-12
        ent = -torch.sum(aw * torch.log2(aw + eps), dim=-1).mean(dim=(0, 2)).numpy()
        conc = 1.0 - ent / max_entropy

        for h in range(num_heads):
            features.append({
                'layer': l, 'head': h,
                'expected_jump': float(exp_jump[h]),
                'self_loop': float(self_loop[h]),
                'move_left': float(left[h]),
                'concentration': float(conc[h]),
            })

    return features


def make_synthetic_rules(seq_len):
    """Causal-compatible synthetic transition matrices."""
    rules = {}

    # Nearest-neighbor (previous token)
    prev = torch.zeros(seq_len, seq_len)
    for i in range(1, seq_len):
        prev[i, i-1] = 1.0
    prev[0, 0] = 1.0
    rules['previous_token'] = prev

    # Uniform backward: attend equally to all earlier tokens
    uni = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(i + 1):
            uni[i, j] = 1.0 / (i + 1)
    rules['uniform_backward'] = uni

    # Self-loop: always attend to self
    rules['self_loop'] = torch.eye(seq_len)

    # Syntax-jump: attend to a token 2-5 positions back (mimics dep. structure)
    syn = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        target = max(0, i - 3)  # ~3 positions back on average
        syn[i, target] = 1.0
    rules['syntax_jump'] = syn

    return rules


def compare_with_synthetic(trans_features, seq_len, output_dir):
    """Compare each real head with synthetic transition rules."""
    synthetic = make_synthetic_rules(seq_len)

    def _feat_from_matrix(tm):
        pos = torch.arange(seq_len, dtype=torch.float32)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        exp_j = (tm * dist).sum(dim=-1).mean().item()
        self_l = tm.diagonal().mean().item()
        return np.array([exp_j, self_l])

    syn_feats = {name: _feat_from_matrix(m) for name, m in synthetic.items()}

    results = []
    for tf in trans_features:
        head_vec = np.array([tf['expected_jump'], tf['self_loop']])
        best_rule = min(syn_feats.keys(), key=lambda r: np.linalg.norm(head_vec - syn_feats[r]))
        dist = np.linalg.norm(head_vec - syn_feats[best_rule])
        results.append({**tf, 'best_synthetic': best_rule, 'dist_to_synthetic': float(dist)})

    df = pd.DataFrame(results)
    print("\n  Synthetic Rule Matches:")
    for rule in synthetic:
        count = (df['best_synthetic'] == rule).sum()
        pct = count / len(df) * 100
        print(f"    {rule:<18}: {count:>3} heads ({pct:.1f}%)")

    return df


def plot_transition_analysis(df_trans, output_dir):
    """PCA scatter of heads in transition-feature space."""
    feat_cols = ['expected_jump', 'self_loop', 'move_left', 'concentration']
    X = df_trans[feat_cols].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xs)
    evr = pca.explained_variance_ratio_

    layers = df_trans['layer'].values

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: by layer
    ax = axes[0]
    sc = ax.scatter(Xp[:, 0], Xp[:, 1], c=layers, cmap='viridis',
                    s=30, alpha=0.7, edgecolors='white', linewidth=0.4)
    plt.colorbar(sc, ax=ax, label='Layer')
    ax.set_xlabel(f'PC1 ({evr[0]*100:.1f}%)', color=_TEXT)
    ax.set_ylabel(f'PC2 ({evr[1]*100:.1f}%)', color=_TEXT)
    _style_ax(ax, 'Transition Features (by layer)')

    # Panel 2: by best synthetic rule
    ax = axes[1]
    rule_colors = {'previous_token': '#E74C3C', 'uniform_backward': '#3498DB',
                   'self_loop': '#2ECC71', 'syntax_jump': '#9B59B6'}
    for rule, color in rule_colors.items():
        mask = df_trans['best_synthetic'] == rule
        if mask.any():
            ax.scatter(Xp[mask, 0], Xp[mask, 1], c=color, s=30, alpha=0.7,
                       edgecolors='white', linewidth=0.4, label=rule.replace('_', ' '))
    ax.set_xlabel(f'PC1 ({evr[0]*100:.1f}%)', color=_TEXT)
    ax.set_ylabel(f'PC2 ({evr[1]*100:.1f}%)', color=_TEXT)
    ax.legend(fontsize=9, framealpha=0.8)
    _style_ax(ax, 'Transition Features (by best synthetic rule)')

    plt.tight_layout()
    path = os.path.join(output_dir, 'rq4_transition_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_rq4(lang, inputs, labels, dep_metadata, sentence_types,
            model, dry_run=False):
    lang_upper = lang.upper()
    device = model.device
    print(f"\n{'='*70}")
    print(f"  RQ4 — Behavioral Stability & Transition Analysis  [{lang_upper}]")
    print(f"  NO PRUNING — analysing the full unpruned model")
    print(f"{'='*70}")

    batch_size = 8 if not dry_run else 2
    n_experts  = 5 if dry_run else N_EXPERTS
    n_samples  = inputs['input_ids'].size(0)
    seq_len    = inputs['input_ids'].size(1)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Forward pass ──────────────────────────────────────────────────
    print(f"\n  Extracting attention from {n_samples} sentences...")
    model.eval()
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
                output_attentions=True, output_hidden_states=False,
                return_dict=True,
            )
            for l, attn in enumerate(out.attentions):
                all_attentions[l][i:i+batch_size] = attn.cpu().half()
            del out
            torch.cuda.empty_cache()
    print("    Done.")

    # ══════════════════════════════════════════════════════════════════
    #  PART A — Stability
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  PART A: Behavioral Stability")
    print(f"{'─'*60}")

    print("\n  Step A1: Identifying expert heads (verb→subject attention)...")
    expert_heads, all_scores = identify_expert_heads(
        all_attentions, dep_metadata, top_k=n_experts)
    print(f"  Expert Heads ({n_experts}): " +
          ", ".join(f"L{l}H{h} ({all_scores[l,h]:.3f})" for l, h in expert_heads))

    layer_dist = Counter(l for l, _ in expert_heads)
    print(f"  Layer Diversity: { {f'L{k}': v for k, v in sorted(layer_dist.items())} }")

    print("\n  Step A2: Computing per-sentence syntax scores...")
    per_sentence = compute_per_sentence_scores(
        all_attentions, dep_metadata, expert_heads)
    print(f"    {len(per_sentence)} per-sentence × per-head scores")

    n_boot = 200 if dry_run else 1000
    print(f"\n  Step A3: Bootstrap analysis ({n_boot} samples):")
    df_stab = stability_analysis(
        per_sentence, expert_heads, sentence_types, dep_metadata, n_boot)

    print("\n  Step A4: Generating stability plot...")
    plot_stability(per_sentence, expert_heads, OUTPUT_DIR)

    if per_sentence:
        csv_path = os.path.join(OUTPUT_DIR, f'rq4_stability_{lang}.csv')
        pd.DataFrame(per_sentence).to_csv(csv_path, index=False)
        print(f"    Saved: {csv_path}")

    # ══════════════════════════════════════════════════════════════════
    #  PART B — Transition Analysis
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  PART B: Transition Matrix Analysis")
    print(f"{'─'*60}")

    print("\n  Step B1: Computing transition features for all heads...")
    trans_features = compute_transition_features(
        all_attentions, inputs['attention_mask'])
    print(f"    Features: expected_jump, self_loop, move_left, concentration")

    print("\n  Step B2: Comparing with synthetic transition rules...")
    df_trans = compare_with_synthetic(trans_features, seq_len, OUTPUT_DIR)

    print("\n  Step B3: Generating transition analysis plots...")
    plot_transition_analysis(df_trans, OUTPUT_DIR)

    csv_path = os.path.join(OUTPUT_DIR, f'rq4_transition_{lang}.csv')
    df_trans.to_csv(csv_path, index=False)
    print(f"    Saved: {csv_path}")

    del all_attentions
    gc.collect()
    torch.cuda.empty_cache()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    if any('ipykernel' in a or 'json' in a for a in sys.argv[1:]):
        sys.argv = sys.argv[:1]

    parser = argparse.ArgumentParser(
        description='RQ4: Behavioral Stability & Transition Analysis (no pruning)')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--lang', type=str, default='en', choices=['en', 'ru'])
    parser.add_argument('--max-sentences', type=int, default=200)
    args = parser.parse_args()

    model = GPT2PrunerWrapper('gpt2-medium')
    model.eval()

    if args.dry_run:
        inputs, labels, dep_metadata, sentence_types = get_quick_test_data()
    else:
        inputs, labels, dep_metadata, sentence_types = load_agreement_dataset(
            lang=args.lang, max_sentences=args.max_sentences,
        )

    run_rq4(args.lang, inputs, labels, dep_metadata, sentence_types,
            model, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
