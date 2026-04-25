"""
RQ3 — Functional Taxonomy & Latent Dimensionality of Attention Heads
=====================================================================

Question A: Do attention heads naturally cluster into a small number
            of functional types (local, syntactic, delimiter, global)?

Question B: Is the variation in head behavior controlled by only a
            few latent dimensions?

NO PRUNING. This script analyses the UNPRUNED model only.

Methodology:
  1. Load 200+ sentences, run them through GPT-2 Medium.
  2. Extract attention matrices for every layer and head.
  3. Compute per-head summary features:
       - entropy, avg attention distance, prev/next/self-token attention,
         offset profile (-5 to +5), syntax-match rate, delimiter attention
  4. Standardise the feature matrix.
  5. PCA → plot variance explained + heads in PC1-PC2 space.
  6. K-Means + hierarchical clustering → label functional types.
  7. Check whether heads from similar layers cluster together.

Usage:
    python rq3_functional_taxonomy.py --lang en --max-sentences 200
    python rq3_functional_taxonomy.py --dry-run
"""

import os, sys, gc, argparse, warnings

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from data_loader import load_agreement_dataset, get_quick_test_data
from pruner_model import GPT2PrunerWrapper

OUTPUT_DIR = 'results_visualizes'

# ── Style ─────────────────────────────────────────────────────────────────────
_TEXT  = '#2C3E50'
_SPINE = '#BDC3C7'
_GRID  = '#E8ECF0'
_BG    = '#FAFAFA'


# ── Feature Extraction ───────────────────────────────────────────────────────

def compute_head_features(all_attentions, attention_mask, dep_metadata):
    """
    Compute per-head summary features from the unpruned model.

    Features per head:
      1. avg_entropy        — how diffuse vs focused the attention is
      2. avg_distance       — average absolute distance between query and key
      3. prev_token_attn    — proportion of attention on the previous token
      4. next_token_attn    — proportion of attention on the next token
      5. self_attn          — proportion of attention on the same token
      6. syntax_match       — does argmax(attn[subject,:]) land on the verb?
      7. delimiter_attn     — attention on padding / EOS tokens
      8. offset_-5 ... +5   — attention at each relative position
    """
    num_layers = len(all_attentions)
    batch_size, num_heads, seq_len, _ = all_attentions[0].shape
    features = []

    for layer_idx in range(num_layers):
        aw = all_attentions[layer_idx].float()  # (B, H, S, S)

        # 1. Entropy
        eps = 1e-12
        entropy = -torch.sum(aw * torch.log2(aw + eps), dim=-1)  # (B, H, S)
        avg_entropy = entropy.mean(dim=(0, 2)).numpy()  # (H,)

        # 2. Average attention distance
        pos = torch.arange(seq_len, dtype=torch.float32)
        # distance[q, k] = |q - k|
        dist_matrix = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()  # (S, S)
        avg_dist = (aw * dist_matrix.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (B, H, S)
        avg_distance = avg_dist.mean(dim=(0, 2)).numpy()  # (H,)

        # 3. Previous token attention
        if seq_len >= 2:
            q_idx = torch.arange(1, seq_len)
            k_idx = torch.arange(0, seq_len - 1)
            prev_attn = aw[:, :, q_idx, k_idx].mean(dim=(0, 2)).numpy()
        else:
            prev_attn = np.zeros(num_heads)

        # 4. Next token attention
        if seq_len >= 2:
            q_idx = torch.arange(0, seq_len - 1)
            k_idx = torch.arange(1, seq_len)
            next_attn = aw[:, :, q_idx, k_idx].mean(dim=(0, 2)).numpy()
        else:
            next_attn = np.zeros(num_heads)

        # 5. Self-token attention
        diag_idx = torch.arange(seq_len)
        self_attn = aw[:, :, diag_idx, diag_idx].mean(dim=(0, 2)).numpy()

        # 6. Syntax match rate (verb → subject, causal direction for GPT-2)
        syntax_match = np.zeros(num_heads)
        total_dep = 0
        if dep_metadata is not None:
            for b in range(batch_size):
                meta = dep_metadata[b] if b < len(dep_metadata) else None
                if meta is None:
                    continue
                subj = meta.get('subject_token_idx', meta.get('subject_idx'))
                verb = meta.get('verb_token_idx', meta.get('verb_idx'))
                if subj is None or verb is None:
                    continue
                if subj >= seq_len or verb >= seq_len:
                    continue
                if verb <= subj:
                    continue  # verb must come after subject for causal attn
                # VERB attending to SUBJECT (causal: verb looks back)
                top_key = aw[b, :, verb, :].argmax(dim=-1).numpy()
                syntax_match += (top_key == subj).astype(float)
                total_dep += 1
            if total_dep > 0:
                syntax_match /= total_dep

        # 7. Delimiter / special token attention
        delim_attn = np.zeros(num_heads)
        if attention_mask is not None:
            for b in range(batch_size):
                real_len = int(attention_mask[b].sum().item())
                if real_len < seq_len:
                    # attention to padding positions
                    delim_attn += aw[b, :, :, real_len:].sum(dim=-1).mean(dim=-1).numpy()
                # attention to last real token (often EOS)
                last_idx = max(0, real_len - 1)
                delim_attn += aw[b, :, :, last_idx].mean(dim=-1).numpy()
            delim_attn /= batch_size

        # 8. Offset profile (-5 to +5)
        offset_features = {}
        for offset in range(-5, 6):
            if seq_len <= 1:
                offset_features[offset] = np.zeros(num_heads)
                continue
            if offset >= 0:
                q_idx = torch.arange(0, max(1, seq_len - offset))
            else:
                q_idx = torch.arange(-offset, seq_len)
            k_idx = q_idx + offset
            valid = (k_idx >= 0) & (k_idx < seq_len) & (q_idx < seq_len)
            q_idx, k_idx = q_idx[valid], k_idx[valid]
            if len(q_idx) == 0:
                offset_features[offset] = np.zeros(num_heads)
            else:
                offset_features[offset] = aw[:, :, q_idx, k_idx].mean(dim=(0, 2)).numpy()

        # Build feature dict per head
        for h in range(num_heads):
            feat = {
                'layer': layer_idx, 'head': h,
                'avg_entropy': float(avg_entropy[h]),
                'avg_distance': float(avg_distance[h]),
                'prev_token_attn': float(prev_attn[h]),
                'next_token_attn': float(next_attn[h]),
                'self_attn': float(self_attn[h]),
                'syntax_match': float(syntax_match[h]),
                'delimiter_attn': float(delim_attn[h]),
            }
            for offset, vals in offset_features.items():
                feat[f'offset_{offset}'] = float(vals[h])
            features.append(feat)

    return features


def features_to_matrix(features):
    """Convert feature dicts to (n_heads, n_features) numpy matrix."""
    exclude = {'layer', 'head'}
    feature_names = sorted(k for k in features[0].keys() if k not in exclude)
    matrix = np.array([[f[k] for k in feature_names] for f in features])
    head_meta = [(f['layer'], f['head']) for f in features]
    return matrix, feature_names, head_meta


# ── PCA ───────────────────────────────────────────────────────────────────────

def run_pca(feature_matrix, n_components=None):
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix)
    n_comp = n_components or min(X.shape)
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(X)
    return {
        'transformed': transformed,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'components': pca.components_,
        'scaler': scaler,
        'pca_model': pca,
    }


# ── Clustering ────────────────────────────────────────────────────────────────

def run_clustering(feature_matrix, n_clusters=4):
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix)
    n_clusters = min(n_clusters, X.shape[0])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    return labels, km


# ── Plots ─────────────────────────────────────────────────────────────────────

def _style_ax(ax, title):
    ax.set_facecolor(_BG)
    ax.set_title(title, fontsize=13, fontweight='bold', color=_TEXT)
    ax.tick_params(colors=_TEXT)
    for s in ax.spines.values():
        s.set_color(_SPINE)
    ax.grid(True, alpha=0.3, color=_GRID)


def plot_taxonomy_scatter(pca_result, cluster_labels, head_meta, output_dir):
    """Scatter of heads in PC1-PC2, coloured by cluster and by layer depth."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    pc = pca_result['transformed']
    evr = pca_result['explained_variance_ratio']
    layers = np.array([m[0] for m in head_meta])

    # Panel 1: by cluster
    ax = axes[0]
    cluster_names = {0: 'Local/Positional', 1: 'Syntactic', 2: 'Delimiter', 3: 'Global/Diffuse'}
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6']
    for c in range(max(cluster_labels) + 1):
        mask = cluster_labels == c
        label = cluster_names.get(c, f'Cluster {c}')
        ax.scatter(pc[mask, 0], pc[mask, 1], c=colors[c % len(colors)],
                   s=40, alpha=0.7, edgecolors='white', linewidth=0.5, label=label)
    ax.set_xlabel(f'PC1 ({evr[0]*100:.1f}% var)', color=_TEXT)
    ax.set_ylabel(f'PC2 ({evr[1]*100:.1f}% var)', color=_TEXT)
    ax.legend(fontsize=9, framealpha=0.8)
    _style_ax(ax, 'Functional Taxonomy (by cluster)')

    # Panel 2: by layer depth
    ax = axes[1]
    scatter = ax.scatter(pc[:, 0], pc[:, 1], c=layers, cmap='viridis',
                         s=40, alpha=0.7, edgecolors='white', linewidth=0.5)
    plt.colorbar(scatter, ax=ax, label='Layer')
    ax.set_xlabel(f'PC1 ({evr[0]*100:.1f}% var)', color=_TEXT)
    ax.set_ylabel(f'PC2 ({evr[1]*100:.1f}% var)', color=_TEXT)
    _style_ax(ax, 'Head Distribution (by layer depth)')

    plt.tight_layout()
    path = os.path.join(output_dir, 'rq3_taxonomy_scatter.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def plot_variance_explained(pca_result, output_dir):
    """Scree plot: individual + cumulative variance explained."""
    evr = pca_result['explained_variance_ratio']
    cum = np.cumsum(evr)
    n = len(evr)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, n+1), evr * 100, alpha=0.6, color='#3498DB', label='Individual')
    ax.plot(range(1, n+1), cum * 100, 'o-', color='#E74C3C', linewidth=2, label='Cumulative')
    ax.axhline(y=90, linestyle='--', color='grey', alpha=0.5, label='90% threshold')
    ax.set_xlabel('Principal Component', color=_TEXT)
    ax.set_ylabel('Variance Explained (%)', color=_TEXT)
    ax.legend()
    _style_ax(ax, 'PCA Variance Explained — Are heads low-dimensional?')

    # Annotate how many PCs needed for 90%
    n_90 = np.searchsorted(cum, 0.90) + 1
    ax.annotate(f'{n_90} PCs explain 90%', xy=(n_90, 90),
                xytext=(n_90 + 2, 80), fontsize=10, color=_TEXT,
                arrowprops=dict(arrowstyle='->', color=_TEXT))

    plt.tight_layout()
    path = os.path.join(output_dir, 'rq3_variance_explained.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def plot_dendrogram(feature_matrix, head_meta, output_dir):
    """Hierarchical clustering dendrogram."""
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix)
    Z = linkage(X, method='ward')
    labels = [f'L{l}H{h}' for l, h in head_meta]

    fig, ax = plt.subplots(figsize=(20, 8))
    dendrogram(Z, labels=labels, ax=ax, leaf_rotation=90, leaf_font_size=5,
               color_threshold=0.7 * max(Z[:, 2]))
    _style_ax(ax, 'Hierarchical Clustering of Attention Heads')
    ax.set_ylabel('Ward Distance', color=_TEXT)

    plt.tight_layout()
    path = os.path.join(output_dir, 'rq3_dendrogram.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def print_cluster_summary(cluster_labels, head_meta, feature_matrix, feature_names):
    """Print summary table of cluster centroids and composition."""
    n_clusters = max(cluster_labels) + 1
    cluster_names = {0: 'Local/Positional', 1: 'Syntactic', 2: 'Delimiter', 3: 'Global/Diffuse'}

    print(f"\n  {'Cluster':<22} | {'Count':>5} | {'Layers':>20} | {'Top Features'}")
    print(f"  {'-'*22}-+-{'-'*5}-+-{'-'*20}-+-{'-'*30}")

    for c in range(n_clusters):
        mask = cluster_labels == c
        count = mask.sum()
        layers = [head_meta[i][0] for i in range(len(head_meta)) if mask[i]]
        layer_range = f'L{min(layers)}-L{max(layers)}'

        # Identify distinguishing features (highest centroid values)
        centroid = feature_matrix[mask].mean(axis=0)
        top_idx = np.argsort(np.abs(centroid))[-3:][::-1]
        top_feats = ', '.join(f'{feature_names[i]}={centroid[i]:.3f}' for i in top_idx)

        name = cluster_names.get(c, f'Cluster {c}')
        print(f"  {name:<22} | {count:>5} | {layer_range:>20} | {top_feats}")


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_rq3(lang, inputs, labels, dep_metadata, sentence_types,
            model, dry_run=False):
    lang_upper = lang.upper()
    device = model.device
    print(f"\n{'='*70}")
    print(f"  RQ3 — Functional Taxonomy  [{lang_upper}]  (device: {device})")
    print(f"  NO PRUNING — analysing the full unpruned model")
    print(f"{'='*70}")

    batch_size = 8 if not dry_run else 2
    n_samples  = inputs['input_ids'].size(0)
    seq_len    = inputs['input_ids'].size(1)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Forward pass to extract ALL attention matrices ────────
    print(f"\n  Step 1: Extracting attention from {n_samples} sentences "
          f"({model.num_layers} layers × {model.num_heads} heads = "
          f"{model.num_layers * model.num_heads} total heads)...")
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

    # ── Step 2: Compute per-head features ─────────────────────────────
    print("\n  Step 2: Computing per-head features (entropy, distance, "
          "prev/next/self attn, syntax match, offsets)...")
    features = compute_head_features(
        all_attentions, inputs['attention_mask'], dep_metadata
    )
    feat_matrix, feat_names, head_meta = features_to_matrix(features)
    print(f"    Feature matrix: {feat_matrix.shape[0]} heads × {feat_matrix.shape[1]} features")
    print(f"    Features: {', '.join(feat_names)}")

    # Save feature matrix as CSV
    df = pd.DataFrame(features)
    csv_path = os.path.join(OUTPUT_DIR, f'rq3_head_features_{lang}.csv')
    df.to_csv(csv_path, index=False)
    print(f"    Saved: {csv_path}")

    # ── Step 3: PCA ───────────────────────────────────────────────────
    print("\n  Step 3: PCA (standardised features)...")
    pca_result = run_pca(feat_matrix)
    evr = pca_result['explained_variance_ratio']
    cum = np.cumsum(evr)
    n_90 = np.searchsorted(cum, 0.90) + 1
    print(f"    PC1 explains {evr[0]*100:.1f}%, PC2 explains {evr[1]*100:.1f}%")
    print(f"    {n_90} PCs needed for 90% variance → "
          f"{'YES' if n_90 <= 5 else 'Partially'}, heads lie in a "
          f"{'low' if n_90 <= 5 else 'moderate'}-dimensional space")

    plot_variance_explained(pca_result, OUTPUT_DIR)

    # ── Step 4: K-Means clustering ────────────────────────────────────
    n_clusters = 4
    print(f"\n  Step 4: K-Means clustering (k={n_clusters})...")
    cluster_labels, km = run_clustering(feat_matrix, n_clusters=n_clusters)
    print_cluster_summary(cluster_labels, head_meta, feat_matrix, feat_names)

    # ── Step 5: Visualise ─────────────────────────────────────────────
    print("\n  Step 5: Generating plots...")
    plot_taxonomy_scatter(pca_result, cluster_labels, head_meta, OUTPUT_DIR)
    plot_dendrogram(feat_matrix, head_meta, OUTPUT_DIR)

    # ── Step 6: Layer clustering analysis ─────────────────────────────
    print("\n  Step 6: Do heads from similar layers cluster together?")
    layers = np.array([m[0] for m in head_meta])
    for c in range(n_clusters):
        mask = cluster_labels == c
        c_layers = layers[mask]
        name = {0: 'Local', 1: 'Syntactic', 2: 'Delimiter', 3: 'Global'}.get(c, f'C{c}')
        print(f"    {name:>12}: layers {np.mean(c_layers):.1f} ± {np.std(c_layers):.1f}  "
              f"(range L{c_layers.min()}-L{c_layers.max()}, n={len(c_layers)})")

    del all_attentions
    gc.collect()
    torch.cuda.empty_cache()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    if any('ipykernel' in a or 'json' in a for a in sys.argv[1:]):
        sys.argv = sys.argv[:1]

    parser = argparse.ArgumentParser(
        description='RQ3: Functional Taxonomy (no pruning)')
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

    run_rq3(args.lang, inputs, labels, dep_metadata, sentence_types,
            model, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
