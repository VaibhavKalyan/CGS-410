"""
functional_taxonomy.py — RQ3: Functional Taxonomy & Latent Space.

PCA dimensionality reduction, K-means / hierarchical clustering,
and visualization of head functional types.

COLAB FIXES:
  - plt.cm.get_cmap deprecated → matplotlib.colormaps[] with fallback
  - plot_dendrogram: leaf labels truncated to avoid Colab memory on large models
  - All plt.close() calls made explicit to prevent figure leakage
  - n_components capped at min(n_samples, n_features) for tiny dry-run datasets
  - KMeans n_clusters capped at n_samples to avoid sklearn crash
  - Added try/except around dendrogram for robustness
"""

import os
import numpy as np
import warnings

import matplotlib
matplotlib.use('Agg')          # Non-interactive — required on Colab
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


# Canonical cluster labels (index → human-readable name)
CLUSTER_LABELS = {0: 'Local', 1: 'Syntactic', 2: 'Delimiter', 3: 'Global'}


# ── PCA ───────────────────────────────────────────────────────────────────────

def run_pca(feature_matrix, n_components=None):
    """
    Standardise features and run PCA.

    Args:
        feature_matrix : numpy (n_heads, n_features)
        n_components   : number of components (None = all valid)
    Returns:
        dict with 'components', 'explained_variance_ratio',
                  'transformed', 'scaler', 'pca_model'.
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)

    # Cap n_components at the maximum sklearn allows
    max_comp = min(feature_matrix.shape)
    if n_components is None or n_components > max_comp:
        n_components = max_comp

    pca         = PCA(n_components=n_components)
    transformed = pca.fit_transform(X_scaled)

    return {
        'components':              pca.components_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'transformed':             transformed,
        'scaler':                  scaler,
        'pca_model':               pca,
    }


# ── Clustering ────────────────────────────────────────────────────────────────

def run_clustering(feature_matrix, n_clusters=4, method='kmeans'):
    """
    Cluster heads into functional types.

    Args:
        feature_matrix : numpy (n_heads, n_features)
        n_clusters     : number of clusters (auto-capped at n_samples)
        method         : 'kmeans' or 'hierarchical'
    Returns:
        labels : numpy array of cluster assignments
        model  : fitted clustering model
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)

    # Guard: cannot have more clusters than samples
    n_clusters = min(n_clusters, feature_matrix.shape[0])

    if method == 'kmeans':
        model  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
    elif method == 'hierarchical':
        model  = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X_scaled)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'kmeans' or 'hierarchical'.")

    return labels, model


# ── Helper: safe colormap ─────────────────────────────────────────────────────

def _get_cmap(name, n):
    """Return a colormap, compatible with old and new matplotlib."""
    try:
        # matplotlib ≥ 3.7
        return matplotlib.colormaps[name].resampled(n)
    except AttributeError:
        # matplotlib < 3.7
        return plt.cm.get_cmap(name, n)


# ── Plot: taxonomy scatter ────────────────────────────────────────────────────

def plot_taxonomy(pca_result, cluster_labels, head_metadata,
                  output_dir='outputs', suffix='', title_prefix=''):
    """
    Scatter of heads in PC1-PC2 space coloured by cluster and by layer depth.
    """
    os.makedirs(output_dir, exist_ok=True)
    transformed = pca_result['transformed']
    layers      = np.array([m[0] for m in head_metadata])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left: coloured by cluster ---
    n_clusters = len(set(cluster_labels))
    cmap       = _get_cmap('tab10', n_clusters)
    for c in range(n_clusters):
        mask  = cluster_labels == c
        label = CLUSTER_LABELS.get(c, f'Cluster {c}')
        axes[0].scatter(
            transformed[mask, 0], transformed[mask, 1],
            color=[cmap(c)], label=label,
            alpha=0.7, s=30, edgecolors='k', linewidths=0.3,
        )
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title(f'{title_prefix}Functional Taxonomy (by Cluster)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # --- Right: coloured by layer depth ---
    sc = axes[1].scatter(
        transformed[:, 0], transformed[:, 1],
        c=layers, cmap='viridis', alpha=0.7, s=30,
        edgecolors='k', linewidths=0.3,
    )
    plt.colorbar(sc, ax=axes[1], label='Layer')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title(f'{title_prefix}Functional Taxonomy (by Layer Depth)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(output_dir, f'taxonomy_{suffix}.png')
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"    Saved: {fname}")
    return fname


# ── Plot: variance explained ──────────────────────────────────────────────────

def plot_variance_explained(pca_result, output_dir='outputs',
                            suffix='', title_prefix=''):
    """Scree plot of individual and cumulative variance explained."""
    os.makedirs(output_dir, exist_ok=True)
    evr        = pca_result['explained_variance_ratio']
    cumulative = np.cumsum(evr)
    n          = len(evr)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, n + 1), evr, alpha=0.6, label='Individual')
    ax.plot(range(1, n + 1), cumulative, 'ro-', markersize=4, label='Cumulative')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained')
    ax.set_title(f'{title_prefix}PCA Variance Explained')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, min(20, n) + 0.5)

    plt.tight_layout()
    fname = os.path.join(output_dir, f'variance_explained_{suffix}.png')
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"    Saved: {fname}")
    return fname


# ── Plot: dendrogram ──────────────────────────────────────────────────────────

def plot_dendrogram(feature_matrix, head_metadata,
                    output_dir='outputs', suffix='', title_prefix=''):
    """Hierarchical clustering dendrogram (optional visual aid)."""
    os.makedirs(output_dir, exist_ok=True)

    # With 288 heads (24 layers × 16 heads in GPT-2 Medium) the dendrogram
    # gets crowded; truncate labels for readability on Colab
    try:
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(feature_matrix)
        Z        = linkage(X_scaled, method='ward')

        n_heads      = len(head_metadata)
        head_labels  = [f'L{m[0]}H{m[1]}' for m in head_metadata]
        leaf_fontsize = max(2, 6 - n_heads // 50)   # shrink font for many heads

        fig, ax = plt.subplots(figsize=(max(14, n_heads // 5), 6))
        dendrogram(
            Z, labels=head_labels, ax=ax,
            leaf_rotation=90, leaf_font_size=leaf_fontsize,
            truncate_mode='lastp' if n_heads > 100 else None,
            p=50 if n_heads > 100 else None,
        )
        ax.set_title(f'{title_prefix}Hierarchical Clustering Dendrogram')
        ax.set_ylabel('Ward Distance')

        plt.tight_layout()
        fname = os.path.join(output_dir, f'dendrogram_{suffix}.png')
        plt.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"    Saved: {fname}")
        return fname

    except Exception as e:
        print(f"    [WARNING] Dendrogram failed: {e}. Skipping.")
        plt.close('all')
        return None


# ── Cluster summary table ─────────────────────────────────────────────────────

def summarize_clusters(cluster_labels, head_metadata, feature_matrix, feature_names):
    """Print summary table of cluster centroids and member counts."""
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)
    n_clusters = len(set(cluster_labels))

    print(f"\n    {'Cluster':<12} {'Count':>6}  Top-3 Features (standardized centroid)")
    print("    " + "-" * 65)
    for c in range(n_clusters):
        mask     = cluster_labels == c
        count    = mask.sum()
        centroid = X_scaled[mask].mean(axis=0)
        label    = CLUSTER_LABELS.get(c, f'Cluster {c}')

        top_idx  = np.argsort(np.abs(centroid))[-3:][::-1]
        top_str  = ', '.join(
            f'{feature_names[i]}={centroid[i]:+.2f}' for i in top_idx
        )
        print(f"    {label:<12} {count:>6}  {top_str}")
