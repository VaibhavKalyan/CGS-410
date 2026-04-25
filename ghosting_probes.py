import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Shared style constants
_TEXT  = '#2C3E50'
_SPINE = '#BDC3C7'
_GRID  = '#E8ECF0'
_BG    = '#FAFAFA'
# Blue → Red progression maps least-pruned → most-pruned milestones
_MILESTONE_PALETTE = {384: '#1A1A2E', 300: '#16213E', 250: '#0F3460',
                      200: '#4C72B0', 150: '#55A868', 100: '#F5A623',
                      50: '#DD8452',  48: '#4C72B0', 30: '#55A868',
                      20: '#F5A623', 10: '#DD8452',  5: '#C44E52'}

def compute_participation_ratio(matrix):
    matrix = matrix.float()
    centred = matrix - matrix.mean(dim=0, keepdim=True)
    try:
        _, s, _ = torch.linalg.svd(centred, full_matrices=False)
    except AttributeError:
        _, s, _ = torch.svd(centred, some=True)

    eigenvalues = s ** 2
    tr_C, tr_C2 = eigenvalues.sum(), (eigenvalues ** 2).sum()
    return (tr_C ** 2 / tr_C2).item() if tr_C2.item() != 0.0 else 0.0

class SyntacticGhostingProbe:
    def __init__(self, max_iter=500, test_size=0.2, random_state=42):
        self.max_iter, self.test_size, self.random_state = max_iter, test_size, random_state

    def train_and_evaluate(self, delta_h, labels):
        self.probe = LogisticRegression(max_iter=self.max_iter, solver='saga', n_jobs=1)
        if len(np.unique(labels)) < 2 or len(labels) < 5:
            self.probe.fit(delta_h, labels)
            self.train_acc = self.test_acc = accuracy_score(labels, self.probe.predict(delta_h))
            return self.test_acc

        X_tr, X_te, y_tr, y_te = train_test_split(delta_h, labels, test_size=self.test_size, random_state=self.random_state, stratify=labels)
        self.probe.fit(X_tr, y_tr)
        self.train_acc = accuracy_score(y_tr, self.probe.predict(X_tr))
        self.test_acc = accuracy_score(y_te, self.probe.predict(X_te))
        return self.test_acc

def analyze_all_layers(hidden_states, attention_mask, labels):
    num_layers = len(hidden_states) - 1
    seq_lengths = attention_mask.sum(dim=1).long() - 1
    max_pos = hidden_states[0].size(1) - 1
    seq_lengths = seq_lengths.clamp(0, max_pos)
    batch_idx = torch.arange(hidden_states[0].size(0))
    results = []

    for l in range(1, num_layers + 1):
        # Cast float16 CPU tensors to float32 before math
        hl = hidden_states[l][batch_idx, seq_lengths, :].float()
        hl_prev = hidden_states[l - 1][batch_idx, seq_lengths, :].float()
        delta_h = hl - hl_prev

        pr = compute_participation_ratio(delta_h)
        delta_h_np = delta_h.numpy()
        
        probe = SyntacticGhostingProbe()
        test_acc = probe.train_and_evaluate(delta_h_np, labels)

        results.append({
            'layer': l, 'participation_ratio': pr,
            'probe_test_accuracy': test_acc, 'probe_train_accuracy': probe.train_acc,
        })
    return results


# ── RQ2 Plot ──────────────────────────────────────────────────────────────────

def plot_rq2_ghosting(df, output_dir='outputs', lang='en'):
    """
    Two-panel line chart answering RQ2:
      Left  — Participation Ratio per layer, one line per pruning milestone.
               A falling PR across milestones means grammar is being compressed
               into fewer residual-stream dimensions ("ghosting").
      Right — Probe accuracy per layer, one line per milestone.
               Staying above chance shows grammar is still decodable despite compression.
    """
    os.makedirs(output_dir, exist_ok=True)

    milestones = sorted(df['milestone'].unique())   # ascending: 5, 10, 20, 30, 48
    if not milestones:
        print("    [RQ2] No ghosting data to plot.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')
    fig.patch.set_facecolor('white')
    fig.suptitle(
        f'[{lang.upper()}]  RQ2 — Syntactic Ghosting in the Residual Stream',
        fontsize=14, fontweight='bold', color=_TEXT, y=1.02,
    )

    specs = [
        ('participation_ratio',  'Participation Ratio',
         'Grammar Compression  (↓ PR = more compressed)'),
        ('probe_test_accuracy',  'Probe Test Accuracy',
         'Grammar Decodability  (↑ = grammar still readable)'),
    ]

    for ax, (metric, ylabel, title) in zip(axes, specs):
        ax.set_facecolor(_BG)

        for m in milestones:
            sub   = df[df['milestone'] == m].sort_values('layer')
            color = _MILESTONE_PALETTE.get(m, '#888')
            ax.plot(sub['layer'], sub[metric],
                    color=color, linewidth=2.1, marker='o', markersize=5,
                    markeredgecolor='white', markeredgewidth=0.9,
                    label=f'{m} heads', zorder=3, alpha=0.92)

        ax.set_title(title, fontsize=12, fontweight='bold', color=_TEXT, pad=12)
        ax.set_xlabel('Transformer Layer', fontsize=11, color=_TEXT, labelpad=6)
        ax.set_ylabel(ylabel, fontsize=11, color=_TEXT, labelpad=6)
        ax.grid(True, color=_GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(_SPINE)
        ax.spines['bottom'].set_color(_SPINE)
        ax.tick_params(colors=_TEXT, labelsize=9)

        leg = ax.legend(fontsize=9, title='Active heads', title_fontsize=9,
                        framealpha=0.92, edgecolor=_SPINE, loc='best')
        leg.get_title().set_color(_TEXT)

    # Chance-level reference line on probe accuracy panel
    axes[1].axhline(0.5, color='#999', linewidth=1.1, linestyle='--',
                    alpha=0.6, zorder=2)
    axes[1].text(df['layer'].max() * 0.97, 0.507, 'chance',
                 fontsize=8.5, color='#999', ha='right', style='italic')
    axes[1].set_ylim(bottom=0.0)

    plt.tight_layout(pad=2.5)
    fname = os.path.join(output_dir, f'rq2_ghosting_plot_{lang}.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {fname}")
    return fname
