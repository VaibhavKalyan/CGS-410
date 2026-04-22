import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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
