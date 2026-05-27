# Syntactic Resilience and Signal Leakage in Heavily Pruned Transformer Residual Streams

> **CGS 410 — Computational Linguistics**
> Department of Cognitive Science, Indian Institute of Technology Kanpur
> Supervised by **Prof. Himanshu Yadav**

---

## Authors

| Name | Roll No. |
|------|----------|
| Anshuman Raj | 240148 |
| Vaibhav Kalyan | 241125 |

---

## Abstract

This project investigates where subject–verb agreement is encoded in GPT-2 Medium under progressive attention head pruning. Using 995 valid subject–verb aligned sentences drawn from the [HuggingFace BLiMP dataset](https://huggingface.co/datasets/nyu-mll/blimp), we run four linked analyses spanning expert head resilience, linear decodability from residual stream updates (syntactic ghosting), low-dimensional functional taxonomy, and behavioral stability. Across all analyses, agreement information remains recoverable under strong pruning, supporting a **distributed residual-stream account** rather than a strictly head-local account.

---

## Research Questions

| ID | Question |
|----|----------|
| **RQ1** | Which attention heads act as "experts" for verb→subject routing, and how resilient are they under pruning? |
| **RQ2** | Does number information remain linearly decodable from residual updates ∆h when most heads are removed? |
| **RQ3** | Do attention heads exhibit a low-dimensional functional taxonomy? |
| **RQ4** | Are expert behaviors stable under bootstrap resampling, and how does sentence difficulty affect syntax-hit rates? |

---

## Key Findings

- **RQ1 — Expert Head Resilience:** Top expert L2H9 achieves a syntax-match of 0.805 at baseline, remaining above 0.82 even at M = 50 active heads (out of 384). Expert specialization is distributed across distinct layers.
- **RQ2 — Syntactic Ghosting:** Layer-3 probe accuracy declines only from 96.98% → 91.46% as active heads drop from 384 → 50, while the participation ratio collapses from 25.04 → 2.82, demonstrating that agreement information persists in a dramatically compressed subspace.
- **RQ3 — Functional Taxonomy:** Four PCA-derived clusters emerge (Syntactic 248, Local/Positional 59, Global/Diffuse 57, Delimiter 20). Just 4 principal components capture 90% of variance, confirming a low-dimensional functional space.
- **RQ4 — Behavioral Stability:** Bootstrap CIs (B = 1000) confirm reliable expert rankings. A clear length effect is observed (short syntax-hit 0.433 vs. long 0.348), consistent with dependency-distance theory.

---

## Dataset

This project uses the **BLiMP (Benchmark of Linguistic Minimal Pairs)** dataset, accessed via HuggingFace:

```python
from datasets import load_dataset
dataset = load_dataset("nyu-mll/blimp", "subject_verb_agreement")
```

> Warstadt, A., et al. (2020). BLiMP: The Benchmark of Linguistic Minimal Pairs for English. *TACL*, 8, 377–392.

From 1000 requested examples, **995 valid subject–verb aligned sentences** are retained after Stanza dependency filtering and GPT-2 token alignment.

---

## Model

**GPT-2 Medium** — 24 layers, 16 heads/layer, hidden size 1024, head size 64, 384 total attention heads.

---

## Repository Structure

```
CGS-410/
├── data_loader.py               # BLiMP loading, Stanza parsing, GPT-2 token alignment
├── pruner_model.py              # Ablation scoring and pruning hooks
├── rq1_functional_resilience.py # RQ1: Expert head tracking under pruning
├── rq2_syntactic_ghosting.py    # RQ2: Residual stream probing
├── ghosting_probes.py           # RQ2: Linear probe utilities
├── rq3_functional_taxonomy.py   # RQ3: PCA and k-means clustering
├── rq4_behavioral_stability.py  # RQ4: Bootstrap stability analysis
├── attention_metrics.py         # RQ4: Attention metric utilities
├── related_papers/              # Background reading and project proposal
│   ├── 2021.cmcl-1.6.pdf
│   └── Project_Proposal_*.pdf
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- HuggingFace `transformers` and `datasets`
- Stanza
- scikit-learn, numpy, matplotlib

### Installation

```bash
git clone https://github.com/VaibhavKalyan/CGS-410.git
cd CGS-410
pip install -r requirements.txt
```

### Reproducing Results

Run each research question independently:

```bash
# RQ1 — Expert head tracking
python rq1_functional_resilience.py --lang en --max-sentences 1000

# RQ2 — Syntactic ghosting / residual probing
python rq2_syntactic_ghosting.py --lang en --max-sentences 1000

# RQ3 — Functional taxonomy (PCA + clustering)
python rq3_functional_taxonomy.py --lang en --max-sentences 1000

# RQ4 — Behavioral stability and transitions
python rq4_behavioral_stability.py --lang en --max-sentences 1000
```

---

## References

- Warstadt, A., et al. (2020). BLiMP: The Benchmark of Linguistic Minimal Pairs for English. *TACL*, 8, 377–392.
- Voita, E., et al. (2019). Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned. *ACL*, 5797–5808.
- Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits. *Transformer Circuits Thread*.
- Clark, K., et al. (2019). What Does BERT Look At? *BlackboxNLP*, 276–286.
- Gibson, E. (2000). The Dependency Locality Theory. *Image, Language, Brain*, 95–126.
- Goldberg, Y. (2019). Assessing BERT's Syntactic Abilities. *arXiv:1901.07931*.

---

## Acknowledgements

We thank **Prof. Himanshu Yadav** (Department of Cognitive Science, IIT Kanpur) for his guidance and supervision throughout this course project.

---

## License

This repository is for academic use only as part of CGS 410, IIT Kanpur.
