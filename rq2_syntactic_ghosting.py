"""
RQ2 — Syntactic Ghosting in the Residual Stream
================================================

Does grammar signal persist ("ghost") in the residual stream
even after the attention heads responsible for syntax have been
pruned? Measured via:
  - Participation Ratio: how many dimensions carry signal
    (lower = grammar compressed into fewer dimensions)
  - Logistic Probe: can a linear classifier still decode
    singular/plural from the residual stream deltas?

Methodology:
  1. Compute ablation importance (which heads matter most).
  2. At each pruning milestone (48→30→20→10→5 active heads),
     prune via hooks, then extract hidden states.
  3. For each layer, compute the residual delta (h_l - h_{l-1}).
  4. Participation Ratio via SVD on the delta matrix.
  5. Train logistic probe to classify number (singular/plural).

Usage:
    python rq2_syntactic_ghosting.py --lang en --max-sentences 200
    python rq2_syntactic_ghosting.py --dry-run
"""

import os, sys, gc, argparse, warnings

import torch
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from data_loader import load_agreement_dataset, get_quick_test_data
from pruner_model import GPT2PrunerWrapper, compute_head_importance, prune_to_target
from ghosting_probes import analyze_all_layers, plot_rq2_ghosting

MILESTONES = [300, 250, 200, 150, 100, 50]
OUTPUT_DIR = 'results_visualizes'


def run_rq2(lang, inputs, labels, dep_metadata, sentence_types,
            model, milestones, dry_run=False):
    lang_u = lang.upper()
    device = model.device
    batch_size = 8 if not dry_run else 2
    n_samples  = inputs['input_ids'].size(0)
    seq_len    = inputs['input_ids'].size(1)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  RQ2 — Syntactic Ghosting  [{lang_u}]  (device: {device})")
    print(f"{'='*70}")

    # ── Step 1: Compute head importance (ablation, ~10 min) ───────────
    print("\n  Step 1: Computing head importance (ablation)...")
    importance = compute_head_importance(model, inputs, batch_size=batch_size)

    all_results = []

    # ── Step 2: Baseline (unpruned) ───────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  [{lang_u}] Baseline: {model.num_layers * model.num_heads} Active Heads (unpruned)")
    print(f"{'─'*60}")

    model.remove_pruning()
    model.eval()
    baseline_hidden = [
        torch.empty(n_samples, seq_len, model.config.n_embd, dtype=torch.float16)
        for _ in range(model.num_layers + 1)
    ]
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            b_ids  = inputs['input_ids'][i:i+batch_size].to(device)
            b_mask = inputs['attention_mask'][i:i+batch_size].to(device)
            out = model.model(
                input_ids=b_ids, attention_mask=b_mask,
                output_attentions=False, output_hidden_states=True, return_dict=True,
            )
            for l, hs in enumerate(out.hidden_states):
                baseline_hidden[l][i:i+batch_size] = hs.cpu().half()
            del out
            torch.cuda.empty_cache()

    ghosting = analyze_all_layers(baseline_hidden, inputs['attention_mask'], labels)
    baseline_ms = model.num_layers * model.num_heads
    for gr in ghosting:
        all_results.append({'lang': lang, 'milestone': baseline_ms, **gr})
    for gr in ghosting[:5]:
        print(f"    Layer {gr['layer']:>2} | PR: {gr['participation_ratio']:>8.2f} "
              f"| Probe: {gr['probe_test_accuracy']*100:>6.2f}%")
    del baseline_hidden
    gc.collect()
    torch.cuda.empty_cache()

    # ── Step 3: Pruning milestones ────────────────────────────────────
    for milestone in milestones:
        print(f"\n{'─'*60}")
        print(f"  [{lang_u}] Milestone: {milestone} Active Heads")
        print(f"{'─'*60}")

        prune_to_target(model, importance, target_active=milestone)
        n_active = model.head_mask_module.get_active_heads()
        print(f"  Active heads: {n_active} / {model.num_layers * model.num_heads}")

        # Forward pass with pruning hooks active
        print("  Extracting hidden states (pruned model)...")
        all_hidden = [
            torch.empty(n_samples, seq_len, model.config.n_embd, dtype=torch.float16)
            for _ in range(model.num_layers + 1)
        ]
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                b_ids  = inputs['input_ids'][i:i+batch_size].to(device)
                b_mask = inputs['attention_mask'][i:i+batch_size].to(device)
                out = model.model(
                    input_ids=b_ids, attention_mask=b_mask,
                    output_attentions=False, output_hidden_states=True, return_dict=True,
                )
                for l, hs in enumerate(out.hidden_states):
                    all_hidden[l][i:i+batch_size] = hs.cpu().half()
                del out
                torch.cuda.empty_cache()

        # Ghosting analysis
        print(f"\n  [RQ2] Syntactic Ghosting (M={milestone}):")
        ghosting = analyze_all_layers(all_hidden, inputs['attention_mask'], labels)
        for gr in ghosting:
            all_results.append({'lang': lang, 'milestone': milestone, **gr})
        for gr in ghosting[:5]:
            print(f"    Layer {gr['layer']:>2} | PR: {gr['participation_ratio']:>8.2f} "
                  f"| Probe: {gr['probe_test_accuracy']*100:>6.2f}%")

        del all_hidden
        gc.collect()
        torch.cuda.empty_cache()

    model.remove_pruning()

    # ── Save & plot ───────────────────────────────────────────────────
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(OUTPUT_DIR, f'rq2_ghosting_{lang}.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved CSV: {csv_path}")
        print("  Generating RQ2 ghosting plot...")
        plot_rq2_ghosting(df, output_dir=OUTPUT_DIR, lang=lang)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    if any('ipykernel' in a or 'json' in a for a in sys.argv[1:]):
        sys.argv = sys.argv[:1]
    parser = argparse.ArgumentParser(description='RQ2: Syntactic Ghosting')
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
    run_rq2(args.lang, inputs, labels, dep_metadata, sentence_types,
            model, milestones, dry_run=args.dry_run)

if __name__ == '__main__':
    main()
