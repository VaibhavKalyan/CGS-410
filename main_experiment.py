"""
main_experiment.py — Full experimental pipeline.
COLAB OOM FIXES APPLIED:
  - Mini-batching for all forward passes.
  - Float16 pre-allocation on CPU to halve RAM consumption.
  - Aggressive garbage collection and VRAM clearing.
"""

import os
import sys
import argparse
import warnings
import gc

import torch
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from data_loader import load_agreement_dataset, get_quick_test_data
from pruner_model import (
    GPT2PrunerWrapper, train_l0_pruning, compute_integrated_gradients, get_expert_heads,
)
from attention_metrics import extract_head_features, features_to_matrix
from ghosting_probes import analyze_all_layers
from functional_taxonomy import (
    run_pca, run_clustering, plot_taxonomy, plot_variance_explained, plot_dendrogram, summarize_clusters,
)
from stability_analysis import run_stability_analysis, plot_stability, print_stability_summary

MILESTONES  = [48, 30, 20, 10, 5]
OUTPUT_DIR  = 'outputs'

def run_pipeline(lang, inputs, labels, dep_metadata, sentence_types, model, milestones, dry_run=False):
    lang_upper = lang.upper()
    device = model.device
    print(f"\n{'='*70}\n  EXPERIMENT — {lang_upper}  (device: {device})\n{'='*70}")

    all_rq1_results = []
    all_rq2_results = []
    n_samples = inputs['input_ids'].size(0)
    seq_len = inputs['input_ids'].size(1)
    batch_size = 8 if not dry_run else 2

    for milestone in milestones:
        print(f"\n{'─'*60}\n  [{lang_upper}] Milestone: {milestone} Active Heads\n{'─'*60}")

        if dry_run:
            total_heads = model.num_layers * model.num_heads
            n_on = min(milestone, total_heads)
            mask_flat = torch.zeros(total_heads, device=device)
            mask_flat[:n_on] = 1.0
            mask_flat = mask_flat[torch.randperm(total_heads, device=device)]
            mask_2d = mask_flat.view(model.num_layers, model.num_heads)
            model.head_mask_module.log_alpha.data = torch.where(mask_2d == 1.0, 10.0, -10.0)
            model.eval()
        else:
            print(f"  Training L₀ gates → {milestone} active heads...")
            train_l0_pruning(model, inputs, target_active=milestone, epochs=30, lr=0.02, lambda_l0=1.5, batch_size=batch_size)

        # ── VRAM-Safe Forward Pass ────────────────────────────────────
        print("  Extracting attentions and hidden states (Batched & Float16)...")
        model.eval()
        
        # Pre-allocate on CPU in half-precision to save RAM
        all_attentions = [torch.empty((n_samples, model.num_heads, seq_len, seq_len), dtype=torch.float16) for _ in range(model.num_layers)]
        all_hidden_states = [torch.empty((n_samples, seq_len, model.config.n_embd), dtype=torch.float16) for _ in range(model.num_layers + 1)]
        
        head_mask = model.head_mask_module(training=False).detach().clone()

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                b_inputs = {k: v[i:i+batch_size].to(device) for k, v in inputs.items()}
                outputs = model.model(
                    input_ids=b_inputs['input_ids'],
                    attention_mask=b_inputs['attention_mask'],
                    head_mask=head_mask,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                for l, attn in enumerate(outputs.attentions):
                    all_attentions[l][i:i+batch_size] = attn.cpu().half()
                for l, hs in enumerate(outputs.hidden_states):
                    all_hidden_states[l][i:i+batch_size] = hs.cpu().half()
                    
                del outputs, b_inputs
                torch.cuda.empty_cache()

        # ── Integrated Gradients ──────────────────────────────────────
        print("  Computing Integrated Gradients attributions...")
        ig_attributions = compute_integrated_gradients(model, inputs, labels, dep_metadata, n_steps=5 if dry_run else 15, batch_size=batch_size)
        top_k = max(1, min(10, milestone))
        expert_heads = get_expert_heads(ig_attributions, top_k=top_k)
        print(f"  Top-{top_k} Expert Heads: " + ", ".join(f"L{l}H{h}" for l, h in expert_heads))

        if not expert_heads:
            continue

        # ── RQ1: Functional Resilience ────────────────────────────────
        print("\n  [RQ1] Functional Resilience:")
        features = extract_head_features(all_attentions, seq_lengths=inputs['attention_mask'].sum(dim=1), dep_metadata=dep_metadata, attention_mask=inputs['attention_mask'])
        
        for (el, eh) in expert_heads[:5]:
            feat = next((f for f in features if f['layer'] == el and f['head'] == eh), None)
            if feat is None: continue
            is_active = head_mask[el, eh].item() > 0.5
            status = "ACTIVE" if is_active else "PRUNED"
            all_rq1_results.append({'lang': lang, 'milestone': milestone, 'layer': el, 'head': eh, 'status': status, 'entropy': feat['avg_entropy'], 'attn_to_target': feat['attn_to_target'], 'syntax_match': feat['syntax_match']})
            print(f"    L{el}H{eh} [{status}] | Entropy: {feat['avg_entropy']:.4f} | Syntax Match: {feat['syntax_match']:.4f}")

        # ── RQ2: Syntactic Ghosting ───────────────────────────────────
        print("\n  [RQ2] Syntactic Ghosting (all layers):")
        ghosting_results = analyze_all_layers(all_hidden_states, inputs['attention_mask'], labels)
        for gr in ghosting_results:
            all_rq2_results.append({'lang': lang, 'milestone': milestone, **gr})
        for gr in ghosting_results[:3]:
            print(f"    Layer {gr['layer']:>2} | PR: {gr['participation_ratio']:>8.2f} | Probe: {gr['probe_test_accuracy']*100:>6.2f}%")

        # ── RQ3: Functional Taxonomy ──────────────────────────────────
        print("\n  [RQ3] Functional Taxonomy:")
        feat_matrix, feat_names, head_meta = features_to_matrix(features)
        pca_result = run_pca(feat_matrix)
        cluster_labels, _ = run_clustering(feat_matrix, n_clusters=min(4, feat_matrix.shape[0]))
        suffix = f"{lang}_{milestone}"
        
        plot_taxonomy(pca_result, cluster_labels, head_meta, output_dir=OUTPUT_DIR, suffix=suffix, title_prefix=f'[{lang_upper} M={milestone}] ')
        plot_variance_explained(pca_result, output_dir=OUTPUT_DIR, suffix=suffix, title_prefix=f'[{lang_upper} M={milestone}] ')

        # ── RQ4: Behavioral Stability ─────────────────────────────────
        print("\n  [RQ4] Behavioral Stability:")
        stability_results = run_stability_analysis(all_attentions, dep_metadata, sentence_types, expert_heads[:5], n_bootstrap=200 if dry_run else 1000)
        print_stability_summary(stability_results)
        plot_stability(stability_results, output_dir=OUTPUT_DIR, suffix=suffix, title_prefix=f'[{lang_upper} M={milestone}] ')

        # Aggressive cleanup
        del all_attentions, all_hidden_states, features
        gc.collect()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if all_rq1_results: pd.DataFrame(all_rq1_results).to_csv(os.path.join(OUTPUT_DIR, f'rq1_resilience_{lang}.csv'), index=False)
    if all_rq2_results: pd.DataFrame(all_rq2_results).to_csv(os.path.join(OUTPUT_DIR, f'rq2_ghosting_{lang}.csv'), index=False)


def main():
    if any('ipykernel' in a or 'json' in a for a in sys.argv[1:]):
        sys.argv = sys.argv[:1]

    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--lang', type=str, default=None, choices=['en', 'ru'])
    parser.add_argument('--milestones', type=str, default=None)
    parser.add_argument('--max-sentences', type=int, default=200) # Lowered default for Colab safety
    args = parser.parse_args()

    milestones = [int(x.strip()) for x in args.milestones.split(',')] if args.milestones else MILESTONES
    languages = ['en', 'ru'] if args.lang is None else [args.lang]

    model = GPT2PrunerWrapper('gpt2-medium')
    model.eval()

    for lang in languages:
        if args.dry_run:
            inputs, labels, dep_metadata, sentence_types = get_quick_test_data()
            run_milestones = [milestones[0], milestones[-1]]
        else:
            inputs, labels, dep_metadata, sentence_types = load_agreement_dataset(lang=lang, max_sentences=args.max_sentences)
            run_milestones = milestones

        run_pipeline(lang, inputs, labels, dep_metadata, sentence_types, model, run_milestones, dry_run=args.dry_run)

if __name__ == '__main__':
    main()
