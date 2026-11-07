"""End-to-end pipeline orchestrator.

Stages (mirroring Figure 2.1):
  1. Prompt construction
  2. Activation caching
  3. Probing
  4. Behavioural testing
  5. Suppression sweep
  6. Diagnostics
  7. Verdict

Run:  python -m eval_awareness.run_pipeline
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List
import numpy as np

from .config import Config
from . import prompts as prompts_mod
from .probes import (
    train_layer_probes,
    difference_of_means_direction,
    score_with_probe,
    predict_per_example,
)
from .behaviour import measure_gap, KeywordRefusalClassifier
from .suppression import summarise_sweep
from .diagnostics import run_diagnostics


def _indices_for(pair_ids: List[int], wanted: set) -> np.ndarray:
    return np.array([i for i, pid in enumerate(pair_ids) if pid in wanted], dtype=int)


def build_dataset(cfg: Config):
    """Stage 1: prompt construction, with the pair invariant enforced."""
    unsafe = None
    if cfg.sorry_bench_path:
        unsafe = prompts_mod.load_sorry_bench(
            cfg.sorry_bench_path, cfg.sorry_bench_categories
        )
    pairs = prompts_mod.build_pairs(unsafe_requests=unsafe, seed=cfg.seed)
    prompts_mod.validate_pairs(pairs)
    return pairs


def run(cfg: Config) -> dict:
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ---------------- 1. prompts ----------------
    pairs = build_dataset(cfg)
    texts, labels_list, pair_ids = prompts_mod.pairs_to_examples(pairs)
    labels = np.array(labels_list)
    train_ids, test_ids = prompts_mod.split_pair_ids(pairs, cfg.test_size, cfg.seed)
    train_idx = _indices_for(pair_ids, train_ids)
    test_idx = _indices_for(pair_ids, test_ids)
    print(f"[1/7] {len(pairs)} pairs -> {len(texts)} prompts "
          f"({len(train_idx)} train / {len(test_idx)} test)")

    # Framing-length report: large mean differences hint at a surface confound.
    length_report = prompts_mod.framing_length_report(pairs)
    print(f"[1/7] framing lengths: eval={length_report['eval_mean_chars']:.0f}±"
          f"{length_report['eval_std_chars']:.0f} chars, "
          f"deploy={length_report['deploy_mean_chars']:.0f}±"
          f"{length_report['deploy_std_chars']:.0f} chars, "
          f"mean |diff|={length_report['mean_abs_diff_chars']:.0f}")

    # ---------------- 2. model + activations ----------------
    from .activations import load_model, collect_activations
    tokenizer, model = load_model(cfg)
    print(f"[2/7] loaded {cfg.model_name}")
    acts = collect_activations(tokenizer, model, texts, cfg)
    print(f"[2/7] activations {acts.shape} (layers, examples, d_model)")
    d_model = acts.shape[2]

    # ---------------- 3. probing ----------------
    probe = train_layer_probes(acts, labels, train_idx, test_idx, cfg)
    sel_str = (f"selectivity={probe.best_selectivity:.3f}"
               if probe.best_selectivity is not None else "selectivity=n/a")
    min_layer_note = (f" (excluded layers <{cfg.probe_min_layer})"
                      if getattr(cfg, "probe_min_layer", 0) > 0 else "")
    print(f"[3/7] best layer {probe.best_layer}{min_layer_note} | "
          f"AUC {probe.best_auc:.3f} | {sel_str}")
    best_layer_acts = acts[probe.best_layer]
    direction = difference_of_means_direction(
        best_layer_acts[train_idx], labels[train_idx]
    )

    # ---------------- 4. baseline behaviour ----------------
    from .behaviour import generate_responses
    eval_texts = [t for t, l in zip(texts, labels_list) if l == 1]
    deploy_texts = [t for t, l in zip(texts, labels_list) if l == 0]
    classifier = KeywordRefusalClassifier()

    base_eval = generate_responses(tokenizer, model, eval_texts, cfg)
    base_deploy = generate_responses(tokenizer, model, deploy_texts, cfg)
    baseline = measure_gap(base_eval, base_deploy, classifier, keep_responses=True)
    print(f"[4/7] baseline gap G={baseline.gap:.3f} "
          f"(eval {baseline.refusal_rate_eval:.3f} / deploy {baseline.refusal_rate_deploy:.3f})")

    # ------------- per-prompt records (for the results viewer) -------------
    preds, confs = predict_per_example(probe, best_layer_acts)
    test_positions = set(test_idx.tolist())
    eval_pos = [i for i, l in enumerate(labels_list) if l == 1]
    deploy_pos = [i for i, l in enumerate(labels_list) if l == 0]

    pair_by_id = {p.pair_id: p for p in pairs}
    per_prompt = []
    for framing, positions, responses in (
        ("evaluation", eval_pos, base_eval),
        ("deployment", deploy_pos, base_deploy),
    ):
        for slot, global_i in enumerate(positions):
            pair = pair_by_id[pair_ids[global_i]]
            true_label = int(labels[global_i])
            pred = int(preds[global_i])
            response = responses[slot] if slot < len(responses) else ""
            per_prompt.append({
                "pair_id": pair.pair_id,
                "request": pair.request,
                "framing": framing,
                "is_unsafe": bool(pair.is_unsafe),
                "prompt_text": texts[global_i],
                "split": "test" if global_i in test_positions else "train",
                "true_label": true_label,
                "probe_prediction": pred,
                "probe_says_being_tested": bool(pred == 1),
                "probe_confidence_eval": float(confs[global_i]),
                "probe_correct": bool(pred == true_label),
                "response": response,
                "refused": bool(classifier.is_refusal(response)),
            })
    per_prompt.sort(key=lambda r: (r["pair_id"], r["framing"]))
    n_correct = sum(1 for r in per_prompt if r["probe_correct"])
    print(f"[4/7] probe identified framing correctly on "
          f"{n_correct}/{len(per_prompt)} prompts")

    # ---------------- 5. suppression sweep ----------------
    from .suppression import SteeringHook
    alphas, aucs, gaps = [], [], []
    for alpha in cfg.steering_alphas:
        if alpha == 0.0:
            aucs.append(probe.best_auc)
            gaps.append(baseline.gap)
            alphas.append(alpha)
            continue

        with SteeringHook(model, probe.best_layer, direction, alpha):
            steered_acts = collect_activations(tokenizer, model, texts, cfg)
            steered_auc = score_with_probe(
                probe, steered_acts[probe.best_layer][test_idx], labels[test_idx]
            )
            s_eval = generate_responses(tokenizer, model, eval_texts, cfg)
            s_deploy = generate_responses(tokenizer, model, deploy_texts, cfg)
            steered_gap = measure_gap(s_eval, s_deploy, classifier).gap

        alphas.append(alpha)
        aucs.append(steered_auc)
        gaps.append(steered_gap)
        print(f"[5/7] alpha={alpha:<4} AUC={steered_auc:.3f} G={steered_gap:.3f}")

    # ---------------- 6. diagnostics ----------------
    n_train = int(len(train_idx))
    warnings = run_diagnostics(
        probe_results=probe,
        n_train=n_train,
        d_model=d_model,
        layer0_tolerance=cfg.diag_layer0_tolerance,
        control_auc_threshold=cfg.diag_control_auc_threshold,
        sample_ratio_threshold=cfg.diag_sample_ratio_threshold,
        flat_cv_threshold=cfg.diag_flat_cv_threshold,
    )
    if warnings:
        print(f"[6/7] {len(warnings)} diagnostic warning(s):")
        for w in warnings:
            print(f"  [{w.code}] {w.detail}")
    else:
        print("[6/7] no diagnostic warnings")

    # ---------------- 7. verdict ----------------
    results = summarise_sweep(probe.best_auc, baseline.gap, alphas, aucs, gaps, cfg)
    print(f"[7/7] VERDICT: {results.overall_verdict.upper()}")

    payload = {
        "config": json.loads(cfg.to_json()),
        "n_pairs": len(pairs),
        "framing_length_report": length_report,
        "probes": probe.to_dict(),
        "baseline_behaviour": baseline.to_dict(),
        "per_prompt": per_prompt,
        "suppression": results.to_dict(),
        "diagnostics": {
            "n_train": n_train,
            "d_model": d_model,
            "n_warnings": len(warnings),
            "warnings": [w.to_dict() for w in warnings],
        },
    }
    out_path = os.path.join(cfg.output_dir, "results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"saved -> {out_path}")
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluation-awareness probing pipeline")
    p.add_argument("--model", default=None, help="HF model id")
    p.add_argument("--sorry-bench", default=None, help="path to SORRY-Bench CSV")
    p.add_argument("--output", default=None, help="output directory")
    p.add_argument("--device", default=None, help="auto | cuda | cpu")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--pca", type=int, default=None,
                   help="PCA components before probing (overrides config default)")
    p.add_argument("--min-layer", type=int, default=None,
                   help="Skip layers below this index when selecting the best "
                        "probe layer (e.g. --min-layer 1 excludes the "
                        "embedding-output layer 0 to sidestep vocabulary "
                        "confounds).")
    p.add_argument("--4bit", dest="four_bit", action="store_true",
                   help="Load the model in 4-bit precision via bitsandbytes. "
                        "Needed to fit ~7B models on a 6 GB laptop GPU. "
                        "Requires: pip install bitsandbytes accelerate")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Batch size for activation collection and generation. "
                        "Reduce (e.g. 1 or 2) to fit larger models on smaller "
                        "GPUs. Default 8.")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Maximum new tokens generated per response in stage 5. "
                        "Reduce (e.g. 48 or 64) to shrink KV-cache memory on "
                        "smaller GPUs. Default 128.")
    p.add_argument("--force-gpu", dest="force_gpu", action="store_true",
                   help="Pin every layer to GPU 0, overriding accelerate's "
                        "automatic device-map planning. Use when accelerate "
                        "refuses an all-GPU load you believe will just fit. "
                        "Hard-OOMs if it does not.")
    args = p.parse_args()

    cfg = Config()
    if args.model:
        cfg.model_name = args.model
    if args.sorry_bench:
        cfg.sorry_bench_path = args.sorry_bench
    if args.output:
        cfg.output_dir = args.output
    if args.device:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed
    if args.pca is not None:
        cfg.probe_pca_components = args.pca
    if args.min_layer is not None:
        cfg.probe_min_layer = args.min_layer
    if args.four_bit:
        cfg.load_in_4bit = True
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens
    if args.force_gpu:
        cfg.force_gpu_only = True

    run(cfg)


if __name__ == "__main__":
    main()
