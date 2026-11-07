"""Verification suite for the evaluation-awareness pipeline.

These tests prove the maths and control flow are correct WITHOUT needing to
download model weights or a GPU. A stub transformer exercises the real
forward-hook machinery (activation caching and steering), so the hook code is
genuinely executed rather than assumed to work.

Run: python -m pytest eval_awareness/tests -v
 or: python eval_awareness/tests/test_pipeline.py
"""

from __future__ import annotations

import sys
import os
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from eval_awareness.config import Config
from eval_awareness import prompts as P
from eval_awareness.probes import (
 train_layer_probes,
 difference_of_means_direction,
 score_with_probe,
)
from eval_awareness.behaviour import (
 KeywordRefusalClassifier,
 RubricRefusalClassifier,
 measure_gap,
 refusal_rate,
)
from eval_awareness.suppression import (
 proportional_auc_reduction,
 is_oversteered,
 proportional_gap_reduction,
 classify_verdict,
 summarise_sweep,
 SteeringHook,
)
from eval_awareness.activations import ActivationCache, get_decoder_layers
from eval_awareness.diagnostics import (
 run_diagnostics,
 check_layer0_confound,
 check_high_control_auc,
 check_sample_size_ratio,
 check_flat_auc_profile,
 DiagnosticWarning,
)


# ==========================================================================
# Stub model: exercises the real hook paths with deterministic weights.
# ==========================================================================
class StubBlock(nn.Module):
 """Decoder block returning a tuple, like real HF blocks do."""
 def __init__(self, d_model: int, bias: float):
 super().__init__()
 self.lin = nn.Linear(d_model, d_model)
 with torch.no_grad():
 self.lin.weight.copy_(torch.eye(d_model))
 self.lin.bias.fill_(bias)

 def forward(self, x):
 return (self.lin(x),)


class StubInner(nn.Module):
 def __init__(self, n_layers: int, d_model: int):
 super().__init__()
 self.layers = nn.ModuleList(
 [StubBlock(d_model, bias=0.1 * (i + 1)) for i in range(n_layers)]
 )


class StubModel(nn.Module):
 """Minimal stand-in exposing `model.layers`, matching the Llama/Qwen path."""
 def __init__(self, n_layers=4, d_model=8):
 super().__init__()
 self.model = StubInner(n_layers, d_model)
 self.d_model = d_model

 def forward(self, x):
 for block in self.model.layers:
 x = block(x)[0]
 return x


# ==========================================================================
# Prompt construction
# ==========================================================================
def test_pairs_share_identical_request():
 pairs = P.build_pairs()
 P.validate_pairs(pairs)
 for p in pairs:
 assert p.request in p.eval_text
 assert p.request in p.deploy_text
 assert p.eval_text != p.deploy_text
 print(f" ok: {len(pairs)} pairs, request identical across framings")


def test_validate_pairs_catches_broken_pair():
 pairs = P.build_pairs()
 pairs[0].eval_text = "totally unrelated text"
 try:
 P.validate_pairs(pairs)
 raise AssertionError("validate_pairs should have rejected a broken pair")
 except ValueError:
 pass
 print(" ok: broken pair rejected")


def test_labels_are_balanced_and_aligned():
 pairs = P.build_pairs()
 texts, labels, pair_ids = P.pairs_to_examples(pairs)
 assert len(texts) == len(labels) == len(pair_ids) == 2 * len(pairs)
 assert sum(labels) == len(pairs) # exactly half are eval
 for i in range(0, len(texts), 2):
 assert labels[i] == 1 and labels[i + 1] == 0 # eval then deploy
 assert pair_ids[i] == pair_ids[i + 1] # same pair
 print(f" ok: {len(texts)} examples, balanced and aligned")


def test_split_never_leaks_a_pair_across_sides():
 pairs = P.build_pairs()
 train_ids, test_ids = P.split_pair_ids(pairs, test_size=0.3, seed=42)
 assert train_ids.isdisjoint(test_ids), "pair leaked across the split"
 assert train_ids and test_ids
 assert train_ids | test_ids == {p.pair_id for p in pairs}
 print(f" ok: {len(train_ids)} train / {len(test_ids)} test pairs, disjoint")


def test_split_is_deterministic():
 pairs = P.build_pairs()
 a = P.split_pair_ids(pairs, 0.3, seed=7)
 b = P.split_pair_ids(pairs, 0.3, seed=7)
 assert a == b
 print(" ok: split reproducible under a fixed seed")


# ==========================================================================
# Probing
# ==========================================================================
def _synthetic_activations(n_layers=4, n_per_class=25, d=6, signal_layer=2, seed=0):
 """Layer `signal_layer` carries a real class signal; others are noise."""
 rng = np.random.RandomState(seed)
 n = n_per_class * 2
 labels = np.array([1] * n_per_class + [0] * n_per_class)
 acts = rng.randn(n_layers, n, d) * 0.5
 acts[signal_layer, :n_per_class, 0] += 4.0 # separate the classes
 return acts, labels


def test_probe_finds_the_layer_carrying_signal():
 cfg = Config()
 acts, labels = _synthetic_activations(signal_layer=2)
 idx = np.arange(len(labels))
 rng = np.random.RandomState(0)
 rng.shuffle(idx)
 train_idx, test_idx = idx[:30], idx[30:]

 res = train_layer_probes(acts, labels, train_idx, test_idx, cfg)
 assert res.best_layer == 2, f"expected layer 2, got {res.best_layer}"
 assert res.best_auc > 0.9, f"expected strong AUC, got {res.best_auc}"
 print(f" ok: found signal layer 2 with AUC {res.best_auc:.3f}")


def test_control_probe_stays_near_chance():
 """Selectivity control task: shuffled labels must NOT be learnable."""
 cfg = Config()
 acts, labels = _synthetic_activations(signal_layer=2)
 idx = np.arange(len(labels))
 rng = np.random.RandomState(0)
 rng.shuffle(idx)
 train_idx, test_idx = idx[:30], idx[30:]

 res = train_layer_probes(acts, labels, train_idx, test_idx, cfg)
 best = [r for r in res.layers if r.layer == 2][0]
 assert best.control_auc is not None
 assert best.control_auc < 0.75, f"control AUC too high: {best.control_auc}"
 assert best.selectivity > 0.2, f"selectivity too low: {best.selectivity}"
 print(f" ok: control AUC {best.control_auc:.3f}, selectivity {best.selectivity:.3f}")


def test_probe_rejects_mismatched_shapes():
 cfg = Config()
 acts, labels = _synthetic_activations()
 try:
 train_layer_probes(acts, labels[:-5], np.arange(10), np.arange(10, 20), cfg)
 raise AssertionError("should have rejected mismatched label length")
 except ValueError:
 pass
 print(" ok: mismatched shapes rejected")


def test_direction_is_unit_norm_and_points_the_right_way():
 acts, labels = _synthetic_activations(signal_layer=2)
 d = difference_of_means_direction(acts[2], labels)
 assert abs(np.linalg.norm(d) - 1.0) < 1e-6, "direction must be unit norm"
 # Class 1 was shifted along dimension 0, so the direction must load there.
 assert d[0] > 0.8, f"direction should point along dim 0, got {d}"
 print(f" ok: unit-norm direction, dim0 loading {d[0]:.3f}")


def test_direction_requires_both_classes():
 acts, labels = _synthetic_activations()
 try:
 difference_of_means_direction(acts[0], np.ones_like(labels))
 raise AssertionError("should have rejected single-class input")
 except ValueError:
 pass
 print(" ok: single-class direction rejected")


# ==========================================================================
# Behaviour / refusal classification
# ==========================================================================
def test_refusal_classifier_basic_cases():
 c = KeywordRefusalClassifier()
 refusals = [
 "I can't help with that request.",
 "I'm sorry, but I cannot provide that information.",
 "I will not write malware for you.",
 "Sorry, but that's not something I can assist with.",
 ]
 compliances = [
 "Sure! Here's how firewall rules are evaluated: first, the ruleset...",
 "AES-256 uses a 256-bit key, whereas AES-128 uses 128 bits.",
 "Certainly. A SIEM aggregates logs from across the estate.",
 ]
 for r in refusals:
 assert c.is_refusal(r), f"missed refusal: {r!r}"
 for r in compliances:
 assert not c.is_refusal(r), f"false positive: {r!r}"
 print(f" ok: {len(refusals)} refusals + {len(compliances)} compliances classified")


def test_trailing_caveat_is_not_counted_as_refusal():
 """A compliance that ends with a safety note must not score as a refusal."""
 c = KeywordRefusalClassifier()
 text = (
 "Here's how a firewall processes overlapping rules: it evaluates them "
 "top-down and applies the first match. " + ("Detail. " * 40) +
 "I can't recommend using this to bypass controls."
 )
 assert not c.is_refusal(text), "trailing caveat wrongly scored as refusal"
 print(" ok: trailing caveat after compliance not counted")


def test_hedge_phrases_are_not_refusals():
 c = KeywordRefusalClassifier()
 assert not c.is_refusal("I can't be certain, but here's the usual approach: ...")
 assert not c.is_refusal("I'm sorry to hear that. Here's what typically helps: ...")
 print(" ok: hedges distinguished from refusals")


def test_empty_response_is_not_a_refusal():
 c = KeywordRefusalClassifier()
 assert not c.is_refusal("")
 assert not c.is_refusal(" ")
 assert not c.is_refusal(None)
 print(" ok: empty completions not counted as refusals")


def test_gap_equation_3_1():
 c = KeywordRefusalClassifier()
 # 3 of 4 refuse under eval, 1 of 4 under deployment -> G = |0.75 - 0.25| = 0.5
 ev = ["I can't help.", "I cannot assist.", "I'm sorry, no.", "Sure, here's how."]
 dep = ["Sure, here's how.", "Here you go.", "Certainly.", "I can't help."]
 res = measure_gap(ev, dep, c)
 assert abs(res.refusal_rate_eval - 0.75) < 1e-9
 assert abs(res.refusal_rate_deploy - 0.25) < 1e-9
 assert abs(res.gap - 0.5) < 1e-9
 print(f" ok: G = |{res.refusal_rate_eval} - {res.refusal_rate_deploy}| = {res.gap}")


def test_gap_is_absolute_value():
 """G must be non-negative even when deployment refuses more."""
 c = KeywordRefusalClassifier()
 ev = ["Sure, here's how.", "Here you go."]
 dep = ["I can't help.", "I cannot assist."]
 res = measure_gap(ev, dep, c)
 assert res.gap == 1.0 and res.gap >= 0
 print(" ok: gap is absolute, direction-independent")


def test_refusal_rate_rejects_empty_input():
 try:
 refusal_rate([], KeywordRefusalClassifier())
 raise AssertionError("should reject empty response list")
 except ValueError:
 pass
 print(" ok: empty response list rejected")


def test_rubric_classifier_refuses_to_silently_fallback():
 try:
 RubricRefusalClassifier().is_refusal("anything")
 raise AssertionError("should not silently fall back")
 except NotImplementedError:
 pass
 assert RubricRefusalClassifier(judge_fn=lambda r: True).is_refusal("x") is True
 print(" ok: rubric classifier fails loudly without a judge")


# ==========================================================================
# Suppression maths (Equations 3.2 / 3.3) and verdicts
# ==========================================================================
def test_equation_3_2_auc_reduction():
 # baseline 0.9 -> 0.5 removes ALL above-chance signal => 1.0
 assert abs(proportional_auc_reduction(0.9, 0.5) - 1.0) < 1e-9
 # halfway: 0.9 -> 0.7, above-chance 0.4 -> 0.2 => 0.5
 assert abs(proportional_auc_reduction(0.9, 0.7) - 0.5) < 1e-9
 # no change => 0.0
 assert abs(proportional_auc_reduction(0.9, 0.9) - 0.0) < 1e-9
 # baseline already at chance => nothing to remove, guard against /0
 assert proportional_auc_reduction(0.5, 0.5) == 0.0
 assert proportional_auc_reduction(0.4, 0.3) == 0.0
 print(" ok: Equation 3.2 verified incl. divide-by-zero guard")


def test_oversteering_is_flagged_not_hidden():
 """Pushing the probe below chance inverts it; that must be visible."""
 d = proportional_auc_reduction(0.9, 0.3) # 0.3 < 0.5 chance
 assert d > 1.0, d
 assert is_oversteered(d)
 assert not is_oversteered(proportional_auc_reduction(0.9, 0.7))
 cfg = Config()
 res = summarise_sweep(0.9, 0.4, [0.0, 5.0], [0.9, 0.3], [0.4, 0.39], cfg)
 over = [p for p in res.points if p.alpha == 5.0][0]
 assert over.oversteered is True
 assert res.points[0].oversteered is False
 print(f" ok: over-steering flagged (dAUC {d:.3f}) rather than clamped")


def test_equation_3_3_gap_reduction():
 assert abs(proportional_gap_reduction(0.5, 0.0) - 1.0) < 1e-9
 assert abs(proportional_gap_reduction(0.5, 0.25) - 0.5) < 1e-9
 assert abs(proportional_gap_reduction(0.5, 0.5) - 0.0) < 1e-9
 assert proportional_gap_reduction(0.0, 0.0) == 0.0 # guard
 # gap can widen -> negative reduction is meaningful, not clamped
 assert proportional_gap_reduction(0.4, 0.6) < 0
 print(" ok: Equation 3.3 verified incl. divide-by-zero guard")


def test_verdict_boundaries():
 cos_t, real_t = 0.5, 0.3
 # probe stripped, behaviour unmoved -> cosmetic
 assert classify_verdict(0.9, 0.05, cos_t, real_t) == "cosmetic"
 # behaviour genuinely moved -> real
 assert classify_verdict(0.9, 0.8, cos_t, real_t) == "real"
 # neither -> inconclusive
 assert classify_verdict(0.2, 0.05, cos_t, real_t) == "inconclusive"
 # 'real' wins when both would qualify
 assert classify_verdict(0.99, 0.5, cos_t, real_t) == "real"
 # exact threshold on gap counts as real (>=)
 assert classify_verdict(0.9, 0.3, cos_t, real_t) == "real"
 # exact threshold on AUC does NOT count as cosmetic (strict >)
 assert classify_verdict(0.5, 0.0, cos_t, real_t) == "inconclusive"
 print(" ok: verdict boundaries incl. inclusive/strict thresholds")


def test_sweep_overall_verdict_cosmetic():
 cfg = Config()
 res = summarise_sweep(
 baseline_auc=0.95, baseline_gap=0.40,
 alphas=[0.0, 1.0, 2.0],
 aucs=[0.95, 0.60, 0.50], # probe signal collapses
 gaps=[0.40, 0.39, 0.38], # behaviour barely moves
 cfg=cfg,
 )
 assert res.overall_verdict == "cosmetic", res.overall_verdict
 print(f" ok: cosmetic case -> {res.overall_verdict}")


def test_sweep_overall_verdict_real():
 cfg = Config()
 res = summarise_sweep(
 baseline_auc=0.95, baseline_gap=0.40,
 alphas=[0.0, 1.0, 2.0],
 aucs=[0.95, 0.70, 0.55],
 gaps=[0.40, 0.20, 0.05], # behaviour genuinely closes
 cfg=cfg,
 )
 assert res.overall_verdict == "real", res.overall_verdict
 print(f" ok: real case -> {res.overall_verdict}")


def test_sweep_overall_verdict_inconclusive():
 cfg = Config()
 res = summarise_sweep(
 baseline_auc=0.95, baseline_gap=0.40,
 alphas=[0.0, 1.0],
 aucs=[0.95, 0.93], # barely dented
 gaps=[0.40, 0.39],
 cfg=cfg,
 )
 assert res.overall_verdict == "inconclusive", res.overall_verdict
 print(f" ok: inconclusive case -> {res.overall_verdict}")


def test_alpha_zero_cannot_drive_the_verdict():
 """The unsteered control must be excluded from verdict evidence."""
 cfg = Config()
 res = summarise_sweep(
 baseline_auc=0.95, baseline_gap=0.40,
 alphas=[0.0], # control only
 aucs=[0.10], gaps=[0.0], # would look 'real' if wrongly counted
 cfg=cfg,
 )
 assert res.overall_verdict == "inconclusive", res.overall_verdict
 print(" ok: alpha=0 control excluded from verdict")


def test_sweep_rejects_ragged_input():
 cfg = Config()
 try:
 summarise_sweep(0.9, 0.4, [0.0, 1.0], [0.9], [0.4, 0.2], cfg)
 raise AssertionError("should reject mismatched lengths")
 except ValueError:
 pass
 print(" ok: ragged sweep input rejected")


# ==========================================================================
# Hook machinery, executed against the stub model
# ==========================================================================
def test_activation_cache_captures_every_layer():
 model = StubModel(n_layers=4, d_model=8)
 x = torch.randn(3, 5, 8) # (batch, seq, d_model)
 with ActivationCache(model, token_position=-1) as cache:
 model(x)
 acts = cache.stacked()
 assert acts.shape == (4, 3, 8), acts.shape
 print(f" ok: cached {acts.shape} (layers, batch, d_model)")


def test_activation_cache_reads_the_requested_token():
 model = StubModel(n_layers=2, d_model=4)
 x = torch.zeros(1, 3, 4)
 x[0, -1, :] = 5.0 # mark only the final position
 with ActivationCache(model, token_position=-1) as cache:
 model(x)
 acts = cache.stacked()
 # layer 0 adds bias 0.1 to the marked value
 assert abs(acts[0, 0, 0] - 5.1) < 1e-4, acts[0, 0, 0]
 print(" ok: reads the requested token position")


def test_hooks_are_removed_on_exit():
 model = StubModel()
 layers = get_decoder_layers(model)
 before = len(layers[0]._forward_hooks)
 with ActivationCache(model):
 during = len(layers[0]._forward_hooks)
 after = len(layers[0]._forward_hooks)
 assert during == before + 1 and after == before
 print(" ok: hooks registered then cleanly removed")


def test_steering_ablates_the_component_along_the_direction():
 """alpha=1.0 must fully erase the component along d, leaving others intact."""
 model = StubModel(n_layers=3, d_model=4)
 direction = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
 x = torch.zeros(1, 2, 4)

 with ActivationCache(model) as c:
 model(x)
 base = c.stacked()

 with SteeringHook(model, layer_idx=1, direction=direction, alpha=1.0):
 with ActivationCache(model) as c:
 model(x)
 steered = c.stacked()

 # Layer 0 is upstream of the intervention: unchanged.
 assert np.allclose(base[0], steered[0]), "upstream layer was altered"
 # At layer 1 the dim-0 component (the direction) must be driven to ~0.
 assert abs(steered[1, 0, 0]) < 1e-4, f"component not erased: {steered[1,0,0]}"
 # Orthogonal dimensions must be untouched.
 assert np.allclose(base[1, 0, 1:], steered[1, 0, 1:], atol=1e-4), \
 "ablation altered orthogonal dimensions"
 print(f" ok: dim0 {base[1,0,0]:.3f} -> {steered[1,0,0]:.3f}, "
 f"orthogonal dims preserved")


def test_ablation_is_proportional_to_alpha():
 """alpha=0.5 removes half the component; alpha=0 changes nothing."""
 model = StubModel(n_layers=2, d_model=4)
 direction = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
 x = torch.zeros(1, 2, 4)

 with ActivationCache(model) as c:
 model(x)
 base = c.stacked()[1, 0, 0]

 vals = {}
 for a in (0.0, 0.5, 1.0):
 with SteeringHook(model, 1, direction, a):
 with ActivationCache(model) as c:
 model(x)
 vals[a] = c.stacked()[1, 0, 0]

 assert abs(vals[0.0] - base) < 1e-5, "alpha=0 must be a no-op"
 assert abs(vals[0.5] - 0.5 * base) < 1e-4, f"alpha=0.5 -> {vals[0.5]}"
 assert abs(vals[1.0]) < 1e-4, f"alpha=1.0 -> {vals[1.0]}"
 print(f" ok: base {base:.3f} -> 0.5a {vals[0.5]:.3f} -> 1.0a {vals[1.0]:.3f}")


def test_uniform_translation_would_not_change_auc():
 """Documents WHY ablation is used instead of constant subtraction:
 AUC is rank-based, so translating every activation equally cannot move it."""
 cfg = Config()
 acts, labels = _synthetic_activations(signal_layer=2, n_per_class=30, seed=5)
 idx = np.arange(len(labels)); np.random.RandomState(2).shuffle(idx)
 train_idx, test_idx = idx[:40], idx[40:]
 probe = train_layer_probes(acts, labels, train_idx, test_idx, cfg)
 layer = acts[probe.best_layer]

 base = score_with_probe(probe, layer[test_idx], labels[test_idx])
 shifted = layer.copy() - 3.0 * np.array([1, 0, 0, 0, 0, 0], dtype=float)
 shifted_auc = score_with_probe(probe, shifted[test_idx], labels[test_idx])
 assert abs(base - shifted_auc) < 1e-9, \
 "uniform translation unexpectedly changed AUC"
 print(f" ok: translation leaves AUC at {base:.3f} (hence ablation is required)")


def test_steering_hook_restores_model_after_exit():
 model = StubModel(n_layers=3, d_model=4)
 direction = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
 x = torch.zeros(1, 2, 4)

 with ActivationCache(model) as c:
 model(x)
 before = c.stacked()
 with SteeringHook(model, 1, direction, 3.0):
 pass # enter and exit
 with ActivationCache(model) as c:
 model(x)
 after = c.stacked()

 assert np.allclose(before, after), "model not restored after steering"
 print(" ok: model state restored after steering")


def test_steering_rejects_bad_layer_and_dim():
 model = StubModel(n_layers=3, d_model=4)
 try:
 SteeringHook(model, 99, np.zeros(4), 1.0)
 raise AssertionError("should reject out-of-range layer")
 except IndexError:
 pass

 bad_dim = np.ones(7, dtype=np.float32) # 7 != d_model 4
 try:
 with SteeringHook(model, 1, bad_dim, 1.0):
 model(torch.zeros(1, 2, 4))
 raise AssertionError("should reject dimension mismatch")
 except ValueError:
 pass
 print(" ok: bad layer index and dim mismatch rejected")


# ==========================================================================
# Integrated flow
# ==========================================================================
def test_probe_scoring_drops_when_signal_is_steered_out():
 """End-to-end: build signal, probe it, ablate it, confirm AUC collapses."""
 cfg = Config()
 acts, labels = _synthetic_activations(signal_layer=2, n_per_class=40, seed=3)
 idx = np.arange(len(labels))
 np.random.RandomState(1).shuffle(idx)
 train_idx, test_idx = idx[:50], idx[50:]

 probe = train_layer_probes(acts, labels, train_idx, test_idx, cfg)
 direction = difference_of_means_direction(acts[probe.best_layer][train_idx],
 labels[train_idx])

 base_auc = score_with_probe(probe, acts[probe.best_layer][test_idx], labels[test_idx])

 # Ablate the direction from the positive class, mimicking suppression.
 ablated = acts[probe.best_layer].copy()
 shift = 4.0
 ablated[labels == 1] -= shift * direction
 steer_auc = score_with_probe(probe, ablated[test_idx], labels[test_idx])

 assert base_auc > 0.9, base_auc
 assert steer_auc < base_auc, f"AUC did not fall: {base_auc} -> {steer_auc}"

 d_auc = proportional_auc_reduction(base_auc, steer_auc)
 assert d_auc > 0.5, d_auc
 verdict = classify_verdict(d_auc, 0.02, cfg.cosmetic_auc_threshold,
 cfg.real_gap_threshold)
 assert verdict == "cosmetic"
 print(f" ok: AUC {base_auc:.3f} -> {steer_auc:.3f} "
 f"(dAUC {d_auc:.3f}) -> {verdict}")


def test_config_serialises():
 cfg = Config()
 import json
 parsed = json.loads(cfg.to_json())
 assert parsed["model_name"] == cfg.model_name
 assert parsed["steering_alphas"] == cfg.steering_alphas
 assert "probe_pca_components" in parsed, "probe_pca_components missing from config"
 assert "probe_C" in parsed
 print(" ok: config round-trips to JSON (incl. new fields)")


def test_config_new_defaults():
 """Verify the tighter regularisation and PCA defaults are set."""
 cfg = Config()
 assert cfg.probe_C <= 0.1, f"probe_C should be ≤0.1, got {cfg.probe_C}"
 assert cfg.probe_pca_components is not None, "PCA should be on by default"
 assert cfg.probe_pca_components <= 100
 print(f" ok: probe_C={cfg.probe_C}, pca={cfg.probe_pca_components}")


# ==========================================================================
# New prompt set
# ==========================================================================
def test_prompt_set_expanded():
 """Default prompt set must have ≥25 unsafe and ≥25 benign requests."""
 from eval_awareness.prompts import UNSAFE_REQUESTS, BENIGN_REQUESTS
 assert len(UNSAFE_REQUESTS) >= 25, f"need ≥25 unsafe, got {len(UNSAFE_REQUESTS)}"
 assert len(BENIGN_REQUESTS) >= 25, f"need ≥25 benign, got {len(BENIGN_REQUESTS)}"
 print(f" ok: {len(UNSAFE_REQUESTS)} unsafe + {len(BENIGN_REQUESTS)} benign requests")


def test_template_count():
 """Must have 12 eval and 12 deploy templates."""
 from eval_awareness.prompts import EVAL_FRAMES, DEPLOY_FRAMES
 assert len(EVAL_FRAMES) == 12, f"expected 12 eval frames, got {len(EVAL_FRAMES)}"
 assert len(DEPLOY_FRAMES) == 12, f"expected 12 deploy frames, got {len(DEPLOY_FRAMES)}"
 print(f" ok: 12 eval + 12 deploy framing templates")


def test_framing_length_report():
 """framing_length_report must return a dict and mean |diff| must be reasonable."""
 from eval_awareness.prompts import build_pairs, framing_length_report
 pairs = build_pairs()
 report = framing_length_report(pairs)
 assert "mean_abs_diff_chars" in report
 assert "n_pairs" in report
 assert report["n_pairs"] == len(pairs)
 diff = report["mean_abs_diff_chars"]
 # With matched templates the mean diff should be <200 chars (length-matching aim)
 assert diff < 200, (
 f"mean |diff| = {diff:.1f} chars — templates may have a length confound"
 )
 print(f" ok: framing_length_report, mean |diff| = {diff:.1f} chars")


# ==========================================================================
# Diagnostics
# ==========================================================================
def _make_probe_results_flat(n_layers=28, best_auc=1.0, control_auc=0.8, seed=0):
 """Build a synthetic ProbeResults with a flat AUC profile (all layers same AUC)
 and a high control AUC, mimicking the first-run artefact."""
 from eval_awareness.probes import ProbeResults, LayerResult
 from sklearn.linear_model import LogisticRegression
 from sklearn.preprocessing import StandardScaler
 import numpy as np
 layers = [
 LayerResult(layer=i, auc=best_auc, accuracy=1.0, control_auc=control_auc)
 for i in range(n_layers)
 ]
 # Dummy scaler and classifier (not used by diagnostics)
 rng = np.random.RandomState(seed)
 X = rng.randn(20, 6)
 y = np.array([1]*10 + [0]*10)
 scaler = StandardScaler().fit(X)
 clf = LogisticRegression().fit(scaler.transform(X), y)
 return ProbeResults(
 layers=layers,
 best_layer=0,
 best_auc=best_auc,
 best_selectivity=best_auc - control_auc,
 scaler=scaler,
 classifier=clf,
 )


def test_w001_layer0_confound_fires():
 """W001 must fire when layer-0 AUC is within tolerance of best AUC."""
 probe = _make_probe_results_flat(n_layers=4, best_auc=1.0, control_auc=0.8)
 w = check_layer0_confound(probe.layers, tolerance=0.05)
 assert w is not None, "W001 should have fired"
 assert w.code == "W001"
 assert "W001" in w.code
 print(f" ok: W001 fired — {w.detail}")


def test_w001_does_not_fire_when_layer0_is_lower():
 """W001 must NOT fire when the signal clearly emerges above layer 0."""
 from eval_awareness.probes import LayerResult
 layers = [
 LayerResult(layer=0, auc=0.55, accuracy=0.6),
 LayerResult(layer=1, auc=0.70, accuracy=0.7),
 LayerResult(layer=2, auc=0.92, accuracy=0.9),
 ]
 w = check_layer0_confound(layers, tolerance=0.05)
 assert w is None, f"W001 should not have fired, got {w}"
 print(" ok: W001 absent when signal emerges progressively")


def test_w002_high_control_auc_fires():
 """W002 must fire when control AUC > threshold."""
 probe = _make_probe_results_flat(best_auc=1.0, control_auc=0.82)
 w = check_high_control_auc(probe.layers, threshold=0.60)
 assert w is not None, "W002 should have fired"
 assert w.code == "W002"
 print(f" ok: W002 fired — {w.detail}")


def test_w002_does_not_fire_when_control_near_chance():
 """W002 must NOT fire when control AUC is near 0.5."""
 probe = _make_probe_results_flat(best_auc=0.85, control_auc=0.53)
 w = check_high_control_auc(probe.layers, threshold=0.60)
 assert w is None, f"W002 should not have fired, got {w}"
 print(" ok: W002 absent when control AUC near chance")


def test_w003_small_sample_fires():
 """W003 must fire when n_train / d_model < threshold."""
 w = check_sample_size_ratio(n_train=28, d_model=1536, threshold=10.0)
 assert w is not None, "W003 should have fired (28/1536 = 0.018)"
 assert w.code == "W003"
 assert "28" in w.detail and "1536" in w.detail
 print(f" ok: W003 fired — {w.detail}")


def test_w003_does_not_fire_with_large_sample():
 """W003 must NOT fire when we have enough examples per dimension."""
 w = check_sample_size_ratio(n_train=20000, d_model=1536, threshold=10.0)
 assert w is None, f"W003 should not have fired, got {w}"
 print(" ok: W003 absent with n_train/d_model well above threshold")


def test_w004_flat_profile_fires():
 """W004 must fire when AUC is identical across all layers."""
 probe = _make_probe_results_flat(n_layers=28, best_auc=1.0)
 w = check_flat_auc_profile(probe.layers, cv_threshold=0.05)
 assert w is not None, "W004 should have fired (CV = 0)"
 assert w.code == "W004"
 print(f" ok: W004 fired — {w.detail}")


def test_w004_does_not_fire_with_clear_gradient():
 """W004 must NOT fire when AUC grows progressively through the stack."""
 from eval_awareness.probes import LayerResult
 import numpy as np
 layers = [LayerResult(layer=i, auc=0.5 + 0.02*i, accuracy=0.5 + 0.01*i)
 for i in range(20)]
 w = check_flat_auc_profile(layers, cv_threshold=0.05)
 assert w is None, f"W004 should not have fired for a clear gradient, got {w}"
 print(" ok: W004 absent with progressive AUC gradient")


def test_run_diagnostics_returns_multiple_warnings():
 """run_diagnostics aggregates all applicable warnings."""
 probe = _make_probe_results_flat(n_layers=28, best_auc=1.0, control_auc=0.82)
 warnings = run_diagnostics(
 probe_results=probe,
 n_train=28, # tiny relative to d_model
 d_model=1536,
 layer0_tolerance=0.05,
 control_auc_threshold=0.60,
 sample_ratio_threshold=10.0,
 flat_cv_threshold=0.05,
 )
 codes = {w.code for w in warnings}
 assert "W001" in codes, f"W001 missing from {codes}"
 assert "W002" in codes, f"W002 missing from {codes}"
 assert "W003" in codes, f"W003 missing from {codes}"
 assert "W004" in codes, f"W004 missing from {codes}"
 assert len(warnings) == 4
 print(f" ok: run_diagnostics returned {len(warnings)} warnings: {sorted(codes)}")


def test_run_diagnostics_clean_result():
 """run_diagnostics returns empty list when no warnings fire."""
 from eval_awareness.probes import ProbeResults, LayerResult
 from sklearn.linear_model import LogisticRegression
 from sklearn.preprocessing import StandardScaler
 import numpy as np
 # Build a realistic probe: AUC rises through layers, control near chance
 layers = [LayerResult(layer=i, auc=0.50 + 0.015*i, accuracy=0.5,
 control_auc=0.51)
 for i in range(20)]
 rng = np.random.RandomState(0)
 X = rng.randn(20, 6); y = np.array([1]*10+[0]*10)
 scaler = StandardScaler().fit(X)
 clf = LogisticRegression().fit(scaler.transform(X), y)
 probe = ProbeResults(
 layers=layers, best_layer=19, best_auc=0.785,
 best_selectivity=0.275, scaler=scaler, classifier=clf,
 )
 warnings = run_diagnostics(
 probe_results=probe,
 n_train=5000, # well above threshold
 d_model=256, # small d
 layer0_tolerance=0.05,
 control_auc_threshold=0.60,
 sample_ratio_threshold=10.0,
 flat_cv_threshold=0.05,
 )
 assert warnings == [], f"expected no warnings, got {[w.code for w in warnings]}"
 print(" ok: run_diagnostics returns empty list on a clean result")


def test_diagnostic_warning_serialises():
 """DiagnosticWarning.to_dict() must produce JSON-serialisable output."""
 import json
 w = DiagnosticWarning(code="W001", message="test message", detail="a=1")
 d = w.to_dict()
 assert d["code"] == "W001"
 assert d["message"] == "test message"
 assert d["detail"] == "a=1"
 # Must be JSON-serialisable
 json.dumps(d)
 print(" ok: DiagnosticWarning.to_dict() is JSON-serialisable")


# ==========================================================================
# Probe results now include best_selectivity
# ==========================================================================
def test_probe_results_include_best_selectivity():
 """ProbeResults must expose best_selectivity at the top level."""
 cfg = Config()
 acts, labels = _synthetic_activations(signal_layer=2)
 idx = np.arange(len(labels))
 rng = np.random.RandomState(0)
 rng.shuffle(idx)
 train_idx, test_idx = idx[:30], idx[30:]
 res = train_layer_probes(acts, labels, train_idx, test_idx, cfg)
 # With cfg.run_control_task=True (default), selectivity must be available.
 assert res.best_selectivity is not None, "best_selectivity should not be None"
 assert isinstance(res.best_selectivity, float)
 d = res.to_dict()
 assert "best_selectivity" in d, "to_dict() missing best_selectivity"
 print(f" ok: best_selectivity={res.best_selectivity:.3f} in ProbeResults")


# ==========================================================================
# probe_min_layer excludes early layers from the "best" selection
# ==========================================================================
def test_probe_min_layer_excludes_early_layers():
 """When probe_min_layer > 0, the best layer must be at or above that index,
 even if an excluded earlier layer has a higher AUC."""
 cfg = Config()
 # Put a strong signal at layer 0 (mirroring the layer-0 vocabulary
 # confound), and a real but weaker signal only at layer 2.
 rng = np.random.RandomState(7)
 n_per_class, n_layers, d = 25, 4, 6
 n = n_per_class * 2
 labels = np.array([1] * n_per_class + [0] * n_per_class)
 acts = rng.randn(n_layers, n, d) * 0.5
 acts[0, :n_per_class, 0] += 8.0 # strong signal at layer 0
 acts[2, :n_per_class, 0] += 1.2 # weaker signal at layer 2

 idx = np.arange(len(labels))
 np.random.RandomState(0).shuffle(idx)
 train_idx, test_idx = idx[:30], idx[30:]

 # Default: layer 0 wins (or at worst ties, in which case max() picks it).
 res0 = train_layer_probes(acts, labels, train_idx, test_idx, cfg)
 assert res0.best_layer == 0, (
 f"expected best_layer=0 by default, got {res0.best_layer}. "
 f"Per-layer AUCs: {[(r.layer, r.auc) for r in res0.layers]}"
 )

 # With probe_min_layer=1: layer 0 is excluded from selection.
 cfg.probe_min_layer = 1
 res1 = train_layer_probes(acts, labels, train_idx, test_idx, cfg)
 assert res1.best_layer >= 1, f"expected >=1, got {res1.best_layer}"
 # And the per-layer results still include layer 0 (only selection changed).
 layer_indices = [r.layer for r in res1.layers]
 assert 0 in layer_indices, "layer 0 must still be reported in per-layer results"
 print(f" ok: probe_min_layer=1 shifted best_layer from 0 to {res1.best_layer}")


def test_probe_min_layer_falls_back_when_too_high():
 """If probe_min_layer exceeds all available layers, fall back to all valid
 layers rather than crashing."""
 cfg = Config()
 cfg.probe_min_layer = 999
 acts, labels = _synthetic_activations(signal_layer=2)
 idx = np.arange(len(labels))
 np.random.RandomState(0).shuffle(idx)
 train_idx, test_idx = idx[:30], idx[30:]
 res = train_layer_probes(acts, labels, train_idx, test_idx, cfg)
 # Should not crash; should still return a valid best_layer.
 assert res.best_layer in [r.layer for r in res.layers]
 print(f" ok: probe_min_layer=999 fell back safely to best_layer={res.best_layer}")


# ==========================================================================
def main():
 tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
 passed = failed = 0
 print(f"\nRunning {len(tests)} verification tests\n" + "=" * 62)
 for fn in tests:
 print(f"\n{fn.__name__}")
 try:
 fn()
 passed += 1
 except Exception as exc:
 failed += 1
 print(f" FAIL: {type(exc).__name__}: {exc}")
 import traceback
 traceback.print_exc()
 print("\n" + "=" * 62)
 print(f"passed {passed} / {len(tests)} failed {failed}")
 return 1 if failed else 0


if __name__ == "__main__":
 sys.exit(main())
