"""Automatic diagnostic checks on probing results.

Run immediately after train_layer_probes to detect known artefacts before
interpreting AUC numbers. All four checks are independent; any non-None
return from a check is a warning worth reporting.

Warning codes
─────────────
W001 Layer-0 AUC near best AUC — surface/keyword confound likely.
W002 Control AUC well above chance at the best layer — selectivity failure; the probe can separate any two groups.
W003 Sample-size ratio n_train/d_model too low — high-dimensional
 regime where any labelling is linearly separable.
W004 AUC is flat across all layers — no representational gradient;
 signal may be in surface wording, not residual-stream geometry.

Usage
─────
 from eval_awareness.diagnostics import run_diagnostics
 warnings = run_diagnostics(probe_results, n_train=n_train, d_model=d_model)
 for w in warnings:
 print(w.code, w.message)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
import numpy as np

from .probes import ProbeResults, LayerResult


# --------------------------------------------------------------------------
# Warning dataclass
# --------------------------------------------------------------------------

@dataclass
class DiagnosticWarning:
 """One diagnostic warning.

 code: W001 – W004
 message: Human-readable explanation suitable for a paper appendix.
 detail: Compact numeric summary for the app's Summary tab.
 """
 code: str
 message: str
 detail: Optional[str] = None

 def to_dict(self) -> Dict:
 return asdict(self)


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------

def check_layer0_confound(
 layers: List[LayerResult],
 tolerance: float = 0.05,
) -> Optional[DiagnosticWarning]:
 """W001: flag when the embedding-layer (layer 0) AUC is within *tolerance*
 of the best AUC across all layers.

 Layer 0 follows immediately from the embedding lookup — no transformer
 block has run. An evaluation-awareness direction cannot be linearly
 decodable there from residual-stream geometry because there is no geometry
 yet; only the token embeddings exist. A near-best AUC at layer 0 therefore
 indicates that the probe is reading surface properties of the framing
 templates (vocabulary, length encoding) rather than a latent direction that
 forms during the forward pass. See the selectivity control literature for the analogous
 diagnostic in probing literature.
 """
 valid = [r for r in layers if not np.isnan(r.auc)]
 if len(valid) < 2:
 return None
 best_auc = max(r.auc for r in valid)
 layer0 = next((r for r in valid if r.layer == 0), None)
 if layer0 is None:
 return None
 if (best_auc - layer0.auc) <= tolerance:
 return DiagnosticWarning(
 code="W001",
 message=(
 "Layer-0 AUC ({l0:.3f}) is within {tol:.3f} of the best AUC "
 "across all layers ({best:.3f}). The probe signal is present at "
 "the embedding layer, before any transformer block has run. This "
 "indicates the templates differ in surface vocabulary or length "
 "rather than encoding a genuine evaluation-awareness direction in "
 "the residual stream. Consider using length-matched templates "
 "and re-running with framing_length_report() to confirm.".format(
 l0=layer0.auc, tol=tolerance, best=best_auc
 )
 ),
 detail=f"layer0_auc={layer0.auc:.3f}, best_auc={best_auc:.3f}, gap={best_auc - layer0.auc:.3f}",
 )
 return None


def check_high_control_auc(
 layers: List[LayerResult],
 threshold: float = 0.60,
) -> Optional[DiagnosticWarning]:
 """W002: flag when the control probe (shuffled labels) AUC at the best
 layer exceeds *threshold*.

 A probe trained on shuffled labels should score
 at chance (0.5) on held-out examples: any partition of a high-dimensional
 space is structurally separable, but only when RANDOMLY labelled examples
 from BOTH framings land on the same probe-positive side. A control AUC
 substantially above 0.5 means the probe can separate arbitrary label
 assignments at this layer, which inflates the main AUC and makes the
 selectivity score — (main AUC) − (control AUC) — a more honest primary
 metric than raw AUC alone.
 """
 valid = [r for r in layers if not np.isnan(r.auc)]
 if not valid:
 return None
 best_auc = max(r.auc for r in valid)
 best_layer = next((r for r in valid if abs(r.auc - best_auc) < 1e-9), None)
 if best_layer is None or best_layer.control_auc is None:
 return None
 if best_layer.control_auc > threshold:
 sel = best_layer.selectivity # may be None if control_auc is None
 sel_str = f"{sel:.3f}" if sel is not None else "n/a"
 return DiagnosticWarning(
 code="W002",
 message=(
 "Control probe AUC at the best layer (layer {l}) is {ctrl:.3f}, "
 "which is substantially above chance (0.5). The selectivity "
 "selectivity score is {sel}. A control AUC above {t} indicates "
 "that a logistic probe can separate arbitrary groupings of these "
 "activations, so the raw AUC of {main:.3f} overstates the "
 "evidence for a genuine evaluation-awareness direction. Prefer "
 "selectivity as the primary metric, and consider increasing "
 "probe regularisation (lower probe_C) or adding PCA.".format(
 l=best_layer.layer,
 ctrl=best_layer.control_auc,
 sel=sel_str,
 t=threshold,
 main=best_auc,
 )
 ),
 detail=(
 f"best_layer={best_layer.layer}, main_auc={best_auc:.3f}, "
 f"control_auc={best_layer.control_auc:.3f}, selectivity={sel_str}"
 ),
 )
 return None


def check_sample_size_ratio(
 n_train: int,
 d_model: int,
 threshold: float = 10.0,
) -> Optional[DiagnosticWarning]:
 """W003: flag when n_train / d_model falls below *threshold*.

 By the Sauer–Shelah lemma, a set of N points in R^d can be shattered by a
 linear classifier when N ≲ d. In practice, AUC of 1.0 is achievable with
 far fewer than d points simply because the training set is too small for
 the probe to overfit to be detected; the generalisation gap is zero only
 because the test set is also tiny. A commonly cited rule of thumb is at
 least ten labelled examples per input dimension. Below this threshold, even
 a control probe trained on random labels will frequently achieve high test
 AUC — this is the mechanism behind W002. The primary fix is a larger prompt
 set; a secondary fix is PCA to reduce d before probing.
 """
 if d_model <= 0:
 return None
 ratio = n_train / d_model
 if ratio < threshold:
 return DiagnosticWarning(
 code="W003",
 message=(
 "Training set contains {n} examples in a {d}-dimensional "
 "activation space (ratio {r:.4f}, threshold {t}). In this "
 "high-dimensional regime, even randomly labelled points are "
 "frequently linearly separable, so an AUC near 1.0 does not "
 "provide evidence that the framing direction is genuinely encoded. "
 "Increase the prompt set size or reduce dimensionality with PCA "
 "(set probe_pca_components in Config).".format(
 n=n_train, d=d_model, r=ratio, t=threshold
 )
 ),
 detail=f"n_train={n_train}, d_model={d_model}, ratio={ratio:.4f}",
 )
 return None


def check_flat_auc_profile(
 layers: List[LayerResult],
 cv_threshold: float = 0.05,
) -> Optional[DiagnosticWarning]:
 """W004: flag when AUC shows little variation across layers.

 If evaluation-awareness is genuinely encoded as a direction in the residual
 stream, it is expected to become more linearly decodable as information
 propagates through the transformer stack . A flat AUC profile — identical or near-identical AUC
 at all 28 layers — suggests instead that the probe is reading a property
 present before any computation (e.g. surface vocabulary), or that the
 signal is so strong that it saturates all layers simultaneously, which is
 also consistent with a confound.

 The coefficient of variation (std / mean) is used as the flatness metric
 because it is scale-invariant; a CV below *cv_threshold* indicates that
 the between-layer spread is negligible relative to the mean level.
 """
 aucs = [r.auc for r in layers if not np.isnan(r.auc)]
 if len(aucs) < 3:
 return None
 arr = np.array(aucs)
 mean_auc = float(arr.mean())
 std_auc = float(arr.std())
 cv = std_auc / mean_auc if mean_auc > 1e-9 else 0.0
 if cv < cv_threshold:
 return DiagnosticWarning(
 code="W004",
 message=(
 "AUC is near-uniform across all {n} layers (mean={m:.3f}, "
 "std={s:.4f}, CV={cv:.4f} < threshold {t}). A genuine "
 "evaluation-awareness direction should become increasingly "
 "decodable through the transformer stack; uniformly high AUC "
 "at every layer, including layer 0, is more consistent with a "
 "surface confound (vocabulary, prompt length) than with a "
 "depth-dependent latent representation.".format(
 n=len(aucs), m=mean_auc, s=std_auc, cv=cv, t=cv_threshold
 )
 ),
 detail=f"n_layers={len(aucs)}, auc_mean={mean_auc:.3f}, auc_std={std_auc:.4f}, cv={cv:.4f}",
 )
 return None


# --------------------------------------------------------------------------
# Aggregator
# --------------------------------------------------------------------------

def run_diagnostics(
 probe_results: ProbeResults,
 n_train: int,
 d_model: int,
 layer0_tolerance: float = 0.05,
 control_auc_threshold: float = 0.60,
 sample_ratio_threshold: float = 10.0,
 flat_cv_threshold: float = 0.05,
) -> List[DiagnosticWarning]:
 """Run all four checks and return a (possibly empty) list of warnings.

 Parameters
 ----------
 probe_results:
 Output of train_layer_probes.
 n_train:
 Number of training *examples* (not pairs); used for W003.
 d_model:
 Hidden-state dimensionality of the model; used for W003.
 layer0_tolerance:
 W001 fires when best_auc − layer0_auc ≤ this value.
 control_auc_threshold:
 W002 fires when control AUC at best layer > this value.
 sample_ratio_threshold:
 W003 fires when n_train / d_model < this value.
 flat_cv_threshold:
 W004 fires when AUC coefficient-of-variation < this value.
 """
 checks = [
 check_layer0_confound(probe_results.layers, layer0_tolerance),
 check_high_control_auc(probe_results.layers, control_auc_threshold),
 check_sample_size_ratio(n_train, d_model, sample_ratio_threshold),
 check_flat_auc_profile(probe_results.layers, flat_cv_threshold),
 ]
 return [w for w in checks if w is not None]
