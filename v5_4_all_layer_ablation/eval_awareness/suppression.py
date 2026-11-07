"""Activation steering and the cosmetic-versus-real suppression audit.

This is the central contribution of the project. Equations refer to Chapter 3:

 (3.2) dAUC = (AUC_base - AUC_final) / (AUC_base - 0.5)
 (3.3) dG = (G_base - G_final) / G_base

Verdict:
 cosmetic -> dAUC > cosmetic_auc_threshold AND dG < real_gap_threshold
 real -> dG >= real_gap_threshold
 inconclusive -> otherwise

--------------------------------------------------------------------------
V5_4 CHANGE (all-layer ablation, replication):

This build swaps the SINGLE-LAYER directional ablation (SteeringHook, at
the probe's best layer only) for the ALL-LAYER ablation used by (2024), "Refusal in Language Models is Mediated by a Single
Direction" (arXiv 2406.11717).

The projection formula per layer is unchanged:

 x'_l = x_l - alpha * (x_l . d_hat) * d_hat

but the intervention is now applied at EVERY transformer decoder layer's
residual stream simultaneously, not just the single best layer. This
removes any downstream "recovery" of the direction from later layers'
attention or MLP outputs.

The original authors also perform the intervention by editing the model's
weight matrices (W_O at every attention block, W_down at every MLP
block, and the embedding matrix) so that no writer can add anything
along d to the residual stream. That variant is equivalent to all-layer
runtime ablation with alpha=1.0 fixed. Runtime hooks are used here so
that the pipeline's alpha sweep still functions and results remain
comparable to the ablation and weight-edit builds.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Any, Dict
import numpy as np
import torch

from .activations import get_decoder_layers, _hidden_from_output


class SteeringHook:
 """Single-layer directional ablation (v5_2 build).

 Kept in this module for reference. The v5_4 pipeline no longer calls
 it -- see AllLayerSteeringHook below.

 x' = x - alpha * (x . d) d
 """

 def __init__(self, model, layer_idx: int, direction: np.ndarray, alpha: float):
 layers = get_decoder_layers(model)
 if not 0 <= layer_idx < len(layers):
 raise IndexError(
 f"layer_idx {layer_idx} out of range for {len(layers)} layers"
 )
 self.layer = layers[layer_idx]

 d = np.asarray(direction, dtype=np.float32)
 if d.ndim != 1:
 raise ValueError("direction must be a 1-D vector")
 norm = float(np.linalg.norm(d))
 if norm < 1e-12:
 raise ValueError("direction has zero norm")
 self.direction = d / norm
 self.alpha = float(alpha)
 self._handle: Optional[Any] = None

 def _hook(self, _module, _inputs, output):
 hidden = _hidden_from_output(output)
 vec = torch.from_numpy(self.direction).to(
 device=hidden.device, dtype=hidden.dtype
 )
 if vec.shape[0] != hidden.shape[-1]:
 raise ValueError(
 f"direction dim {vec.shape[0]} != hidden dim {hidden.shape[-1]}"
 )
 proj = (hidden * vec).sum(dim=-1, keepdim=True)
 steered = hidden - self.alpha * proj * vec
 if isinstance(output, tuple):
 return (steered,) + output[1:]
 return steered

 def __enter__(self) -> "SteeringHook":
 self._handle = self.layer.register_forward_hook(self._hook)
 return self

 def __exit__(self, *exc) -> None:
 if self._handle is not None:
 self._handle.remove()
 self._handle = None


# ---------------------------------------------------------------------------
# all-layer ablation
# ---------------------------------------------------------------------------


class AllLayerSteeringHook:
 """Directional ablation applied at EVERY transformer decoder layer.

 For each layer l of the model, we register
 a forward hook that projects out the direction d from the layer's
 residual-stream output:

 x'_l = x_l - alpha * (x_l . d_hat) * d_hat

 All hooks share the same direction and alpha. On __exit__ every hook
 is removed, so nested / sequential sweeps do not compound.

 Why every layer, not just one:
 A signal that lives in the residual stream at layer L may have been
 written by any earlier attention or MLP block. Removing it at only
 layer L lets later blocks re-introduce it. Ablating at every layer
 ensures that no downstream writer can carry the direction forward,
 matching what is needed for behavioural change.

 Notes:
 - The projection is applied to the RESIDUAL STREAM, i.e. the block
 output that gets read at the corresponding layer index in the
 activation cache. This is where the direction was extracted from.
 - alpha semantics are identical to SteeringHook: 0.0 = no
 intervention, 1.0 = full ablation, >1.0 over-corrects past zero
 (marked oversteered).
 - Cost is O(N_layers) hooks per forward pass. Each hook is a
 rank-1 subtraction, so overhead is small.
 """

 def __init__(self, model, direction: np.ndarray, alpha: float):
 self.layers = list(get_decoder_layers(model))
 if not self.layers:
 raise RuntimeError("no decoder layers found on model")

 d = np.asarray(direction, dtype=np.float32)
 if d.ndim != 1:
 raise ValueError("direction must be a 1-D vector")
 norm = float(np.linalg.norm(d))
 if norm < 1e-12:
 raise ValueError("direction has zero norm")
 self.direction = d / norm
 self.alpha = float(alpha)
 self._handles: List[Any] = []

 def _make_hook(self):
 vec_np = self.direction
 alpha = self.alpha

 def _hook(_module, _inputs, output):
 hidden = _hidden_from_output(output)
 vec = torch.from_numpy(vec_np).to(
 device=hidden.device, dtype=hidden.dtype
 )
 if vec.shape[0] != hidden.shape[-1]:
 raise ValueError(
 f"direction dim {vec.shape[0]} != hidden dim {hidden.shape[-1]}"
 )
 proj = (hidden * vec).sum(dim=-1, keepdim=True)
 steered = hidden - alpha * proj * vec
 if isinstance(output, tuple):
 return (steered,) + output[1:]
 return steered

 return _hook

 def __enter__(self) -> "AllLayerSteeringHook":
 hook_fn = self._make_hook()
 self._handles = [layer.register_forward_hook(hook_fn) for layer in self.layers]
 return self

 def __exit__(self, *exc) -> None:
 for h in self._handles:
 try:
 h.remove()
 except Exception:
 pass
 self._handles = []


# ---------------------------------------------------------------------------
# Sweep summary (unchanged)
# ---------------------------------------------------------------------------


def proportional_auc_reduction(auc_base: float, auc_final: float) -> float:
 """Equation 3.2."""
 denom = auc_base - 0.5
 if denom <= 1e-12:
 return 0.0
 return float((auc_base - auc_final) / denom)


def is_oversteered(d_auc: float) -> bool:
 return d_auc > 1.0


def proportional_gap_reduction(gap_base: float, gap_final: float) -> float:
 """Equation 3.3."""
 if gap_base <= 1e-12:
 return 0.0
 return float((gap_base - gap_final) / gap_base)


def classify_verdict(
 d_auc: float, d_gap: float, cosmetic_auc_threshold: float, real_gap_threshold: float
) -> str:
 if d_gap >= real_gap_threshold:
 return "real"
 if d_auc > cosmetic_auc_threshold:
 return "cosmetic"
 return "inconclusive"


@dataclass
class SweepPoint:
 alpha: float
 auc: float
 gap: float
 d_auc: float
 d_gap: float
 verdict: str
 oversteered: bool = False


@dataclass
class SuppressionResults:
 baseline_auc: float
 baseline_gap: float
 points: List[SweepPoint]
 overall_verdict: str

 def to_dict(self) -> Dict:
 return {
 "baseline_auc": self.baseline_auc,
 "baseline_gap": self.baseline_gap,
 "overall_verdict": self.overall_verdict,
 "sweep": [asdict(p) for p in self.points],
 }


def summarise_sweep(
 baseline_auc: float,
 baseline_gap: float,
 alphas: List[float],
 aucs: List[float],
 gaps: List[float],
 cfg,
) -> SuppressionResults:
 if not (len(alphas) == len(aucs) == len(gaps)):
 raise ValueError("alphas, aucs and gaps must be the same length")
 if not alphas:
 raise ValueError("Empty sweep")

 points: List[SweepPoint] = []
 for alpha, auc, gap in zip(alphas, aucs, gaps):
 d_auc = proportional_auc_reduction(baseline_auc, auc)
 d_gap = proportional_gap_reduction(baseline_gap, gap)
 points.append(
 SweepPoint(
 alpha=float(alpha),
 auc=float(auc),
 gap=float(gap),
 d_auc=d_auc,
 d_gap=d_gap,
 verdict=classify_verdict(
 d_auc, d_gap, cfg.cosmetic_auc_threshold, cfg.real_gap_threshold
 ),
 oversteered=is_oversteered(d_auc),
 )
 )

 effective = [p for p in points if p.alpha > 0]
 verdicts = {p.verdict for p in effective}
 if "real" in verdicts:
 overall = "real"
 elif "cosmetic" in verdicts:
 overall = "cosmetic"
 else:
 overall = "inconclusive"

 return SuppressionResults(baseline_auc, baseline_gap, points, overall)
