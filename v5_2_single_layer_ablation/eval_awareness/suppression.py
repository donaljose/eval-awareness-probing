"""Activation steering and the cosmetic-versus-real suppression audit.

This is the central contribution of the project. Equations refer to Chapter 3:

 (3.2) dAUC = (AUC_base - AUC_final) / (AUC_base - 0.5)
 (3.3) dG = (G_base - G_final) / G_base

Verdict:
 cosmetic -> dAUC > cosmetic_auc_threshold AND dG < real_gap_threshold
 real -> dG >= real_gap_threshold
 inconclusive -> otherwise
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Any, Dict
import numpy as np
import torch

from .activations import get_decoder_layers, _hidden_from_output


class SteeringHook:
 """Erases the evaluation-awareness direction from a layer's residual stream.

 Implements DIRECTIONAL ABLATION: for each
 token position, the component of the activation lying along the unit
 direction d is removed, rather than subtracting a constant multiple of d.

 x' = x - alpha * (x . d) d

 At alpha = 1.0 the component along d is fully erased; alpha < 1 partially
 erases it; alpha > 1 over-corrects past zero and inverts the component.

 Why not `x - alpha * d`? That is a uniform translation of every activation
 by the same vector. AUC is rank-based, so a uniform translation cannot
 change probe AUC at all -- the sweep would report no effect regardless of
 alpha, which would be a measurement artefact rather than a finding.
 Ablation removes a per-example quantity ((x . d) differs by example) and
 therefore genuinely changes separability.
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
 # Normalise defensively: the projection formula assumes a unit vector.
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
 # projection coefficient per position: (..., 1)
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


def proportional_auc_reduction(auc_base: float, auc_final: float) -> float:
 """Equation 3.2. Normalised by (AUC_base - 0.5) because 0.5 is chance,
 so the recoverable signal is what sits ABOVE chance, not above zero.

 Returns 0.0 when the baseline probe was already at/below chance: there was
 no signal to remove, so no reduction can be claimed.

 NOTE ON VALUES > 1.0: if steering is strong enough to push the probe BELOW
 chance (AUC < 0.5), the probe has been inverted rather than merely
 neutralised, and this returns a value greater than 1.0. That is deliberate.
 Clamping to 1.0 would hide over-steering, which is a real experimental
 condition worth seeing: it indicates alpha is large enough to be distorting
 the representation rather than cleanly ablating the direction. Treat
 dAUC > 1.0 as a signal to inspect that sweep point, not as a stronger
 version of the same result.
 """
 denom = auc_base - 0.5
 if denom <= 1e-12:
 return 0.0
 return float((auc_base - auc_final) / denom)


def is_oversteered(d_auc: float) -> bool:
 """True when steering inverted the probe rather than neutralising it."""
 return d_auc > 1.0


def proportional_gap_reduction(gap_base: float, gap_final: float) -> float:
 """Equation 3.3.

 Returns 0.0 when there was no baseline gap to close.
 """
 if gap_base <= 1e-12:
 return 0.0
 return float((gap_base - gap_final) / gap_base)


def classify_verdict(
 d_auc: float, d_gap: float, cosmetic_auc_threshold: float, real_gap_threshold: float
) -> str:
 """Order matters: 'real' is checked first, so a genuine behavioural change
 is never mislabelled cosmetic merely because the probe signal also fell."""
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
 """Build the sweep table and derive one overall verdict.

 Overall verdict rule: if ANY steering strength produced a real behavioural
 change, the effect is real. Otherwise, if any strength stripped the probe
 signal without moving behaviour, it is cosmetic. Otherwise inconclusive.
 """
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

 # alpha == 0 is the unsteered control and cannot itself be evidence.
 effective = [p for p in points if p.alpha > 0]
 verdicts = {p.verdict for p in effective}
 if "real" in verdicts:
 overall = "real"
 elif "cosmetic" in verdicts:
 overall = "cosmetic"
 else:
 overall = "inconclusive"

 return SuppressionResults(baseline_auc, baseline_gap, points, overall)
