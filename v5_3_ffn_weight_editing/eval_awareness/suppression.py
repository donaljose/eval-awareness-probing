"""Activation steering and the cosmetic-versus-real suppression audit.

This is the central contribution of the project. Equations refer to Chapter 3:

 (3.2) dAUC = (AUC_base - AUC_final) / (AUC_base - 0.5)
 (3.3) dG = (G_base - G_final) / G_base

Verdict:
 cosmetic -> dAUC > cosmetic_auc_threshold AND dG < real_gap_threshold
 real -> dG >= real_gap_threshold
 inconclusive -> otherwise

--------------------------------------------------------------------------
V5_3 CHANGE (weight-edit variant):
This build swaps the DIRECTIONAL ABLATION intervention (SteeringHook) for
the WEIGHT-EDIT method of 14617). Instead of removing the direction from activations at
inference time, we permanently edit the FFN up-projection weights of the
target layer for the duration of the intervention:

 w_1,k <- w_1,k - alpha * d for the top-K rows of W_1 most
 cosine-aligned with d

The rest of the pipeline (probing, difference-of-means direction, alpha
sweep, verdict system) is unchanged.

Sign convention: this pipeline's alpha > 0 means "suppress awareness", to
stay consistent with the ablation build. The original
formula is w_1,k <- w_1,k + alpha * d with alpha < 0 to suppress. We flip
the sign internally, so the caller passes positive alpha for suppression
whichever intervention is in use.

Calibrated range: |alpha| < 0.35. Values from the
default config's steering_alphas (up to 1.5) are useful for observing
severe over-correction and capability collapse, but should not be read as
"stronger ablation" -- they are qualitatively a different regime.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Any, Dict
import numpy as np
import torch

from .activations import get_decoder_layers, _hidden_from_output


class SteeringHook:
 """Erases the evaluation-awareness direction from a layer's residual stream.

 Kept in this module for reference and for tests that still exercise the
 ablation path. The v5_3 pipeline no longer calls it -- see
 WeightEditSteering below.

 Implements DIRECTIONAL ABLATION: for each
 token position, the component of the activation lying along the unit
 direction d is removed, rather than subtracting a constant multiple of d.

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
# parameter editing
# ---------------------------------------------------------------------------


def _find_up_projection(mlp_module) -> torch.nn.Module:
 """Locate the FFN up-projection weight matrix W_1 across common
 architectures. Returns the Linear (or bnb-quantised) submodule whose
 .weight has shape (intermediate_size, hidden_size)."""
 for name in ("up_proj", "gate_proj", "fc1", "w1"):
 if hasattr(mlp_module, name):
 return getattr(mlp_module, name)
 # Fallback: first Linear-shaped child that reads from hidden dim.
 for _, child in mlp_module.named_children():
 w = getattr(child, "weight", None)
 if isinstance(w, torch.Tensor) and w.ndim == 2:
 return child
 raise AttributeError(
 f"Could not find FFN up-projection in {type(mlp_module).__name__}"
 )


class WeightEditSteering:
 """Suppress evaluation-awareness by editing MLP weights in place.

 Follows . At the target layer, we take the FFN
 up-projection W_1 (shape (intermediate_size, hidden_size)), compute the
 absolute cosine similarity between each row and the probe direction d,
 keep the top-K rows, and shift each of them:

 w_1,k <- w_1,k - alpha * d (alpha > 0 suppresses)

 The original weights are cloned on __enter__ and restored on __exit__,
 so any exception inside the block still leaves the model untouched.

 Sign flip: their paper uses alpha < 0 to suppress. This build takes
 alpha > 0 to suppress, matching the rest of the pipeline. Positive
 alpha therefore SUBTRACTS the direction from the selected rows.

 Note on quantised weights: on bitsandbytes 4-bit or 8-bit layers the
 .weight tensor is not a plain float tensor and cannot be edited in
 place this way. In that case this class raises a clear error so the
 caller can fall back or reload in full precision.
 """

 def __init__(
 self,
 model,
 layer_idx: int,
 direction: np.ndarray,
 alpha: float,
 top_k: int = 800,
 ):
 layers = get_decoder_layers(model)
 if not 0 <= layer_idx < len(layers):
 raise IndexError(
 f"layer_idx {layer_idx} out of range for {len(layers)} layers"
 )
 self.layer = layers[layer_idx]

 if not hasattr(self.layer, "mlp"):
 raise AttributeError(
 f"Layer {layer_idx} has no .mlp submodule "
 f"(type: {type(self.layer).__name__}); this architecture is "
 f"not supported by WeightEditSteering."
 )
 self.target = _find_up_projection(self.layer.mlp)

 d = np.asarray(direction, dtype=np.float32)
 if d.ndim != 1:
 raise ValueError("direction must be a 1-D vector")
 norm = float(np.linalg.norm(d))
 if norm < 1e-12:
 raise ValueError("direction has zero norm")
 self.direction = d / norm

 self.alpha = float(alpha)
 self.top_k = int(top_k)

 self._original_weight: Optional[torch.Tensor] = None
 self._edited_indices: Optional[torch.Tensor] = None

 def __enter__(self) -> "WeightEditSteering":
 W = self.target.weight
 if not isinstance(W, torch.Tensor) or W.ndim != 2 or W.dtype not in (
 torch.float16, torch.bfloat16, torch.float32
 ):
 raise TypeError(
 f"WeightEditSteering requires a plain 2-D float weight "
 f"tensor; got {type(W).__name__} dtype={getattr(W, 'dtype', None)}. "
 f"This usually means the model was loaded in a quantised "
 f"format (bitsandbytes 4-bit/8-bit). Reload in float16 or "
 f"bfloat16 to use weight editing."
 )

 direction = torch.from_numpy(self.direction).to(
 device=W.device, dtype=W.dtype
 )
 if direction.shape[0] != W.shape[1]:
 raise ValueError(
 f"direction dim {direction.shape[0]} != W input dim {W.shape[1]}"
 )

 with torch.no_grad():
 # Cosine similarity between each row of W and the direction.
 row_norms = W.norm(dim=1) + 1e-8
 cos_sim = (W @ direction) / row_norms
 k = min(self.top_k, W.shape[0])
 top_k_idx = torch.topk(cos_sim.abs(), k=k).indices
 self._edited_indices = top_k_idx

 # Clone before modifying so we can restore exactly.
 self._original_weight = W.data[top_k_idx].detach().clone()

 # Positive alpha here suppresses; sign flipped from the paper.
 update = self.alpha * direction # shape: (hidden_size,)
 W.data[top_k_idx] = W.data[top_k_idx] - update.unsqueeze(0)

 return self

 def __exit__(self, *exc) -> None:
 if self._original_weight is not None and self._edited_indices is not None:
 with torch.no_grad():
 self.target.weight.data[self._edited_indices] = self._original_weight
 self._original_weight = None
 self._edited_indices = None


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
