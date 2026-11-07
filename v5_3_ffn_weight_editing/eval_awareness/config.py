"""Central configuration for the evaluation-awareness pipeline.

Every tunable lives here so that a run is reproducible from a single object.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


@dataclass
class Config:
 # ---------------- model ----------------
 model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
 device: str = "auto"
 dtype: str = "float16"
 trust_remote_code: bool = False
 load_in_4bit: bool = False
 force_gpu_only: bool = False

 # ---------------- data ----------------
 sorry_bench_path: Optional[str] = None
 sorry_bench_categories: List[int] = field(default_factory=lambda: [12, 13])
 test_size: float = 0.3
 seed: int = 42

 # ---------------- activations ----------------
 token_position: int = -1
 max_prompt_tokens: int = 512
 batch_size: int = 8

 # ---------------- probing ----------------
 probe_C: float = 0.01
 probe_max_iter: int = 2000
 run_control_task: bool = True

 probe_pca_components: Optional[int] = 50
 probe_min_layer: int = 0

 # ---------------- behaviour ----------------
 max_new_tokens: int = 128
 generation_temperature: float = 0.0

 # ---------------- suppression ----------------
 # In v5_3 these alphas are consumed by WeightEditSteering, not
 # directional ablation. Positive alpha still means "suppress
 # awareness" (sign flipped internally from the original
 # convention). Their calibrated range was |alpha| < 0.35; the values
 # above are kept for continuity with the ablation build so both
 # regimes are visible in one sweep. Expect large-alpha points to be
 # over-corrections that degrade general capability.
 steering_alphas: List[float] = field(
 default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
 )

 # Number of FFN up-projection rows edited per intervention.
 # used 800 (~2.5% of a 32B model's
 # intermediate width). For smaller models this may edit a larger
 # relative fraction; if the model collapses at all alphas, reduce.
 weight_edit_top_k: int = 800

 cosmetic_auc_threshold: float = 0.5
 real_gap_threshold: float = 0.3

 # ---------------- diagnostics ----------------
 diag_layer0_tolerance: float = 0.05
 diag_control_auc_threshold: float = 0.60
 diag_sample_ratio_threshold: float = 10.0
 diag_flat_cv_threshold: float = 0.05

 # ---------------- output ----------------
 output_dir: str = "results"

 def to_json(self) -> str:
 return json.dumps(asdict(self), indent=2)


DEFAULT_CONFIG = Config()
