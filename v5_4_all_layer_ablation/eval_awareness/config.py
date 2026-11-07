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
    device: str = "auto"          # "auto" | "cuda" | "cpu"
    dtype: str = "float16"        # "float16" | "bfloat16" | "float32"
    trust_remote_code: bool = False
    # 4-bit quantisation via bitsandbytes. Needed to fit ~7B models on a
    # 6 GB laptop GPU. Requires: pip install bitsandbytes accelerate
    load_in_4bit: bool = False
    # Force EVERY layer onto GPU 0, overriding accelerate's automatic
    # device-map planning. Use when accelerate refuses an all-GPU load that
    # you believe will just fit. Will hard-OOM at allocation time if it does
    # not; there is no CPU fallback.
    force_gpu_only: bool = False

    # ---------------- data ----------------
    # If None, the built-in prompt set in prompts.py is used.
    sorry_bench_path: Optional[str] = None
    # SORRY-Bench categories used (12 = System Intrusion, 13 = Malware Code Generation)
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
    run_control_task: bool = True   # Belinkov (2022) selectivity check
    probe_pca_components: Optional[int] = 50
    probe_min_layer: int = 0

    # ---------------- behaviour ----------------
    max_new_tokens: int = 128
    generation_temperature: float = 0.0   # 0.0 => greedy, deterministic

    # ---------------- suppression ----------------
    # In v5_4 these alphas are consumed by AllLayerSteeringHook, which
    # applies directional ablation at EVERY transformer layer's residual
    # stream simultaneously (Arditi et al. 2024). alpha semantics are
    # identical to the single-layer build: 0.0 = no intervention,
    # 1.0 = the direction is fully erased at every layer, >1.0 over-steers
    # past zero.
    steering_alphas: List[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    )

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
