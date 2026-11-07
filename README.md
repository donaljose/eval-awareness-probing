# Evaluation-Awareness Probing Pipeline

Companion code for the paper:

> **Probing Whether a Language Model Knows It Is Being Tested:
> Linear Probing and Directional Ablation of Evaluation-Awareness Representations
> in Small Language Models**
>
> Donal Jose and Sumaiya Thaseen 
> De Montfort University, Dubai

## Overview

This repository contains the full experimental pipeline used to investigate
whether small language models encode a detectable internal signal that
distinguishes evaluation-framed from deployment-framed inputs — and whether
suppressing that signal closes the behavioural gap between the two framings.

### Research question

> *Do language models behave differently when they detect they are being
> evaluated, and can the internal representation responsible for that
> difference be located and erased?*

### Key finding

On **Qwen1.5-1.8B** under **all-layer directional ablation** (v5_4), the
pipeline produces a **REAL non-oversteered** verdict: both probe AUC and the
behavioural gap fall in parallel as suppression strength increases, confirming
that the model's refusal behaviour is causally linked to a linearly separable
evaluation-awareness direction in the residual stream — not merely a surface
artefact.

## Repository structure

The pipeline was developed across three intervention scopes, each in its own
directory. All three share the same seven-stage backbone (Stages 1–4, 6–7) and
differ only in the suppression method (Stage 5):

```
eval-awareness-probing/
├── v5_2_single_layer_ablation/ # Scope 1: directional ablation at best probe layer
├── v5_3_ffn_weight_editing/ # Scope 2: feed-forward weight editing (top-K rows)
├── v5_4_all_layer_ablation/ # Scope 3: all-layer ablation 
└── README.md
```

Each directory has the same module layout:

```
eval_awareness/
├── __init__.py
├── config.py # Centralised configuration (all hyperparameters)
├── prompts.py # Stage 1 — 574-pair corpus construction
├── activations.py # Stage 2 — residual-stream activation caching
├── probes.py # Stage 3 — linear probing (logistic regression)
├── behaviour.py # Stage 4 — behavioural gap measurement
├── suppression.py # Stage 5 — intervention sweep (varies by scope)
├── diagnostics.py # Stage 6 — diagnostic warnings (W001–W004)
├── run_pipeline.py # Stage 7 — orchestrator, verdict, JSON output
└── tests/
 ├── test_pipeline.py
 └── test_orchestration.py
```

### Intervention scopes

| Directory | Method | Reference |
|-----------|--------|-----------|
| `v5_2_single_layer_ablation` | Directional ablation at the single best probing layer | — |
| `v5_3_ffn_weight_editing` | Editing top-K FFN rows by cosine alignment with the probe direction | — |
| `v5_4_all_layer_ablation` | All-layer directional ablation at every decoder block | |

## Pipeline stages

1. **Prompt construction** — builds 574 length-matched pairs (287 unsafe + 287 benign × eval/deploy framings); mean |Δchars| = 1.33
2. **Activation caching** — extracts residual-stream activations at the final token position from every decoder layer
3. **Linear probing** — trains a per-layer logistic-regression probe to classify eval vs. deploy; selects the best layer by AUC
4. **Behavioural testing** — measures refusal rates per framing and computes the baseline behavioural gap *G*
5. **Suppression sweep** — applies the intervention at varying strengths α ∈ [0, 1] and re-measures AUC and *G*
6. **Diagnostics** — fires warnings W001 (layer-0 confound), W002 (dropped AUC), W003 (small sample ratio), W004 (flat AUC profile)
7. **Verdict** — classifies the outcome as REAL, COSMETIC, or INCONCLUSIVE using ρ_A ≥ 0.5 and ρ_G ≥ 0.3

## Models tested

| Model | Parameters | Layers | Hidden dim |
|-------|-----------|--------|------------|
| Qwen2.5-1.5B-Instruct | 1.5 B | 28 | 1536 |
| Qwen1.5-1.8B-Chat | 1.8 B | 24 | 2048 |
| Gemma-2-2B-it | 2.0 B | 26 | 2304 |
| TinyLlama-1.1B-Chat-v1.0 | 1.1 B | 22 | 2048 |

## Quickstart

```bash
# Clone and enter the v5_4 directory (the headline finding)
git clone https://github.com/donaljose/eval-awareness-probing.git
cd eval-awareness-probing/v5_4_all_layer_ablation

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline on Qwen1.5-1.8B (requires a CUDA GPU)
python -m eval_awareness.run_pipeline \
 --model Qwen/Qwen1.5-1.8B-Chat \
 --output results.json

# Run the test suite
bash run_all_tests.sh
```

### Configuration

All hyperparameters are centralised in `config.py`. Key options:

```python
Config(
 model_name="Qwen/Qwen1.5-1.8B-Chat",
 device="auto", # auto | cuda | cpu
 dtype="float16", # float16 | bfloat16 | float32
 load_in_4bit=False, # 4-bit quantisation for larger models
 test_size=0.3, # probe train/test split
 seed=42, # global random seed
 alpha_values=[0.0, 0.25, 0.5, 0.75, 1.0], # suppression sweep
 cosmetic_auc_threshold=0.5, # ρ_A threshold for verdict
 real_gap_threshold=0.3, # ρ_G threshold for verdict
)
```

## Corpus

The 574-pair prompt set is constructed from four public benchmarks and
researcher-authored benign items:

- **SORRY-Bench** — categories 12 (System Intrusion) and 13 (Malware Generation)
- **AdvBench**
- **HarmBench**
- **MaliciousInstruct**
- 287 benign requests (researcher-authored, covering coding, writing, and general knowledge)

All pairs are length-matched: the same base request appears under both
evaluation and deployment framings, differing only in contextual wrapper text.

## Verdict logic

```
if ρ_A ≥ 0.5 AND ρ_G ≥ 0.3:
 verdict = REAL # suppression genuinely closes the gap
elif ρ_A ≥ 0.5 AND ρ_G < 0.3:
 verdict = COSMETIC # probe AUC drops but behaviour unchanged
else:
 verdict = INCONCLUSIVE # insufficient probe-accuracy suppression
```

Where:
- ρ_A = (AUC_baseline − AUC_best) / (AUC_baseline − 0.5) — normalised probe-accuracy drop
- ρ_G = (G_baseline − G_best) / G_baseline — normalised behavioural-gap reduction

## Results summary

| Model | v5_2 | v5_3 | v5_4 |
|-------|------|------|------|
| Qwen2.5-1.5B | COSMETIC | INCONCLUSIVE | COSMETIC |
| **Qwen1.5-1.8B** | COSMETIC | INCONCLUSIVE | **REAL** |
| Gemma-2-2B | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE |
| TinyLlama-1.1B | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE |

## Citation

If you use this code, please cite:

```bibtex
@mastersthesis{jose2026probing,
 title = {Probing Whether a Language Model Knows It Is Being Tested:
 Linear Probing and Directional Ablation of Evaluation-Awareness
 Representations in Small Language Models},
 author = {Jose, Donal},
 school = {De Montfort University},
 year = {2026},
 type = {MSc Dissertation}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

Supervised by Dr. Sumaiya Thaseen, De Montfort University, Dubai.
