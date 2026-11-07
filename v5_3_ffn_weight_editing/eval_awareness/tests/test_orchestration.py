"""Prove run_pipeline.run() wires together correctly, using a stub model.

This does NOT test model quality - it tests that the orchestrator's data flow,
shapes, and stage handoffs are correct end to end.
"""
import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import torch
import torch.nn as nn

from eval_awareness.config import Config
from eval_awareness import run_pipeline
from eval_awareness.tests.test_pipeline import StubModel

# --- stub tokenizer -------------------------------------------------
class StubTok:
    pad_token = "<pad>"; eos_token = "</s>"; pad_token_id = 0
    padding_side = "right"; chat_template = None
    def __call__(self, texts, **kw):
        # Text-dependent ids so eval/deploy framings differ in activations,
        # mimicking a real tokenizer well enough to exercise the pipeline.
        L = 6
        rows = []
        for t in texts:
            is_eval = ("EVALUATION" in t.upper() or "BENCHMARK" in t.upper()
                       or "evaluated" in t or "TEST CASE" in t.upper())
            base = 5 if is_eval else 1
            rows.append([base + (hash(t) % 3)] * L)
        ids = torch.tensor(rows, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    def decode(self, ids, **kw): return "I can't help with that."

# --- stub model that accepts **kwargs like HF and can generate -------
D = 8
class HFStub(StubModel):
    def __init__(self):
        super().__init__(n_layers=3, d_model=D)
        self.embed = nn.Embedding(10, D)
    def forward(self, input_ids=None, attention_mask=None, **kw):
        x = self.embed(input_ids)
        for blk in self.model.layers:
            x = blk(x)[0]
        return types.SimpleNamespace(logits=x)
    def generate(self, input_ids=None, attention_mask=None, **kw):
        n, L = input_ids.shape
        new = kw.get("max_new_tokens", 8)
        return torch.ones(n, L + new, dtype=torch.long)

stub_tok, stub_model = StubTok(), HFStub()
run_pipeline.__dict__.setdefault('_orig', None)

# monkeypatch load_model at the point of use
import eval_awareness.activations as act_mod
act_mod.load_model = lambda cfg: (stub_tok, stub_model)

cfg = Config()
cfg.device = "cpu"; cfg.max_new_tokens = 8
cfg.steering_alphas = [0.0, 0.5, 1.0]
cfg.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_orch_out")

out = run_pipeline.run(cfg)

print("\n--- ORCHESTRATION CHECKS ---")
assert out["n_pairs"] == 160, out["n_pairs"]
assert "probes" in out and "baseline_behaviour" in out and "suppression" in out
sweep = out["suppression"]["sweep"]
assert len(sweep) == 3, len(sweep)
assert sweep[0]["alpha"] == 0.0
aucs = [p["auc"] for p in sweep]
assert len(set(round(a,6) for a in aucs)) > 1, f"AUC did not vary across sweep: {aucs}"
assert all("verdict" in p and "oversteered" in p for p in sweep)
assert out["suppression"]["overall_verdict"] in {"real","cosmetic","inconclusive"}
assert os.path.exists(os.path.join(cfg.output_dir, "results.json"))
n_layers = len(out["probes"]["layers"])
assert n_layers == 3, n_layers
print(f"n_pairs=160 OK | layers={n_layers} OK | sweep points={len(sweep)} OK")
print(f"verdict={out['suppression']['overall_verdict']} | results.json written OK")
print("ORCHESTRATION VERIFIED")
