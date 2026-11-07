"""Model loading, chat templating, and residual-stream activation caching.

Uses Hugging Face Transformers with plain PyTorch forward hooks rather than a
third-party interpretability wrapper. The hook attaches to each decoder block's
output, which IS the residual stream at that layer.
"""

from __future__ import annotations

from typing import List, Optional, Callable, Any
import numpy as np
import torch


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_model(cfg) -> tuple:
    """Load tokenizer and model. Imported lazily so the rest of the package
    (and its tests) do not require transformers to be installed.

    Supports optional 4-bit quantisation via bitsandbytes for larger models
    (~7B) on GPUs that could not otherwise fit them in float16.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(cfg.device)
    dtype = DTYPE_MAP[cfg.dtype]
    # float16 on CPU is slow and numerically awkward; fall back cleanly.
    if device == "cpu" and dtype is torch.float16:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, trust_remote_code=cfg.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        torch_dtype=dtype,
        trust_remote_code=cfg.trust_remote_code,
    )

    load_in_4bit = bool(getattr(cfg, "load_in_4bit", False))
    if load_in_4bit:
        if device != "cuda":
            raise RuntimeError(
                "load_in_4bit=True requires a CUDA device; got device=%r." % device
            )
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise ImportError(
                "4-bit quantisation requires bitsandbytes. Install with:\n"
                "  pip install bitsandbytes accelerate"
            ) from e
        # NF4 with double-quant is the standard QLoRA recipe; compute in fp16.
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        # By default let accelerate plan the device_map. `force_gpu_only=True`
        # skips accelerate's headroom-based safety check and pins EVERY layer
        # to GPU 0 — use it when accelerate refuses an all-GPU plan you know
        # will just barely fit. If the model actually doesn't fit, it will
        # OOM at allocation time rather than at planning time.
        force_gpu_only = bool(getattr(cfg, "force_gpu_only", False))
        if force_gpu_only:
            load_kwargs["device_map"] = {"": 0}
        else:
            load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **load_kwargs)
    if not load_in_4bit:
        model.to(device)
    model.eval()
    return tokenizer, model


def apply_chat_template(tokenizer, user_text: str) -> str:
    """Wrap raw text in the model's own chat template.

    Falls back to the raw string if the tokenizer has no template, so that
    stub/base models still work.
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_text


def get_decoder_layers(model) -> List[torch.nn.Module]:
    """Locate the list of decoder blocks across common architectures."""
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj: Any = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and isinstance(obj, (list, torch.nn.ModuleList)):
            return list(obj)
    raise AttributeError(
        "Could not locate decoder layers on this model. "
        "Add its attribute path to get_decoder_layers()."
    )


def _hidden_from_output(out) -> torch.Tensor:
    """Decoder blocks return either a tensor or a tuple whose first element
    is the hidden state."""
    if isinstance(out, tuple):
        return out[0]
    return out


class ActivationCache:
    """Captures residual-stream activations at one token position, all layers."""

    def __init__(self, model, token_position: int = -1):
        self.layers = get_decoder_layers(model)
        self.token_position = token_position
        self._captured: List[Optional[torch.Tensor]] = [None] * len(self.layers)
        self._handles: List[Any] = []

    def _make_hook(self, idx: int) -> Callable:
        def hook(_module, _inputs, output):
            hidden = _hidden_from_output(output)
            # hidden: (batch, seq, d_model) -> take one token position
            self._captured[idx] = hidden[:, self.token_position, :].detach().float().cpu()
        return hook

    def __enter__(self) -> "ActivationCache":
        self._handles = [
            layer.register_forward_hook(self._make_hook(i))
            for i, layer in enumerate(self.layers)
        ]
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def stacked(self) -> np.ndarray:
        """Return (n_layers, batch, d_model)."""
        if any(c is None for c in self._captured):
            raise RuntimeError("Not all layers captured; run a forward pass first.")
        return np.stack([c.numpy() for c in self._captured], axis=0)


@torch.no_grad()
def collect_activations(
    tokenizer, model, texts: List[str], cfg, batch_size: Optional[int] = None
) -> np.ndarray:
    """Run the texts through the model and return (n_layers, n_texts, d_model).

    Padding is LEFT-side so that position -1 is the true final content token
    for every sequence in a batch, not a pad token.
    """
    device = next(model.parameters()).device
    original_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"

    if batch_size is None:
        batch_size = int(getattr(cfg, "batch_size", 8))

    per_batch: List[np.ndarray] = []
    try:
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            templated = [apply_chat_template(tokenizer, t) for t in chunk]
            enc = tokenizer(
                templated,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_prompt_tokens,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with ActivationCache(model, cfg.token_position) as cache:
                model(**enc)
                per_batch.append(cache.stacked())
    finally:
        tokenizer.padding_side = original_side

    return np.concatenate(per_batch, axis=1)
