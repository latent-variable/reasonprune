"""Activation capture for MLX models: per-channel MLP statistics.

Replaces each decoder layer's `mlp` with an instrumented wrapper that
accumulates statistics of the SwiGLU hidden activation h = silu(gate(x))*up(x)
— the input to down_proj, i.e. the "value memory" read strength per channel
(Geva et al.: FFN as key-value memory; channel j active = memory j recalled).

Works on the dense Qwen3.5 family (Qwen3NextMLP: gate_proj/up_proj/down_proj).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def _swiglu(g: mx.array, u: mx.array) -> mx.array:
    return nn.silu(g) * u


def _out_features(linear: nn.Module) -> int:
    """Output dim of a Linear or QuantizedLinear (packed weight width differs)."""
    if hasattr(linear, "group_size"):  # QuantizedLinear: scales are unpacked
        return linear.scales.shape[0]
    return linear.weight.shape[0]


class InstrumentedMLP(nn.Module):
    """Drop-in replacement for Qwen3NextMLP that accumulates channel stats."""

    def __init__(self, mlp: nn.Module, skip_first: int = 4):
        super().__init__()
        self.gate_proj = mlp.gate_proj
        self.up_proj = mlp.up_proj
        self.down_proj = mlp.down_proj
        self.skip_first = skip_first
        d_inter = _out_features(self.gate_proj)
        self.sum_abs = mx.zeros((d_inter,), dtype=mx.float32)
        self.sum_sq = mx.zeros((d_inter,), dtype=mx.float32)
        self.count = 0
        # Runtime channel mask (1 = keep). Equivalent to zeroing gate/up rows
        # and down columns, but works unchanged on quantized weights.
        self.channel_mask = None
        # Stats accumulate ONLY while capture=True (scoring). Left on, every
        # decode step chains lazy adds nobody evaluates — unbounded growth.
        self.capture = False

    def __call__(self, x: mx.array) -> mx.array:
        h = _swiglu(self.gate_proj(x), self.up_proj(x))
        if self.channel_mask is not None:
            h = h * self.channel_mask
        if not self.capture:
            return self.down_proj(h)
        # Stats over token positions, skipping attention-sink prefix.
        flat = h.reshape(-1, h.shape[-1])
        if flat.shape[0] > self.skip_first:
            sample = flat[self.skip_first:].astype(mx.float32)
            self.sum_abs = self.sum_abs + mx.abs(sample).sum(axis=0)
            self.sum_sq = self.sum_sq + mx.square(sample).sum(axis=0)
            self.count += sample.shape[0]
        return self.down_proj(h)

    def stats(self) -> dict:
        n = max(self.count, 1)
        return {
            "mean_abs": self.sum_abs / n,
            "rms": mx.sqrt(self.sum_sq / n),
            "count": self.count,
        }

    def reset(self) -> None:
        self.sum_abs = mx.zeros_like(self.sum_abs)
        self.sum_sq = mx.zeros_like(self.sum_sq)
        self.count = 0


def down_col_norms(mlp) -> mx.array:
    """L2 norm of down_proj columns (per hidden channel), quant-aware."""
    w = mlp.down_proj
    if hasattr(w, "group_size"):
        dense = mx.dequantize(w.weight, w.scales, w.biases,
                              group_size=w.group_size, bits=w.bits)
    else:
        dense = w.weight
    return mx.linalg.norm(dense.astype(mx.float32), axis=0)


def decoder_layers(model) -> list:
    """The decoder layer list, tolerant of the multimodal wrapper."""
    m = model
    for attr in ("language_model", "model"):
        while hasattr(m, attr):
            m = getattr(m, attr)
    return list(m.layers)


def instrument(model, skip_first: int = 4) -> list[InstrumentedMLP]:
    """Wrap every layer's dense MLP; returns the wrappers (layer order)."""
    wrappers = []
    for layer in decoder_layers(model):
        if not hasattr(layer.mlp, "gate_proj"):
            raise ValueError("layer.mlp has no gate_proj — MoE layer? "
                             "use expert-level scoring instead")
        if not isinstance(layer.mlp, InstrumentedMLP):
            layer.mlp = InstrumentedMLP(layer.mlp, skip_first=skip_first)
        wrappers.append(layer.mlp)
    return wrappers


def reset_all(wrappers: list[InstrumentedMLP]) -> None:
    for w in wrappers:
        w.reset()
