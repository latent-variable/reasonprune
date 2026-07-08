"""MoE expert-level scoring and pruning (Qwen3.5/3.6 A3B family).

REAP saliency per expert (Cerebras REAP, arXiv:2510.13999):
    S(l, e) = E over tokens routing to e of [ router_score_e * ||expert_e(x)|| ]
computed per calibration set; the differential ratio then marks
knowledge-specialized experts for removal.

Pruning masks an expert by pinning its router logit to -inf so it is never
selected; the router renormalizes over survivors (norm_topk_prob=True in this
family), which is REAP's "keep router control" property for free.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .capture import decoder_layers


class InstrumentedMoE(nn.Module):
    """Wraps Qwen3NextSparseMoeBlock: captures REAP stats, applies expert masks."""

    def __init__(self, moe: nn.Module):
        super().__init__()
        self.gate = moe.gate
        self.switch_mlp = moe.switch_mlp
        self.shared_expert = moe.shared_expert
        self.shared_expert_gate = moe.shared_expert_gate
        self.norm_topk_prob = moe.norm_topk_prob
        self.num_experts = moe.num_experts
        self.top_k = moe.top_k
        # -inf logit additive mask; zeros = no expert masked.
        self.logit_mask = mx.zeros((self.num_experts,))
        self.capture = False
        self.saliency = np.zeros(self.num_experts, dtype=np.float64)
        self.hits = np.zeros(self.num_experts, dtype=np.int64)
        self.tokens = 0

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x) + self.logit_mask
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)

        y = self.switch_mlp(x, inds)

        if self.capture:
            norms = mx.linalg.norm(y.astype(mx.float32), axis=-1)  # [..., k]
            flat_inds = np.array(inds.reshape(-1, k))
            flat_sal = np.array((scores.astype(mx.float32) * norms).reshape(-1, k))
            np.add.at(self.saliency, flat_inds.ravel(), flat_sal.ravel())
            np.add.at(self.hits, flat_inds.ravel(), 1)
            self.tokens += flat_inds.shape[0]

        y = (y * scores[..., None]).sum(axis=-2)
        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        return y + shared_y

    def mean_saliency(self) -> np.ndarray:
        return self.saliency / max(self.tokens, 1)

    def reset_stats(self) -> None:
        self.saliency = np.zeros(self.num_experts, dtype=np.float64)
        self.hits = np.zeros(self.num_experts, dtype=np.int64)
        self.tokens = 0


def instrument_moe(model) -> list[InstrumentedMoE]:
    wrappers = []
    for layer in decoder_layers(model):
        if not hasattr(layer.mlp, "switch_mlp"):
            raise ValueError("dense layer found; expected MoE block")
        if not isinstance(layer.mlp, InstrumentedMoE):
            layer.mlp = InstrumentedMoE(layer.mlp)
        wrappers.append(layer.mlp)
    return wrappers


def collect_expert_saliency(model, tokenizer, items, format_prompt,
                            max_len: int = 1024) -> np.ndarray:
    """Returns [n_layers, num_experts] mean REAP saliency over `items`."""
    wrappers = instrument_moe(model)
    for w in wrappers:
        w.reset_stats()
        w.capture = True
    for it in items:
        text = format_prompt(tokenizer, it.prompt) + it.answer
        tokens = tokenizer.encode(text)[:max_len]
        out = model(mx.array(tokens)[None])
        mx.eval(out)
    for w in wrappers:
        w.capture = False
    return np.stack([w.mean_saliency() for w in wrappers])


def apply_expert_mask(model, mask: np.ndarray) -> int:
    """mask [n_layers, num_experts], True = disable expert. Returns #masked."""
    total = 0
    for layer, layer_mask in zip(decoder_layers(model), mask):
        mlp = layer.mlp
        assert isinstance(mlp, InstrumentedMoE), "instrument_moe() first"
        neg = np.where(layer_mask, -np.inf, 0.0).astype(np.float32)
        mlp.logit_mask = mx.array(neg)
        total += int(layer_mask.sum())
    return total
