"""Structured pruning: select and zero MLP hidden channels.

Masking (zeroing down_proj columns + gate/up rows) is mathematically identical
to removing the channel; physical slicing for memory/speed wins comes later.
Sweeps reload the model between configurations instead of undoing masks.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from .capture import decoder_layers


def select_channels(
    scores: np.ndarray,
    frac: float,
    protect: np.ndarray | None = None,
    protect_quantile: float = 0.70,
    per_layer: bool = True,
) -> np.ndarray:
    """Boolean prune mask [n_layers, d_inter]: True = prune.

    scores: higher = more prunable (e.g. differential ratio D).
    protect: e.g. I_reason — channels above its per-layer protect_quantile are
             never pruned regardless of score (the overlap guard from DESIGN).
    """
    n_layers, d = scores.shape
    mask = np.zeros_like(scores, dtype=bool)
    eligible = np.ones_like(scores, dtype=bool)
    if protect is not None:
        thresh = np.quantile(protect, protect_quantile, axis=1, keepdims=True)
        eligible &= protect < thresh
    if per_layer:
        k = int(d * frac)
        for l in range(n_layers):
            idx = np.where(eligible[l])[0]
            if len(idx) == 0:
                continue
            order = idx[np.argsort(-scores[l, idx])]
            mask[l, order[:k]] = True
    else:
        flat_scores = np.where(eligible, scores, -np.inf).ravel()
        k = int(scores.size * frac)
        top = np.argsort(-flat_scores)[:k]
        mask.ravel()[top] = True
    return mask


def random_mask(shape: tuple, frac: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_layers, d = shape
    mask = np.zeros(shape, dtype=bool)
    k = int(d * frac)
    for l in range(n_layers):
        mask[l, rng.choice(d, size=k, replace=False)] = True
    return mask


def apply_mask(model, mask: np.ndarray) -> int:
    """Zero the masked channels in-place. Returns #channels pruned."""
    layers = decoder_layers(model)
    assert len(layers) == mask.shape[0], (len(layers), mask.shape)
    total = 0
    for layer, layer_mask in zip(layers, mask):
        if not layer_mask.any():
            continue
        keep = mx.array(~layer_mask)          # [d_inter] bool, True = keep
        mlp = layer.mlp
        col = keep[None, :].astype(mlp.down_proj.weight.dtype)   # [1, d_inter]
        row = keep[:, None].astype(mlp.gate_proj.weight.dtype)   # [d_inter, 1]
        mlp.down_proj.weight = mlp.down_proj.weight * col
        mlp.gate_proj.weight = mlp.gate_proj.weight * row
        mlp.up_proj.weight = mlp.up_proj.weight * row
        total += int(layer_mask.sum())
    mx.eval(model.parameters())
    return total
