"""Importance scoring: run calibration sets, produce per-channel scores.

Wanda-style importance of MLP hidden channel j at layer l:
    I(l, j) = rms_act(l, j) * ||W_down[:, j]||_2
computed separately on the knowledge (K) and reasoning (R) calibration sets.

Differential ratio (Pochinkov & Schoots 2024, transplanted to compression):
    D(l, j) = I_K(l, j) / (I_R(l, j) + eps)
High D = knowledge-specialized channel = prune candidate.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .capture import InstrumentedMLP, down_col_norms, instrument, reset_all
from .datagen import Item
from .evalharness import format_prompt


def forward_collect(model, tokenizer, items: list[Item],
                    wrappers: list[InstrumentedMLP],
                    include_answer: bool = True,
                    max_len: int = 1024) -> int:
    """Teacher-forced forward passes over prompt(+gold answer) to fill stats."""
    import sys

    n_tokens = 0
    for i, it in enumerate(items):
        text = format_prompt(tokenizer, it.prompt)
        if include_answer:
            text = text + it.answer
        tokens = tokenizer.encode(text)[:max_len]
        model(mx.array(tokens)[None])
        # Evaluate ALL accumulated stats: anything left lazy chains a growing
        # graph across items and leaks unboundedly (SIGKILLed the 27B runs).
        mx.eval([w.sum_abs for w in wrappers] + [w.sum_sq for w in wrappers])
        n_tokens += len(tokens)
        if (i + 1) % 25 == 0:
            mx.clear_cache()
            print(f"  calib {i+1}/{len(items)} ({n_tokens} tok, "
                  f"peak {mx.get_peak_memory()/1e9:.1f}GB)", flush=True)
    return n_tokens


def collect_importance(model, tokenizer, items: list[Item]) -> np.ndarray:
    """Returns [n_layers, d_inter] Wanda importance on `items`."""
    wrappers = instrument(model)
    reset_all(wrappers)
    forward_collect(model, tokenizer, items, wrappers)
    rows = []
    for w in wrappers:
        act_rms = w.stats()["rms"]                     # [d_inter]
        w_norm = down_col_norms(w)                     # [d_inter]
        rows.append(np.array(act_rms * w_norm, copy=False))
    return np.stack(rows)


def differential(i_know: np.ndarray, i_reason: np.ndarray,
                 eps_quantile: float = 0.10) -> np.ndarray:
    """Ratio score with a per-layer floor so dead channels don't explode."""
    eps = np.quantile(i_reason, eps_quantile, axis=1, keepdims=True) + 1e-8
    return i_know / (i_reason + eps)


def save_scores(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_scores(path: Path) -> dict:
    return dict(np.load(path))
