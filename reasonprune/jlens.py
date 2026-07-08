"""Sketched Jacobian-transport alignment: the J-space protection signal.

The workspace paper defines the Jacobian lens transport J_l = E[dh_final/dh_l]
(expectation over prompts, source positions t, and target positions t' >= t).
A channel's causal reach to the output is ||J_l w_j|| where w_j is its
down_proj write vector. Computing J_l exactly is a d x d matrix per layer;
instead we sketch:

    ||J_l w||^2 = E_{u ~ N(0,I)} [ (J_l^T u . w)^2 ]

so K random probes u_k and one VJP each give an unbiased estimate for ALL
channels at once. One backward pass per probe yields J_l^T u_k for EVERY
layer simultaneously, by differentiating w.r.t. zero-perturbations injected
at each layer input (eps trick).

Estimator follows jlens.fitting: cotangent summed over target positions,
gradient averaged over valid source positions (skip attention-sink prefix,
drop the last position).

Cost: n_prompts x n_probes forward+backward passes of the model.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .capture import decoder_layers, down_col_norms
from .evalharness import format_prompt

SKIP_FIRST = 16  # attention-sink prefix excluded from the average (paper: 16)


def _text_model(model):
    m = model
    for attr in ("language_model", "model"):
        while hasattr(m, attr):
            m = getattr(m, attr)
    return m


def _masks_for(tm, hidden):
    """Attention + SSM masks as the qwen3_5 forward builds them (no cache)."""
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    fa = create_attention_mask(hidden, None)
    ssm = create_ssm_mask(hidden, None)
    return fa, ssm


def transport_vectors(model, tokenizer, prompts: list[str],
                      n_probes: int = 64, max_len: int = 384,
                      seed: int = 0) -> np.ndarray:
    """Sketch vectors v_k = J_l^T u_k per layer, [n_layers, K, d_model],
    averaged over prompts and valid source positions. Probes are FIXED across
    prompts (the estimator needs E_prompt[J]^T u, not per-prompt draws)."""
    import time as _time

    tm = _text_model(model)
    layers = decoder_layers(model)
    L = len(layers)
    d = tm.embed_tokens.weight.shape[1]
    rng = np.random.default_rng(seed)
    probes = mx.array(rng.standard_normal((n_probes, d)).astype(np.float32))
    acc = mx.zeros((L, n_probes, d), dtype=mx.float32)
    n_prompts = 0
    t0 = _time.time()

    # Train mode routes GatedDeltaNet through its differentiable ops path
    # (the fast Metal kernel has no VJP): use_kernel=not self.training.
    model.train(True)

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)[:max_len]
        S = len(tokens)
        if S <= SKIP_FIRST + 2:
            continue
        hidden0 = tm.embed_tokens(mx.array(tokens)[None])
        fa_mask, ssm_mask = _masks_for(tm, hidden0)
        valid = mx.zeros((1, S, 1))
        valid[:, SKIP_FIRST:S - 1, :] = 1.0
        n_valid = S - 1 - SKIP_FIRST

        def fwd(eps_list):
            x = hidden0
            for layer, eps in zip(layers, eps_list):
                mask = ssm_mask if layer.is_linear else fa_mask
                x = layer(x + eps, mask=mask, cache=None)
            return tm.norm(x)

        zeros = [mx.zeros_like(hidden0) for _ in range(L)]
        for k in range(n_probes):
            u = probes[k]

            def scalar_fn(*eps_list):
                out = fwd(list(eps_list))          # [1, S, d]
                # cotangent = u at every target position (sum over t' >= t
                # happens implicitly through causality: dh_out[t']/deps[t]=0
                # for t' < t, so grad at t = sum over t' >= t).
                return (out * u).sum()

            grads = mx.grad(scalar_fn, argnums=tuple(range(L)))(*zeros)
            g = mx.stack([(gr * valid).sum(axis=1)[0] / n_valid
                          for gr in grads])        # [L, d]
            acc[:, k, :] = acc[:, k, :] + g.astype(mx.float32)
            mx.eval(acc)
        mx.clear_cache()
        n_prompts += 1
        print(f"  jlens prompt {n_prompts}/{len(prompts)} "
              f"({_time.time()-t0:.0f}s elapsed, "
              f"peak {mx.get_peak_memory()/1e9:.1f}GB)", flush=True)
    model.train(False)
    return np.array(acc / max(n_prompts, 1))


def jspace_alignment(model, V: np.ndarray) -> np.ndarray:
    """[n_layers, d_inter] estimated ||J_l w_j|| / ||w_j|| per channel.

    V: sketch vectors from transport_vectors, [n_layers, K, d_model].
    """
    out = []
    for l, layer in enumerate(decoder_layers(model)):
        mlp = layer.mlp
        w = mlp.down_proj
        if hasattr(w, "group_size"):
            dense = mx.dequantize(w.weight, w.scales, w.biases,
                                  group_size=w.group_size, bits=w.bits)
        else:
            dense = w.weight
        W = np.array(dense.astype(mx.float32))       # [d_model, d_inter]
        proj = V[l] @ W                              # [K, d_inter]
        est = np.sqrt((proj ** 2).mean(axis=0))      # ~ ||J_l w_j||
        norms = np.array(down_col_norms(mlp))
        out.append(est / (norms + 1e-8))
    return np.stack(out)


def collect_jspace(model, tokenizer, items, n_prompts: int = 48,
                   n_probes: int = 64) -> np.ndarray:
    """J-space alignment scored on reasoning-context prompts."""
    prompts = [format_prompt(tokenizer, it.prompt) + it.answer
               for it in items[:n_prompts]]
    V = transport_vectors(model, tokenizer, prompts, n_probes=n_probes)
    return jspace_alignment(model, V)
