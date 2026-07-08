#!/usr/bin/env python3
"""Physically slice a model using saved differential scores, then verify.

Usage:
  slice_model.py --model qwen-0.8b --frac 0.3 [--strategy diff] [--out DIR]

Verification: loads the sliced checkpoint fresh, generates from a smoke
prompt, and reports peak memory + decode tok/s vs the original.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler

from reasonprune.config import MODELS, REPO_ROOT, RESULTS_DIR
from reasonprune.score import differential, load_scores
from reasonprune.prune import select_channels
from reasonprune.slicer import slice_checkpoint

SMOKE = "Give three uses for a brick, numbered."


def resolve_snapshot(model_id: str) -> Path:
    p = Path(model_id)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_id))


def bench(model_path: str, label: str) -> dict:
    mx.clear_cache()
    mx.reset_peak_memory()
    model, tokenizer = load(model_path)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": SMOKE}], add_generation_prompt=True,
        tokenize=False)
    t0 = time.time()
    out = generate(model, tokenizer, prompt=prompt, max_tokens=120,
                   sampler=make_sampler(temp=0.0), verbose=False)
    dt = time.time() - t0
    n_out = len(tokenizer.encode(out))
    peak_gb = mx.get_peak_memory() / 1e9
    print(f"[{label}] peak={peak_gb:.2f}GB decode~{n_out/dt:.1f} tok/s")
    print(f"[{label}] sample: {out[:200]!r}")
    del model
    mx.clear_cache()
    return {"peak_gb": round(peak_gb, 2), "tok_s": round(n_out / dt, 1),
            "sample": out[:300]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen-0.8b")
    p.add_argument("--strategy", default="diff")
    p.add_argument("--frac", type=float, required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--skip-bench", action="store_true")
    args = p.parse_args()

    scores = load_scores(RESULTS_DIR / args.model / "scores.npz")
    i_know, i_reason = scores["i_know"], scores["i_reason"]
    d = differential(i_know, i_reason)
    if args.strategy == "diff":
        mask = select_channels(d, args.frac, protect=i_reason)
    elif args.strategy == "lowmag":
        mask = select_channels(-(i_know + i_reason), args.frac)
    else:
        raise SystemExit(f"unsupported strategy {args.strategy}")

    src = resolve_snapshot(MODELS[args.model])
    out = Path(args.out) if args.out else (
        REPO_ROOT / "models" / f"{args.model}-rp{int(args.frac*100)}")
    meta = slice_checkpoint(src, out, mask, scores=d)
    print(json.dumps(meta, indent=1))

    if not args.skip_bench:
        results = {"original": bench(str(src), "original"),
                   "sliced": bench(str(out), "sliced"), "slice_meta": meta}
        (RESULTS_DIR / args.model /
         f"slice_{args.strategy}_{args.frac}.json").write_text(
            json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
