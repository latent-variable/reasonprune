#!/usr/bin/env python3
"""End-to-end experiment driver.

Usage:
  experiment.py baseline --model qwen-0.8b [--limit N]
  experiment.py score    --model qwen-0.8b
  experiment.py sweep    --model qwen-0.8b --strategies diff,know,random --fracs 0.1,0.2,0.3,0.4
Results land in results/<model>/.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import numpy as np
from mlx_lm import load

from reasonprune.config import DATA_DIR, MODELS, RESULTS_DIR
from reasonprune.evalharness import load_items, run_eval, perplexity
from reasonprune.prune import apply_mask, random_mask, select_channels
from reasonprune.score import (collect_importance, differential, load_scores,
                               save_scores)

NEUTRAL_TEXT_FILE = DATA_DIR / "neutral.txt"


def out_dir(model_key: str) -> Path:
    d = RESULTS_DIR / model_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_model(model_key: str):
    t0 = time.time()
    model, tokenizer = load(MODELS[model_key])
    print(f"loaded {model_key} in {time.time()-t0:.1f}s", flush=True)
    return model, tokenizer


def eval_sets(limit: int | None):
    know = load_items(DATA_DIR / "knowledge.eval.jsonl")
    reason = load_items(DATA_DIR / "reasoning.eval.jsonl")
    if limit:
        know, reason = know[:limit], reason[:limit]
    return know + reason


def full_eval(model, tokenizer, limit: int | None, max_tokens: int) -> dict:
    res = run_eval(model, tokenizer, eval_sets(limit), max_tokens=max_tokens)
    if NEUTRAL_TEXT_FILE.exists():
        res["ppl_neutral"] = perplexity(model, tokenizer,
                                        NEUTRAL_TEXT_FILE.read_text())
    return res


def cmd_baseline(args):
    model, tokenizer = load_model(args.model)
    res = full_eval(model, tokenizer, args.limit, args.max_tokens)
    res["config"] = {"model": args.model, "strategy": "baseline", "frac": 0.0}
    path = out_dir(args.model) / "baseline.json"
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "records"}, indent=1))


def cmd_score(args):
    model, tokenizer = load_model(args.model)
    know = load_items(DATA_DIR / "knowledge.calib.jsonl")
    reason = load_items(DATA_DIR / "reasoning.calib.jsonl")
    t0 = time.time()
    i_know = collect_importance(model, tokenizer, know)
    print(f"knowledge importance done in {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    i_reason = collect_importance(model, tokenizer, reason)
    print(f"reasoning importance done in {time.time()-t0:.0f}s", flush=True)
    save_scores(out_dir(args.model) / "scores.npz",
                i_know=i_know, i_reason=i_reason)
    d = differential(i_know, i_reason)
    print("differential score summary per layer (p50/p95/p99):")
    for l in range(d.shape[0]):
        print(f"  L{l:02d} {np.percentile(d[l], 50):.3f} "
              f"{np.percentile(d[l], 95):.3f} {np.percentile(d[l], 99):.3f}")


def build_mask(strategy: str, frac: float, scores: dict) -> np.ndarray:
    i_know, i_reason = scores["i_know"], scores["i_reason"]
    if strategy == "diff":
        return select_channels(differential(i_know, i_reason), frac,
                               protect=i_reason)
    if strategy == "diff_noguard":
        return select_channels(differential(i_know, i_reason), frac)
    if strategy == "know":
        # Prune what knowledge needs MOST (sanity: should hurt knowledge).
        return select_channels(i_know, frac)
    if strategy == "lowmag":
        # Classic: prune lowest overall importance (Wanda-style baseline).
        combined = i_know + scores["i_reason"]
        return select_channels(-combined, frac)
    if strategy == "random":
        return random_mask(i_know.shape, frac)
    raise ValueError(f"unknown strategy {strategy}")


def cmd_sweep(args):
    scores = load_scores(out_dir(args.model) / "scores.npz")
    results_path = out_dir(args.model) / "sweep.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            c = json.loads(line)["config"]
            done.add((c["strategy"], c["frac"]))
    for strategy in args.strategies.split(","):
        for frac in [float(f) for f in args.fracs.split(",")]:
            if (strategy, frac) in done:
                print(f"skip {strategy}@{frac} (done)")
                continue
            model, tokenizer = load_model(args.model)
            mask = build_mask(strategy, frac, scores)
            n = apply_mask(model, mask)
            print(f"=== {strategy} frac={frac}: pruned {n} channels "
                  f"({n/mask.size:.1%} of MLP hidden)", flush=True)
            res = full_eval(model, tokenizer, args.limit, args.max_tokens)
            res["config"] = {"model": args.model, "strategy": strategy,
                             "frac": frac, "n_pruned": n}
            del res["records"]
            with results_path.open("a") as f:
                f.write(json.dumps(res) + "\n")
            print(json.dumps(res["kinds"], indent=1), flush=True)
            print(f"knowledge={res['knowledge_acc']:.3f} "
                  f"reasoning={res['reasoning_acc']:.3f} "
                  f"ppl={res.get('ppl_neutral')}", flush=True)
            del model, tokenizer
            gc.collect()
            mx.clear_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["baseline", "score", "sweep"])
    p.add_argument("--model", default="qwen-0.8b")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--strategies", default="diff,know,lowmag,random")
    p.add_argument("--fracs", default="0.1,0.2,0.3")
    args = p.parse_args()
    {"baseline": cmd_baseline, "score": cmd_score, "sweep": cmd_sweep}[args.cmd](args)


if __name__ == "__main__":
    main()
