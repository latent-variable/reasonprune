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
from reasonprune.prune import (apply_mask, apply_runtime_mask, random_mask,
                               select_channels)
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


def eval_sets(limit: int | None, bench_limit: int = 0):
    know = load_items(DATA_DIR / "knowledge.eval.jsonl")
    reason = load_items(DATA_DIR / "reasoning.eval.jsonl")
    if limit:
        know, reason = know[:limit], reason[:limit]
    items = know + reason
    if bench_limit:
        for f in sorted((DATA_DIR / "bench").glob("*.jsonl")):
            items += load_items(f)[:bench_limit]
    return items


def full_eval(model, tokenizer, limit: int | None, max_tokens: int,
              bench_limit: int = 0) -> dict:
    res = run_eval(model, tokenizer, eval_sets(limit, bench_limit),
                   max_tokens=max_tokens)
    if NEUTRAL_TEXT_FILE.exists():
        res["ppl_neutral"] = perplexity(model, tokenizer,
                                        NEUTRAL_TEXT_FILE.read_text())
    return res


def cmd_baseline(args):
    model, tokenizer = load_model(args.model)
    res = full_eval(model, tokenizer, args.limit, args.max_tokens,
                    args.bench_limit)
    res["config"] = {"model": args.model, "strategy": "baseline", "frac": 0.0}
    path = out_dir(args.model) / f"baseline{args.suffix}.json"
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "records"}, indent=1))


def reason_calib_items():
    """Reasoning calibration: synthetic short-form + GSM8K-train CoT text.

    The CoT slice protects long-form working; without it, GSM8K chain-of-
    thought degrades under pruning even when short-answer reasoning holds
    (observed on qwen-0.8b, see .agents/memory/experiment-log.md).
    """
    items = load_items(DATA_DIR / "reasoning.calib.jsonl")
    cot = DATA_DIR / "bench" / "gsm8k_train_calib.jsonl"
    if cot.exists():
        items = items + load_items(cot)
    return items


def cmd_score(args):
    model, tokenizer = load_model(args.model)
    know = load_items(DATA_DIR / "knowledge.calib.jsonl")
    reason = reason_calib_items()
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
    results_path = out_dir(args.model) / f"sweep{args.suffix}.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            c = json.loads(line)["config"]
            done.add((c["strategy"], c["frac"]))
    model, tokenizer = load_model(args.model)
    for strategy in args.strategies.split(","):
        for frac in [float(f) for f in args.fracs.split(",")]:
            if (strategy, frac) in done:
                print(f"skip {strategy}@{frac} (done)")
                continue
            mask = build_mask(strategy, frac, scores)
            n = apply_runtime_mask(model, mask)
            print(f"=== {strategy} frac={frac}: pruned {n} channels "
                  f"({n/mask.size:.1%} of MLP hidden)", flush=True)
            res = full_eval(model, tokenizer, args.limit, args.max_tokens,
                            args.bench_limit)
            res["config"] = {"model": args.model, "strategy": strategy,
                             "frac": frac, "n_pruned": n}
            del res["records"]
            with results_path.open("a") as f:
                f.write(json.dumps(res) + "\n")
            print(f"knowledge={res['knowledge_acc']:.3f} "
                  f"reasoning={res['reasoning_acc']:.3f} "
                  f"ppl={res.get('ppl_neutral')}", flush=True)
            apply_runtime_mask(model, None)


def cmd_score_moe(args):
    from reasonprune.evalharness import format_prompt
    from reasonprune.moe import collect_expert_saliency
    model, tokenizer = load_model(args.model)
    know = load_items(DATA_DIR / "knowledge.calib.jsonl")
    reason = reason_calib_items()
    t0 = time.time()
    s_know = collect_expert_saliency(model, tokenizer, know, format_prompt)
    print(f"knowledge expert saliency in {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    s_reason = collect_expert_saliency(model, tokenizer, reason, format_prompt)
    print(f"reasoning expert saliency in {time.time()-t0:.0f}s", flush=True)
    save_scores(out_dir(args.model) / "expert_scores.npz",
                i_know=s_know, i_reason=s_reason)
    d = differential(s_know, s_reason)
    print(f"experts: {s_know.shape}; differential p50/p95/p99 = "
          f"{np.percentile(d, 50):.3f}/{np.percentile(d, 95):.3f}/"
          f"{np.percentile(d, 99):.3f}")


def cmd_sweep_moe(args):
    from reasonprune.moe import apply_expert_mask, instrument_moe
    scores = load_scores(out_dir(args.model) / "expert_scores.npz")
    results_path = out_dir(args.model) / f"sweep_moe{args.suffix}.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            c = json.loads(line)["config"]
            done.add((c["strategy"], c["frac"]))
    model, tokenizer = load_model(args.model)
    wrappers = instrument_moe(model)
    n_layers, n_experts = scores["i_know"].shape
    for strategy in args.strategies.split(","):
        for frac in [float(f) for f in args.fracs.split(",")]:
            if (strategy, frac) in done:
                print(f"skip {strategy}@{frac} (done)")
                continue
            mask = build_mask(strategy, frac, scores)
            n = apply_expert_mask(model, mask)
            print(f"=== {strategy} frac={frac}: masked {n}/{mask.size} experts",
                  flush=True)
            res = full_eval(model, tokenizer, args.limit, args.max_tokens,
                            args.bench_limit)
            res["config"] = {"model": args.model, "strategy": strategy,
                             "frac": frac, "n_pruned": n, "unit": "expert"}
            del res["records"]
            with results_path.open("a") as f:
                f.write(json.dumps(res) + "\n")
            print(f"knowledge={res['knowledge_acc']:.3f} "
                  f"reasoning={res['reasoning_acc']:.3f} "
                  f"ppl={res.get('ppl_neutral')}", flush=True)
    # Clear masks at the end so a reused process isn't left pruned.
    apply_expert_mask(model, np.zeros((n_layers, n_experts), dtype=bool))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["baseline", "score", "sweep",
                                   "score-moe", "sweep-moe"])
    p.add_argument("--model", default="qwen-0.8b")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--bench-limit", type=int, default=0,
                   help="items per data/bench/*.jsonl set (0 = skip bench)")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--strategies", default="diff,know,lowmag,random")
    p.add_argument("--fracs", default="0.1,0.2,0.3")
    p.add_argument("--suffix", default="", help="results filename suffix")
    args = p.parse_args()
    {"baseline": cmd_baseline, "score": cmd_score, "sweep": cmd_sweep,
     "score-moe": cmd_score_moe, "sweep-moe": cmd_sweep_moe}[args.cmd](args)


if __name__ == "__main__":
    main()
