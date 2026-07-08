#!/usr/bin/env python3
"""Fetch real benchmark slices into data/bench/ as Item jsonl.

- reason.gsm8k: GSM8K test problems (grade-school math, CoT then '#### N').
  Facts all in-problem: a reasoning benchmark under our definition.
- know.trivia: TriviaQA rc.nocontext validation (closed-book open recall,
  the knowledge probe the J-space paper's task split says to use, NOT MMLU).

Sources: openai/grade-school-math raw jsonl; HF datasets-server rows API.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reasonprune.datagen import Item

BENCH_DIR = Path(__file__).resolve().parent.parent / "data" / "bench"

GSM8K_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/"
             "master/grade_school_math/data/test.jsonl")
GSM8K_TRAIN_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/"
                   "master/grade_school_math/data/train.jsonl")
TRIVIA_API = ("https://datasets-server.huggingface.co/rows"
              "?dataset=mandarjoshi%2Ftrivia_qa&config=rc.nocontext"
              "&split=validation")

GSM8K_SUFFIX = ("\nWork through this step by step, then give the final "
                "numeric answer on a new line formatted as: #### <number>")


def fetch_gsm8k(n: int = 200) -> list[Item]:
    lines = requests.get(GSM8K_URL, timeout=60).text.splitlines()
    items = []
    for i, line in enumerate(lines[:n]):
        d = json.loads(line)
        gold = d["answer"].split("####")[-1].strip().replace(",", "")
        items.append(Item(
            id=f"reason.gsm8k.{i:04d}", kind="reason.gsm8k",
            prompt=d["question"] + GSM8K_SUFFIX,
            answer=gold, aliases=[],
            meta={"check": "final_number", "max_tokens": 512},
        ))
    return items


def fetch_trivia(n: int = 200) -> list[Item]:
    items = []
    offset = 0
    while len(items) < n:
        r = requests.get(f"{TRIVIA_API}&offset={offset}&length=100", timeout=60)
        r.raise_for_status()
        rows = r.json()["rows"]
        if not rows:
            break
        for row in rows:
            d = row["row"]
            ans = d["answer"]
            aliases = list({*ans.get("aliases", []), *ans.get("normalized_aliases", [])})
            q = d["question"].strip()
            items.append(Item(
                id=f"know.trivia.{len(items):04d}", kind="know.trivia",
                prompt=q + " Answer concisely.",
                answer=ans["value"], aliases=aliases[:24],
                meta={"max_tokens": 48},
            ))
            if len(items) >= n:
                break
        offset += 100
    return items


def fetch_gsm8k_train_calib(n: int = 150) -> list[Item]:
    """TRAIN split with full worked solutions as the target text.

    Calibration-only: teacher-forcing over the solution exposes long-form CoT
    activations so the reasoning-protection guard covers chain-of-thought
    machinery, not just short answers. Never used for evaluation.
    """
    lines = requests.get(GSM8K_TRAIN_URL, timeout=60).text.splitlines()
    items = []
    for i, line in enumerate(lines[:n]):
        d = json.loads(line)
        solution = re.sub(r"<<[^>]*>>", "", d["answer"])
        items.append(Item(
            id=f"calib.gsm8k_cot.{i:04d}", kind="reason.gsm8k_cot",
            prompt=d["question"] + "\nWork through this step by step.",
            answer=solution, aliases=[], meta={"calib_only": True},
        ))
    return items


def main():
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    for name, items in (("gsm8k", fetch_gsm8k()), ("trivia", fetch_trivia()),
                        ("gsm8k_train_calib", fetch_gsm8k_train_calib())):
        path = BENCH_DIR / f"{name}.jsonl"
        path.write_text("\n".join(it.to_json() for it in items) + "\n")
        print(f"{path}: {len(items)} items")


if __name__ == "__main__":
    main()
