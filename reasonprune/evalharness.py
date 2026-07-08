"""Evaluation harness: run eval jsonl sets against an in-memory MLX model.

Scores exact-match (normalized substring) per item kind, so results split
cleanly along the knowledge/reasoning axis. Also computes perplexity on a
neutral text sample as a coherence sanity check.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler

from .datagen import Item, matches


def load_items(path: Path) -> list[Item]:
    items = []
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            items.append(Item(**d))
    return items


def format_prompt(tokenizer, question: str) -> str:
    """Chat-format a question, disabling thinking mode when supported."""
    messages = [{"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False)
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)


def strip_thinking(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


_NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def _final_number(text: str) -> str | None:
    if "####" in text:
        text = text.split("####")[-1]
    nums = _NUM_RE.findall(text)
    if not nums:
        return None
    return nums[-1].replace(",", "").replace("$", "").rstrip(".")


def check_item(pred: str, item: Item) -> bool:
    pred = strip_thinking(pred)
    if item.meta.get("check") == "final_number":
        got = _final_number(pred)
        try:
            return got is not None and float(got) == float(item.answer)
        except ValueError:
            return False
    if item.meta.get("check") == "json_tool":
        # Tool-call items: parse the first JSON object and compare structurally.
        try:
            start = pred.index("{")
            obj = json.loads(pred[start:pred.rindex("}") + 1])
            args = obj.get("arguments") or obj.get("parameters") or {}
            return (obj.get("name") == "get_weather"
                    and args.get("city") == item.meta["city"]
                    and args.get("unit") == item.meta["unit"])
        except (ValueError, KeyError, AttributeError):
            return False
    return matches(pred, item)


def run_eval(model, tokenizer, items: list[Item], max_tokens: int = 64,
             verbose: bool = False) -> dict:
    sampler = make_sampler(temp=0.0)
    per_kind: dict[str, list[int]] = {}
    records = []
    t0 = time.time()
    for it in items:
        prompt = format_prompt(tokenizer, it.prompt)
        out = generate(model, tokenizer, prompt=prompt,
                       max_tokens=int(it.meta.get("max_tokens", max_tokens)),
                       sampler=sampler, verbose=False)
        ok = check_item(out, it)
        per_kind.setdefault(it.kind, []).append(int(ok))
        records.append({"id": it.id, "pred": strip_thinking(out), "ok": ok})
        if verbose:
            print(f"[{'+' if ok else ' '}] {it.id}: {strip_thinking(out)[:60]!r}"
                  f" (want {it.answer!r})")
    summary = {k: {"acc": sum(v) / len(v), "n": len(v)}
               for k, v in sorted(per_kind.items())}
    know = [r for k, v in per_kind.items() if k.startswith("know.") for r in v]
    reason = [r for k, v in per_kind.items() if k.startswith("reason.") for r in v]
    return {
        "kinds": summary,
        "knowledge_acc": sum(know) / len(know) if know else None,
        "reasoning_acc": sum(reason) / len(reason) if reason else None,
        "n_items": len(items),
        "seconds": round(time.time() - t0, 1),
        "records": records,
    }


def perplexity(model, tokenizer, text: str, max_tokens: int = 2048) -> float:
    tokens = tokenizer.encode(text)[:max_tokens]
    inp = mx.array(tokens)[None, :-1]
    tgt = mx.array(tokens)[None, 1:]
    logits = model(inp)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    nll = -mx.take_along_axis(logprobs, tgt[..., None], axis=-1).mean()
    return float(mx.exp(nll))
