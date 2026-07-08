"""Calibration + eval set generation.

Two contrasting distributions:
  K (knowledge): closed-book factual recall — answer must come from weights.
  R (reasoning): multi-step problems with ALL facts in context — answer must
                 come from manipulating the prompt, never from world knowledge.

R is generated programmatically so ground truth is correct by construction.
K comes from curated fact tables (see data/facts/*.tsv) so truth is verified.
An optional LLM paraphrase pass adds surface diversity without touching answers.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared types


@dataclass
class Item:
    id: str
    kind: str          # e.g. "know.capital", "reason.arith2"
    prompt: str
    answer: str        # canonical answer (short string, exact-match after norm)
    aliases: list      # acceptable alternative answers
    meta: dict

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _norm(s: str) -> str:
    return " ".join(s.lower().replace(",", "").replace(".", "").split())


def matches(pred: str, item: Item) -> bool:
    p = _norm(pred)
    golds = [item.answer] + list(item.aliases)
    return any(_norm(g) in p for g in golds if g)


# ---------------------------------------------------------------------------
# K: knowledge probes from fact tables

KNOW_TEMPLATES = {
    "capital": [
        "What is the capital of {key}? Answer with just the city name.",
        "The capital city of {key} is",
    ],
    "element": [
        "What is the chemical symbol for {key}? Answer with just the symbol.",
        "The chemical symbol for the element {key} is",
    ],
    "author": [
        "Who wrote the book \"{key}\"? Answer with just the author's name.",
        "The novel \"{key}\" was written by",
    ],
    "state_capital": [
        "What is the capital of the US state of {key}? Answer with just the city.",
        "The capital of the US state of {key} is",
    ],
    "currency": [
        "What is the official currency of {key}? Answer with just the currency name.",
        "The official currency of {key} is the",
    ],
}


def load_fact_table(path: Path) -> list[tuple[str, str, list]]:
    """TSV: key<TAB>answer[<TAB>alias1|alias2]. Lines starting with # skipped."""
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        key, ans = parts[0], parts[1]
        aliases = parts[2].split("|") if len(parts) > 2 else []
        rows.append((key, ans, aliases))
    return rows


def gen_knowledge(facts_dir: Path, rng: random.Random) -> list[Item]:
    items = []
    for table, templates in KNOW_TEMPLATES.items():
        tsv = facts_dir / f"{table}.tsv"
        if not tsv.exists():
            continue
        for i, (key, ans, aliases) in enumerate(load_fact_table(tsv)):
            tpl = templates[i % len(templates)]
            items.append(Item(
                id=f"know.{table}.{i:04d}",
                kind=f"know.{table}",
                prompt=tpl.format(key=key),
                answer=ans,
                aliases=aliases,
                meta={"key": key, "table": table},
            ))
    rng.shuffle(items)
    return items


# ---------------------------------------------------------------------------
# R: reasoning problems, correct by construction

FIRST = ["Ava", "Ben", "Chloe", "Dan", "Elena", "Farid", "Gita", "Hugo",
         "Iris", "Jonas", "Kira", "Liam", "Mona", "Nico", "Omar", "Priya"]
OBJECTS = ["apples", "pencils", "coins", "stickers", "marbles", "books",
           "cards", "shells", "bolts", "tokens"]
PLACES = ["kitchen", "garage", "office", "garden", "attic", "hallway",
          "studio", "basement", "balcony", "workshop"]


def gen_arith2(rng: random.Random, i: int) -> Item:
    """Two-step arithmetic word problem."""
    a, b, m = rng.randint(3, 40), rng.randint(3, 40), rng.randint(2, 9)
    p1, p2 = rng.sample(FIRST, 2)
    obj = rng.choice(OBJECTS)
    op = rng.choice(["gives", "takes"])
    if op == "gives":
        total = (a + b) * m
        story = (f"{p1} has {a} {obj}. {p2} gives {p1} {b} more {obj}. "
                 f"Then {p1}'s collection is multiplied {m} times in a game. "
                 f"How many {obj} does {p1} have now? Answer with just the number.")
    else:
        a = max(a, b + 2)
        total = (a - b) * m
        story = (f"{p1} has {a} {obj}. {p2} takes {b} {obj} away from {p1}. "
                 f"Then {p1}'s collection is multiplied {m} times in a game. "
                 f"How many {obj} does {p1} have now? Answer with just the number.")
    return Item(f"reason.arith2.{i:04d}", "reason.arith2", story, str(total), [], {})


def gen_arith1(rng: random.Random, i: int) -> Item:
    """Single-step arithmetic — easy enough for sub-1B models to answer directly."""
    a, b = rng.randint(6, 60), rng.randint(2, 30)
    p1, p2 = rng.sample(FIRST, 2)
    obj = rng.choice(OBJECTS)
    if rng.random() < 0.5:
        ans = a + b
        story = (f"{p1} has {a} {obj}. {p2} gives {p1} {b} more {obj}. "
                 f"How many {obj} does {p1} have now? Answer with just the number.")
    else:
        a = max(a, b + 3)
        ans = a - b
        story = (f"{p1} has {a} {obj}. {p2} takes {b} {obj} from {p1}. "
                 f"How many {obj} does {p1} have now? Answer with just the number.")
    return Item(f"reason.arith1.{i:04d}", "reason.arith1", story, str(ans), [], {})


def gen_multihop(rng: random.Random, i: int) -> Item:
    """2-hop relational chain over synthetic entities. Facts fully in context."""
    people = rng.sample(FIRST, 4)
    places = rng.sample(PLACES, 4)
    # chain: p0 works with p1; p1 is in places[k]
    k = rng.randrange(4)
    facts = []
    assignment = dict(zip(people, places))
    assignment[people[1]] = places[k]
    for p in people[1:]:
        facts.append(f"{p} is in the {assignment[p]}.")
    facts.append(f"{people[0]} is working with {people[1]}.")
    rng.shuffle(facts)
    prompt = (" ".join(facts) +
              f" In which room is the person working with {people[0]}? "
              f"Answer with just the room name.")
    return Item(f"reason.hop2.{i:04d}", "reason.hop2", prompt, places[k], [], {})


def gen_compare_chain(rng: random.Random, i: int) -> Item:
    """Transitive ordering: A > B > C > D, ask for max/min."""
    people = rng.sample(FIRST, 4)
    order = people[:]  # order[0] tallest
    stmts = [f"{order[j]} is taller than {order[j+1]}." for j in range(3)]
    rng.shuffle(stmts)
    ask_max = rng.random() < 0.5
    q = "tallest" if ask_max else "shortest"
    ans = order[0] if ask_max else order[-1]
    prompt = (" ".join(stmts) + f" Who is the {q}? Answer with just the name.")
    return Item(f"reason.order.{i:04d}", "reason.order", prompt, ans, [], {})


def gen_toolcall(rng: random.Random, i: int) -> Item:
    """Emit a correct JSON tool call from an in-context tool spec."""
    city = rng.choice(["Osaka", "Lyon", "Cusco", "Tallinn", "Da Nang", "Windhoek"])
    unit = rng.choice(["celsius", "fahrenheit"])
    prompt = (
        'You have one tool:\n'
        '{"name": "get_weather", "parameters": {"city": "string", "unit": "celsius|fahrenheit"}}\n'
        f'The user asks: "What is the weather in {city}, in {unit}?"\n'
        'Reply with ONLY the JSON tool call object with keys "name" and "arguments".'
    )
    ans = json.dumps({"name": "get_weather",
                      "arguments": {"city": city, "unit": unit}})
    return Item(f"reason.tool.{i:04d}", "reason.tool", prompt, ans, [],
                {"check": "json_tool", "city": city, "unit": unit})


def gen_openbook(rng: random.Random, i: int) -> Item:
    """Fact given IN CONTEXT with fictional entities — recall machinery not needed.

    Mirrors the knowledge templates but self-contained: tests reading, not memory.
    Used to verify pruning kept in-context extraction intact.
    """
    fake_country = rng.choice(["Veldoria", "Kastrania", "Ombrelle", "Tessary",
                               "Quillmark", "Zavendia"])
    fake_city = rng.choice(["Port Halden", "Miravelle", "Doria", "Askenholm",
                            "Ruta Vieja", "Calder Point"])
    prompt = (f"According to the atlas, the capital of {fake_country} is {fake_city}. "
              f"What is the capital of {fake_country}? Answer with just the city name.")
    return Item(f"reason.openbook.{i:04d}", "reason.openbook", prompt, fake_city, [], {})


REASON_GENERATORS = {
    "arith1": gen_arith1,
    "arith2": gen_arith2,
    "hop2": gen_multihop,
    "order": gen_compare_chain,
    "tool": gen_toolcall,
    "openbook": gen_openbook,
}


def gen_reasoning(n_per_kind: int, rng: random.Random) -> list[Item]:
    items = []
    for name, fn in REASON_GENERATORS.items():
        for i in range(n_per_kind):
            items.append(fn(rng, i))
    rng.shuffle(items)
    return items


# ---------------------------------------------------------------------------
# Entry


def write_jsonl(items: list[Item], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(it.to_json() for it in items) + "\n")


def generate_all(data_dir: Path, seed: int = 7, n_reason_per_kind: int = 120,
                 calib_frac: float = 0.5) -> dict:
    """Generate K and R, each split into calibration and held-out eval."""
    rng = random.Random(seed)
    know = gen_knowledge(data_dir / "facts", rng)
    reason = gen_reasoning(n_reason_per_kind, rng)

    out = {}
    for name, items in (("knowledge", know), ("reasoning", reason)):
        cut = int(len(items) * calib_frac)
        write_jsonl(items[:cut], data_dir / f"{name}.calib.jsonl")
        write_jsonl(items[cut:], data_dir / f"{name}.eval.jsonl")
        out[name] = {"calib": cut, "eval": len(items) - cut}
    return out
