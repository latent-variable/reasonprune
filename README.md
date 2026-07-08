# reasonprune

Prune local LLMs to keep the reasoning engine and shed memorized trivia.

Importance of every MLP channel (or MoE expert) is measured twice: once on
closed-book factual recall, once on in-context reasoning where all facts are
given in the prompt. Channels that knowledge needs but reasoning doesn't get
pruned; shared circuits are guarded. Inspired by Anthropic's workspace paper
(reasoning lives in a small mid-layer subspace; recall lives in the rest) and
the selective-pruning unlearning line of work. Everything runs on Apple
Silicon via MLX.

## Run

```bash
PY=~/Documents/LatentPlayground/omlx/.venv-codex/bin/python  # mlx venv

$PY scripts/experiment.py baseline --model qwen-0.8b   # eval unpruned
$PY scripts/experiment.py score    --model qwen-0.8b   # importance on K and R sets
$PY scripts/experiment.py sweep    --model qwen-0.8b \
    --strategies diff,know,lowmag,random --fracs 0.1,0.2,0.3,0.4
$PY scripts/chart.py --model qwen-0.8b                 # tradeoff curves + heatmap

# MoE expert-level (Qwen3.6-35B-A3B):
$PY scripts/experiment.py score-moe --model qwen-35b-a3b
$PY scripts/experiment.py sweep-moe --model qwen-35b-a3b --fracs 0.1,0.25,0.5
```

Results land in `results/<model>/` (sweeps are resumable). Datasets are
deterministic and regenerable; reasoning items are correct by construction,
knowledge items come from curated fact tables in `data/facts/`.

## Status

Early results on Qwen3.5-0.8B: pruning 10% of MLP channels by the guarded
differential score drops closed-book knowledge 0.80 to 0.30 while in-context
reasoning holds (0.68 to 0.69). Full sweeps, baselines, and the MoE headline
experiment in progress.

Implementation detail and operating contract: `AGENTS.md`. Method and
hypotheses: `DESIGN.md`.
