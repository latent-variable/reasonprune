# reasonprune — design

Prune local LLMs to keep the reasoning engine and shed parametric trivia.
Inspired by Anthropic's workspace/J-space paper (transformer-circuits.pub/2026/workspace):
ablating the J-space workspace kills multi-hop reasoning while leaving shallow recall
untouched. We want the *inverse* operation: identify and remove the components that
serve closed-book factual recall, protect everything the reasoning workspace depends on.

## Hypothesis

H1. Parametric factual knowledge and multi-step reasoning are carried by partially
    disjoint structured components (MLP neurons / MoE experts / heads). Mid-layer MLPs
    act as key-value fact memories (Geva et al., ROME); reasoning flows through
    attention + the workspace band of the residual stream.

H2. A contrastive importance score — importance on a knowledge-recall calibration set
    minus importance on an in-context-reasoning calibration set — separates these
    populations well enough that pruning the knowledge-heavy, reasoning-light tail
    degrades closed-book recall much faster than reasoning benchmarks.

H3 (stretch). Protecting components whose outputs align with J-space directions
    (cheap sketched Jacobian lens) improves the tradeoff over pure contrastive scores.

## Method

### Signals (per structured unit: MLP hidden neuron, attention head, MoE expert)

1. **Activation salience** (Wanda-style, no gradients): E over calib set of
   |activation| x ||output weight row||. Cheap, works quantized.
2. **Gradient x activation** (first-order Taylor of loss change if unit removed):
   needs backprop; run on small/dense models in fp16 via mlx autograd.
3. **Differential score**: D(u) = I_know(u) − λ·I_reason(u), normalized per layer.
   Prune descending D with an overlap guard: never prune u if I_reason(u) is above
   the p-th percentile of its layer (protect shared circuits).
4. **J-space guard (v2)**: sketch J_l = E[∂h_final/∂h_l] with sampled VJPs
   (random unembedding directions, ~64–256 probes/layer). Protect units whose
   write-vectors have high alignment with the top J-space cone on reasoning prompts.

### Calibration sets (contrast is the whole game)

- **K (knowledge)**: closed-book factual recall. Trivia QA-style, entity facts,
  long-tail geography/people/dates. Model must answer *from weights*.
- **R (reasoning)**: multi-step problems where all needed facts are IN CONTEXT:
  GSM8K-style word math, logic grids, multi-hop over a provided passage,
  tool-call planning traces, instruction following. Nothing depends on world facts.
- Both partially synthetic, generated via local Qwen3-30B-A3B worker through oMLX.
- Held-out eval splits are disjoint from calibration.

### Pruning modes

- MoE expert removal (REAP-style but contrastive) — Qwen3-30B-A3B target.
- Structured MLP-neuron pruning (rows/cols of gate/up/down) — dense Qwen3 targets.
- Head pruning and layer dropping as baselines.
- Baselines to beat: uniform magnitude, Wanda on mixed calib, random units.

### Evaluation

- **Knowledge axis**: closed-book QA accuracy (held-out), country-capital probes,
  PopQA-style long tail.
- **Reasoning axis**: GSM8K subset, in-context multi-hop, IFEval-lite instruction
  compliance, tool-call JSON validity, bAbI-style tasks.
- **Sanity**: perplexity on neutral text, generation coherence.
- Success = curves that separate: at X% units pruned, knowledge acc drops >> reasoning acc.
- All runs logged to results/*.json; charts of tradeoff curves + layer heatmaps.

## Experimental ladder (models actually on disk)

1. **Qwen3.5-0.8B dense** (HF cache) — validate the whole pipeline fast
   (scores, prune, eval, charts). Signal may be weak at this scale; it's plumbing.
2. **Qwen3.5-2B / 4B dense** (HF cache) — first real result. Gradient scores feasible.
3. **Qwen3.6-35B-A3B MoE** (8bit, ~/.omlx/models) — expert-level contrastive
   pruning; the headline. Same family as the oMLX worker.
4. (If time) J-space guard ablation study: does H3 beat H2 alone?

## Environment

- Apple M5 Max, 128 GB unified memory. Mind the oMLX server's resident models.
- oMLX server: http://localhost:5599 (OpenAI-compatible, auth required; key in
  ~/.omlx/settings.json). Worker for data paraphrase: Qwen3.6-35B-A3B-oQ4e-mtp.
- Direct analysis venv: reuse omlx/.venv-codex (mlx 0.31.2, mlx-lm 0.31.3, py>=3.11).
- Analysis models load from ~/.cache/huggingface/hub via mlx_lm.load().

## Non-goals (for now)

- Distillation (the Reddit post's other half) — noted in FUTURE.md, out of scope.
- Recovering pruned quality via finetuning/healing — maybe a light LoRA heal pass
  at the end if curves look good.
- Anything requiring CUDA. Everything runs on Apple Silicon via MLX.

## Repo layout (planned)

```
reasonprune/
  reasonprune/        python package: scoring, pruning, eval, jlens
  data/               calibration + eval sets (generated, versioned by script)
  scripts/            gen_data, score, prune, eval, chart entrypoints
  results/            json logs + charts
  DESIGN.md           this file
  AGENTS.md           operating contract
```
