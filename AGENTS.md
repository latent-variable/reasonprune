# reasonprune — agent notes

Toolkit for reasoning-preserving, knowledge-shedding pruning of local MLX models.
Read `DESIGN.md` first: hypotheses, method, experimental ladder.

## Operating

- Python: reuse the oMLX venv — `~/Documents/LatentPlayground/omlx/.venv-codex/bin/python`
  (mlx 0.31.2 / mlx-lm 0.31.3, matches the running server). No project venv.
- Full pipeline: `scripts/experiment.py {baseline,score,sweep} --model qwen-0.8b`.
  Results append to `results/<model>/` (sweep.jsonl is resumable — done configs skip).
- Data: `python3 -c "from pathlib import Path; from reasonprune.datagen import generate_all; generate_all(Path('data'))"`.
  Deterministic (seed 7). Reasoning items are correct by construction; knowledge
  items come from `data/facts/*.tsv` (curated, verified — don't add unverified facts).
- oMLX server (synthetic paraphrase only, NOT used for evals): localhost:5599,
  key in `~/.omlx/settings.json`, client in `reasonprune/worker.py`.
- Memory: the M5 Max has 128 GB but the oMLX server holds ~50+ GB resident.
  Loading the 35B-A3B (35 GB) alongside is fine; don't load two big models at once.

## Gotchas

- `~/.cache/huggingface/hub/models--Qwen--*` entries may be TEMPLATE-ONLY stubs
  (just chat_template.jinja, no weights). Check for *.safetensors before assuming.
- Qwen3.5 dense = hybrid arch (GatedDeltaNet + full attention every 4th layer);
  MLPs are standard SwiGLU (`mlx_lm/models/qwen3_5.py`, MLP from qwen3_next).
- Eval prompts disable thinking via `enable_thinking=False`; the template injects
  an empty `<think></think>`. `strip_thinking` handles models that still emit it.
- Pruning = masking (zeroing channels), mathematically equal to removal; sweeps
  reload the model per config rather than undoing masks.
- Unstructured sparsity gives ZERO speedup on Metal — findings only count when
  cashed out as whole channels/experts/layers.
- Reference J-lens implementation (torch): ~/Documents/LatentPlayground/jacobian-lens.

## Agent context (scope + memory)
<!-- BEGIN agent-context (managed by ~/.agents/bin/project-sync.sh) -->
- You are in **PROJECT scope** (this repo). User-scope canon = `~/.agents` and transcends projects — don't conflate them. `.claude`/`.agents` here may be symlinks; verify with `readlink` before claiming a write landed.
- Project memory + shared skills: `.agents/` (gitignored). Read `.agents/memory/MEMORY.md` first.
- **Commit proactively** (canon doctrine): finished+tested chunk → commit. Commits are free and revertible.
- **Why: nightly audit.** `latent-git-agents` audits only **committed code on the default branch**. Uncommitted / branch-stranded work is invisible to it — no review, no fixes. Finishing work without committing drops it out of coverage; if you leave any uncommitted, flag it to Lino.
- Refresh infra: `~/.agents/bin/project-sync.sh .`
<!-- END agent-context -->
