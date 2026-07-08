

## Agent context (scope + memory)
<!-- BEGIN agent-context (managed by ~/.agents/bin/project-sync.sh) -->
- You are in **PROJECT scope** (this repo). User-scope canon = `~/.agents` and transcends projects — don't conflate them. `.claude`/`.agents` here may be symlinks; verify with `readlink` before claiming a write landed.
- Project memory + shared skills: `.agents/` (gitignored). Read `.agents/memory/MEMORY.md` first.
- **Commit proactively** (canon doctrine): finished+tested chunk → commit. Commits are free and revertible.
- **Why: nightly audit.** `latent-git-agents` audits only **committed code on the default branch**. Uncommitted / branch-stranded work is invisible to it — no review, no fixes. Finishing work without committing drops it out of coverage; if you leave any uncommitted, flag it to Lino.
- Refresh infra: `~/.agents/bin/project-sync.sh .`
<!-- END agent-context -->
