# Multi-agent playbook

How to run **parallel Cursor agents** without colliding on git state, ROADMAP, or `.omg/state/`.

## Golden rules

1. **One GitHub issue per agent** — never bundle multiple issues in one worker.
2. **One worktree per issue** — `worktrees/issue-N/` on branch `issue/N-slug`.
3. **Set env vars before editing `src/`, `conf/`, or `tests/`:**
   ```bash
   export ORBIT_WARS_ISSUE_ID=127
   export ORBIT_WARS_AGENT_ID=cursor-issue-127   # unique per worker
   source scripts/agent_env.sh
   ```
4. **Per-issue impl-gates** — `.omg/state/impl-gates/issue-N.json` (not a single global file).
5. **Always wrap up** — stale claims block path overlap for everyone.

## Git landing (read this)

Worktrees exist for **isolated editing**, not as the long-lived branch on GitHub.

| Phase | Where | What |
|-------|--------|------|
| **Implement** | `worktrees/issue-N/` on `issue/N-*` | Edit code, commit locally on the issue branch |
| **Land** | Repo root on `main` | `roadmap.py land-issue --issue N` merges issue branch → `main` |
| **Publish** | Repo root | `git push origin main` **only when the user asks** |

**Do not** `git push origin issue/N-…`. The pre-tool hook blocks pushes of `issue/*` branches.

```bash
# In worktree — commit only
cd worktrees/issue-127
git add … && git commit -m "feat(telemetry): …"

# At repo root — merge to main
cd /path/to/orbit_wars
uv run python scripts/roadmap.py land-issue --issue 127
make test-fast

# Push only on explicit user request
git push origin main
```

**PR alternative:** If you prefer a pull request, open one from the local issue branch without pushing the issue branch first (`gh pr create` can push once — set `ORBIT_WARS_ALLOW_ISSUE_BRANCH_PUSH=1` only for that deliberate step). Default agent flow is **land-issue → main**.

## Coordinator (parent) checklist

Before spawning parallel executors:

```bash
uv run python scripts/roadmap.py claims
uv run python scripts/roadmap.py claims --stale
uv run python scripts/roadmap.py release-stale --apply   # drop finished claims
uv run python scripts/roadmap.py check-session --global --require-clean
```

Each subagent prompt must include:

- `export ORBIT_WARS_ISSUE_ID=N`
- `export ORBIT_WARS_AGENT_ID=cursor-issue-N`
- `uv run python scripts/roadmap.py claim --issue N --path … --setup-worktree`
- `cd worktrees/issue-N/` (or `agent-workspace --issue N`)
- **Do not push issue branches** — parent runs `land-issue` per finished worker

After workers finish (parent turn):

```bash
uv run python scripts/roadmap.py check-session --global --require-clean
# Per issue: land-issue → tests → ROADMAP Done → gh issue close → wrap-up
uv run python scripts/roadmap.py land-issue --issue N
```

Do **not** trust subagent prose for wrap-up — verify claims, merges, and gates in the parent session.

## Worker checklist

```bash
export ORBIT_WARS_ISSUE_ID=127 ORBIT_WARS_AGENT_ID=cursor-issue-127
uv run python scripts/roadmap.py begin "<user request>"
uv run python scripts/roadmap.py claim --issue 127 --path src/telemetry/ --setup-worktree
uv run python scripts/roadmap.py agent-workspace --issue 127   # open printed path in Cursor
uv run python scripts/roadmap.py approve-impl --issue 127 --summary "…"
# implement + commit in worktree (no push)
make test-fast
# hand off to coordinator OR from repo root:
uv run python scripts/roadmap.py land-issue --issue 127
# ROADMAP Done row + make roadmap-check
gh issue close 127 --comment "Evidence: …"
uv run python scripts/roadmap.py wrap-up --issue 127 --evidence "land-issue + make test-fast; …"
uv run python scripts/roadmap.py check-session --require-clean
```

## Commands reference

| Command | Purpose |
|---------|---------|
| `land-issue --issue N` | Merge `issue/N-*` into `main` at repo root |
| `land-issue --issue N --dry-run` | Show merge plan without applying |
| `push-guard` | Hook helper; blocks `git push` of `issue/*` |
| `claims --stale` | List open claims for Done/closed/malformed issues |
| `release-stale --apply` | Release stale claims + clear their impl-gates |
| `check-session --global` | All open claims repo-wide (coordinators) |
| `agent-workspace --issue N` | Worktree path + env exports + git landing hints |
| `approve-impl --issue N` | Writes `.omg/state/impl-gates/issue-N.json` |
| `clear-impl --all-gates` | Remove all gates (recovery) |

Override issue-branch push block (rare): `ORBIT_WARS_ALLOW_ISSUE_BRANCH_PUSH=1`.

## ROADMAP serialization

Multiple agents finishing at once will conflict on `docs/ROADMAP.md`. Prefer:

1. Workers commit code on issue branches in worktrees.
2. Coordinator runs `land-issue` and updates ROADMAP **Done** rows sequentially on `main`.
3. Run `make roadmap-check` once after ROADMAP edits.

## Path claims

Use **narrow** paths from [OWNERSHIP.md](OWNERSHIP.md). Broad claims (e.g. all of `src/jax/`) block unrelated parallel work.

Repeat `--path` for each directory — never comma-separated paths in one argument.

## Recovery

```bash
uv run python scripts/roadmap.py claims --stale
uv run python scripts/roadmap.py release-stale --apply
uv run python scripts/roadmap.py clear-impl --all-gates
uv run python scripts/roadmap.py check-session --global
```

See also [OWNERSHIP.md](OWNERSHIP.md) and ROADMAP **Agent workflow** in [ROADMAP.md](ROADMAP.md).
