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
6. **Serial test execution** — at most **one** pytest/Makefile test process repo-wide; executors run **targeted** tests only; the **coordinator** runs `make test-fast` after each `land-issue` (see below).

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
# Per issue: land-issue → make test-fast (coordinator only) → ROADMAP Done → gh issue close → wrap-up
uv run python scripts/roadmap.py land-issue --issue N
# Confirm no executor pytest still running (terminals folder), then:
make test-fast
```

Do **not** trust subagent prose for wrap-up — verify claims, merges, gates, and **coordinator** `make test-fast` in the parent session.

Each subagent prompt must also state: **do not run `make test-fast`** — targeted tests only (see [Parallel work: serial test execution](#parallel-work-serial-test-execution)).

## Worker checklist

```bash
export ORBIT_WARS_ISSUE_ID=127 ORBIT_WARS_AGENT_ID=cursor-issue-127
uv run python scripts/roadmap.py begin "<user request>"
uv run python scripts/roadmap.py claim --issue 127 --path src/telemetry/ --setup-worktree
uv run python scripts/roadmap.py agent-workspace --issue 127   # open printed path in Cursor
uv run python scripts/roadmap.py approve-impl --issue 127 --summary "…"
# implement + commit in worktree (no push)
# targeted tests only — see issue table below; do NOT run make test-fast
uv run --group dev pytest tests/test_metric_registry.py -m "not slow and not jax" -q
# hand off to coordinator (worker does NOT land or run full suite)
```

Coordinator at repo root after worker commits:

```bash
uv run python scripts/roadmap.py land-issue --issue 127
make test-fast   # coordinator only; one test process repo-wide
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

## Parallel work: serial test execution

**Hard rule:** At most **one** pytest or Makefile test process repo-wide at any time. Five concurrent `make test-fast` jobs can lock up WSL2 workstations. Multitask parallelism applies to **implementation**, not **verification**.

### Roles

| Role | Testing responsibility |
|------|------------------------|
| **Executor (worker)** | **Targeted tests only** for files/areas touched; smallest command that covers the change |
| **Coordinator (parent)** | **`make test-fast`** once per issue at **land** time on repo root `main` (after `land-issue`, before `wrap-up`) |

### Executor rules

1. Before running any test, check the **terminals folder** — if another session is already running pytest or `make test-*`, **wait** or skip and report `tests deferred to coordinator`.
2. Never run `make test-fast`, `make test`, `make test-jax`, or unfiltered `pytest`.
3. Prefer Makefile domain targets when they match; otherwise single test **files** from the issue's claim paths.
4. If an issue needs two targeted commands, run them **serially** in the same worker — never parallel pytest in one worker.
5. Workers report: targeted command run (or `deferred`), pass/fail, files changed — **not** full-suite coverage.

### Coordinator rules

1. Workers land **without** having run the full suite; the coordinator owns verification on `main`.
2. After each `land-issue --issue N`: confirm no executor pytest is still running, then run **`make test-fast`** (serial).
3. Do **not** start the next parallel batch or the next `land-issue` until coordinator `make test-fast` on `main` has finished.
4. Optional smoke after land when an issue touches train/CLI/submission: `uv run ow train …` — also **serial**, never parallel with pytest.

### Targeted tests — src audit issues (#136–#155)

Executors run the **smallest** row that covers their change. Coordinator always runs `make test-fast` after land.

| Issue | Executor targeted command |
|-------|---------------------------|
| **#136** | `uv run --group dev pytest tests/test_cli_train_hosts.py -m "not slow and not jax" -q` |
| **#137** | `make test-domain-artifacts` or `pytest tests/test_tournament.py tests/test_run_paths.py -m "not slow and not jax" -q` |
| **#138** | `uv run --group dev pytest tests/test_kaggle_runner.py -m "not slow and not jax" -q` |
| **#139** | `uv run --group dev pytest tests/test_metric_registry.py -m "not slow and not jax" -q` |
| **#140** | `uv run --group dev pytest tests/test_metric_registry.py tests/test_telemetry.py -m "not slow and not jax" -q` |
| **#141** | `make test-domain-config` |
| **#142** | `uv run --group dev pytest tests/test_trajectory_shield.py -m "not slow and not jax" -q` |
| **#143** | `uv run --group dev pytest tests/test_curriculum.py -m "not slow and not jax" -q` |
| **#144** | `uv run --group dev pytest tests/test_feature_registry.py tests/test_feature_encoding_golden.py -m "not slow and not jax" -q` |
| **#145** | `make test-domain-config` |
| **#146** | `uv run --group dev pytest tests/test_cli_train_hosts.py tests/test_kaggle_runner.py -m "not slow and not jax" -q` |
| **#147** | `uv run --group dev pytest tests/test_trajectory_shield.py -m "not slow and not jax" -q` |
| **#148** | `uv run --group dev pytest tests/test_jax_scripted_opponents.py -m "jax and not slow" -q` |
| **#149** | `make test-domain-config` then `pytest tests/test_curriculum.py -m "not slow and not jax" -q` (serial) |
| **#150** | `uv run --group dev pytest tests/test_promotion.py tests/test_tournament.py -m "not slow and not jax" -q` |
| **#151–#153** | `uv run --group dev pytest tests/test_artifact_pipeline.py tests/test_promotion.py tests/test_tournament.py -m "not slow and not jax" -q` (+ new module tests if added) |
| **#152** | `uv run --group dev pytest tests/test_metric_registry.py tests/test_telemetry.py -m "not slow and not jax" -q` |
| **#154** | `make test-domain-features` |
| **#155** | `make test-domain-features` then `pytest tests/test_intercept_edge_features.py tests/test_checkpoint_compat.py -m "not slow and not jax" -q` (serial) |

**Coordinator smoke (optional, serial):** after landing #148, #155, or CLI/train issues — `uv run ow train …` smoke; never while pytest is running.

See also `AGENTS.md` **Test Selection For Coding Agents** for domain Makefile targets.

## Recovery

```bash
uv run python scripts/roadmap.py claims --stale
uv run python scripts/roadmap.py release-stale --apply
uv run python scripts/roadmap.py clear-impl --all-gates
uv run python scripts/roadmap.py check-session --global
```

See also [OWNERSHIP.md](OWNERSHIP.md) and ROADMAP **Agent workflow** in [ROADMAP.md](ROADMAP.md).
