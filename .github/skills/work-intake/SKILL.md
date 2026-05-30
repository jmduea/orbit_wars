---
name: work-intake
description: >
  ROADMAP funnel for agents: auto-run on any implementation request (free-form chat).
  Maps user text through intake, claim, approve-impl, implement, wrap-up.
  Activate when: fix, implement, build, bug, issue #, roadmap work, or before src/conf/tests edits.
argument-hint: "[optional — defaults to latest user message]"
---

# Work intake (ROADMAP funnel)

Users may speak in **free form**. You must run the funnel without asking them to invoke this skill.

## Automatic entry (every implementation session)

On the **first** turn where the user wants code changes (any wording):

```bash
export ORBIT_WARS_AGENT_ID="${ORBIT_WARS_AGENT_ID:-cursor-$(hostname)-$$}"
# One issue per parallel agent/subagent:
# export ORBIT_WARS_ISSUE_ID=102
uv run python scripts/roadmap.py begin "<verbatim user request>"
uv run python scripts/omg_workflow_manifest.py active
```

Read JSON output:

| Field | Meaning |
|-------|---------|
| `may_implement: false` | **Stop** — follow `next_steps` (planning, Later capture, claim, approve-impl) |
| `capture_to: later` | ROADMAP **Later** only; no `src/conf/tests` edits |
| `requires_planning: true` | `/ralplan` or `/deep-interview` before approve-impl |
| `primary_issue` | Default GitHub issue for claim/approve/wrap-up |

**Do not** edit `src/`, `conf/`, or `tests/` until `approve-impl` succeeds. Cursor **pre-tool hook** blocks those paths without `.omg/state/impl-gate.json`.

`docs/brain_dump.md` is **retired** — never triage it.

## Phase 2–3 — Planning (when `requires_planning`)

- Consensus plan in `.omg/plans/`, manifest `planned`/`executing`
- No implementation-path edits during planning
- GitHub issues + ROADMAP promote after execution plan

## Phase 4–5 — Claim + approve

```bash
uv run python scripts/roadmap.py claims
export ORBIT_WARS_ISSUE_ID=N
export ORBIT_WARS_AGENT_ID="cursor-issue-N"   # unique per parallel worker
uv run python scripts/roadmap.py claim --issue N --path <dirs> --setup-worktree
# Open the printed worktrees/issue-N/ path as this agent's workspace (or cd there).
uv run python scripts/roadmap.py approve-impl --issue N --summary "<short scope>"
uv run python scripts/roadmap.py gate --request "<user request>" --require-allowed
```

**Parallel subagents:** never share `main` for `src/`/`conf/`/`tests/` edits. Each worker gets its own `ORBIT_WARS_AGENT_ID`, `ORBIT_WARS_ISSUE_ID`, branch `issue/N-…`, and `worktrees/issue-N/`.

## Multitask / background subagents

When a parent agent spawns executor subagents for parallel work:

1. **Parent must not** spawn executors for `src/`/`conf/`/`tests/` without active claims and per-issue env vars in each subagent prompt.
2. **One issue per executor** — never bundle multiple issues in one worker.
3. **Task prompt must include** `export ORBIT_WARS_AGENT_ID=…` and `export ORBIT_WARS_ISSUE_ID=N` (unique per worker).
4. **Require** `claim --setup-worktree` and cwd `worktrees/issue-N/` before implementation edits.
5. **Parent waits** for workers, runs `check-session --require-clean` in the parent turn, and does not run parallel executors on the shared repo root without worktrees.

## Phase 6 — Implement

- Work only on the issue branch inside the issue worktree (hook blocks protected `main` once the claim records a branch).
- Tests per `AGENTS.md` tiers

## Phase 7–9 — ROADMAP Done, wrap-up, session end

**Order matters** — do not push until `make roadmap-check` passes:

1. Add a row under ROADMAP **Done** (≤5 items; drop oldest Done row if at cap).
2. Remove the row from **Now** or **Next** (never leave a closed issue only in Later/Now).
3. `make roadmap-check` (or `roadmap.py validate` + `pytest tests/test_roadmap.py -q`).
4. Commit ROADMAP with the code change when possible (same PR).

```bash
gh issue close N --comment "Evidence: …"
uv run python scripts/roadmap.py wrap-up --issue N --evidence "tests + commit + paths (≥40 chars)"
uv run python scripts/roadmap.py check-wrap-up --issue N --require-passed
uv run python scripts/roadmap.py check-session --require-clean
```

`wrap-up` **blocks** when GitHub issue is closed (or `--skip-github-check` in tests) but the issue is not listed under **Done**. Manifest `complete` when applicable.

## Hooks

`vscode_askQuestions` when `requires_planning` and scope is ambiguous (header: `work-intake-planning`).

## Disable guards (tests only)

`ORBIT_WARS_HOOK_DISABLE=1` — never set during normal agent work.
