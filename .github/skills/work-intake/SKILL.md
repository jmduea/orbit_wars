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
uv run python scripts/roadmap.py claim --issue N --path <dirs from docs/OWNERSHIP.md>
uv run python scripts/roadmap.py approve-impl --issue N --summary "<short scope>"
uv run python scripts/roadmap.py gate --request "<user request>" --require-allowed
```

## Phase 6 — Implement

- Branch `issue/N-short-slug`
- Tests per `AGENTS.md` tiers

## Phase 7–8 — Wrap-up + Done

```bash
gh issue close N --comment "Evidence: …"
uv run python scripts/roadmap.py wrap-up --issue N --evidence "tests + commit + paths (≥40 chars)"
uv run python scripts/roadmap.py check-wrap-up --issue N --require-passed
uv run python scripts/roadmap.py check-session --require-clean
uv run python scripts/roadmap.py validate
```

Move ROADMAP row to **Done**; manifest `complete` when applicable.

## Hooks

`vscode_askQuestions` when `requires_planning` and scope is ambiguous (header: `work-intake-planning`).

## Disable guards (tests only)

`ORBIT_WARS_HOOK_DISABLE=1` — never set during normal agent work.
