---
name: work-intake
description: >
  Mandatory ROADMAP-first intake and implementation gate before coding.
  Activate when: start work, implement, fix, build, new task, pick up issue,
  or before editing src/conf/tests.
argument-hint: "<what you want to do>"
---

# Work intake (ROADMAP funnel)

Single funnel: **ROADMAP → planning → execution plan → issues/manifest → implement**.

`docs/brain_dump.md` is **retired** — never read it as backlog.

## Phase 0 — Status

```bash
uv run python scripts/roadmap.py agent
uv run python scripts/omg_workflow_manifest.py active
```

Read linked GitHub issues for **Now** rows. Human **Now** wins over manifest.

## Phase 1 — Intake

```bash
uv run python scripts/roadmap.py intake "<user request>"
```

| Outcome | Action |
|---------|--------|
| `capture_to: later` | Add **Later** row only; run `/deep-interview` or `/ralplan`; **stop** |
| `requires_planning: true` | `/deep-interview` → `/ralplan` (or `/omg-autopilot` through **spec approval**) |
| `suggested_workflow: execute` | Known **Now** issue — still run execution plan if multi-file |
| `suggested_workflow: quick` | Trivial doc/typo only — `approve-impl` then edit |

## Phase 2 — Planning

- **Non-trivial:** consensus plan in `.omg/plans/`, manifest `planned`/`executing`
- **Do not** edit `src/`, `conf/`, `tests/` during planning

## Phase 3 — Execution plan

Before code:

1. Chunk order and acceptance criteria
2. Create/update **GitHub issues** (`type:*` + `area:*`) from plan
3. Promote ROADMAP rows to **Next** / **Now** (≤3)
4. `omg_workflow_manifest_register` / update when using agent packages

## Phase 4 — Approve implementation

```bash
uv run python scripts/roadmap.py approve-impl --issue 96 --summary "docker validation fix"
# or: --manifest-id kaggle-wandb-population
```

Optional strict enforcement for agents:

```bash
export ORBIT_WARS_IMPL_GATE=1
uv run python scripts/roadmap.py gate --request "<same request>" --require-allowed
```

## Phase 5 — Implement

- Run tests per `AGENTS.md` tiers
- `roadmap.py gate` should pass when strict mode is on

## Phase 6 — Done

1. Close GitHub issue with evidence
2. Move row to ROADMAP **Done** (≤5 rows)
3. Manifest → `complete` with evidence
4. `uv run python scripts/roadmap.py clear-impl`
5. `uv run python scripts/roadmap.py validate`

## Hooks

Use `vscode_askQuestions` when intake says `requires_planning` and scope is ambiguous:

- header: `work-intake-planning`
- options: ralplan vs deep-interview vs defer to Later
