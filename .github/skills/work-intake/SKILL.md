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

## Phase 4 — Claim (multi-agent)

```bash
export ORBIT_WARS_AGENT_ID=cursor-session-a   # unique per agent
uv run python scripts/roadmap.py claims
uv run python scripts/roadmap.py claim --issue 97 --path src/orchestration/ --path scripts/kaggle_wandb_population.py
```

## Phase 5 — Approve implementation

```bash
uv run python scripts/roadmap.py approve-impl --issue 97 --summary "population worker fix"
```

Optional strict enforcement for agents:

```bash
export ORBIT_WARS_IMPL_GATE=1
uv run python scripts/roadmap.py gate --request "<same request>" --require-allowed
```

## Phase 6 — Implement

- Run tests per `AGENTS.md` tiers
- `roadmap.py gate` should pass when strict mode is on

## Phase 7 — Wrap-up (mandatory)

```bash
gh issue close 97 --comment "Evidence: make test-domain-artifacts; commit abc123; …"
uv run python scripts/roadmap.py wrap-up --issue 97 --evidence "make test-domain-artifacts passed; commit abc123; fixed W&B secret in kernel metadata"
uv run python scripts/roadmap.py check-wrap-up --issue 97 --require-passed
```

`wrap-up` requires:
- GitHub issue **CLOSED** (via `gh`)
- Evidence text ≥40 chars (tests, commit, paths)
- Releases claim and clears impl-gate

## Phase 8 — Done

1. Close GitHub issue with evidence
2. Move row to ROADMAP **Done** (≤5 rows)
3. Manifest → `complete` with evidence
4. `uv run python scripts/roadmap.py check-session --require-clean`
5. `uv run python scripts/roadmap.py validate`

## Hooks

Use `vscode_askQuestions` when intake says `requires_planning` and scope is ambiguous:

- header: `work-intake-planning`
- options: ralplan vs deep-interview vs defer to Later
