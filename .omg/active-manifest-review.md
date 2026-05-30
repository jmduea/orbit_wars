# Active manifest review (2026-05-30)

Eleven entries are **draft**, **planned**, or **executing** in `.omg/workflow-manifest.json`.
Artifacts were restored from pre-clutter-cleanup; **status unchanged** pending your decision.

All **64** manifest-registered spec/plan files were restored so `scripts/omg_workflow_manifest.py validate` passes (complete entries still reference paths).

**How to respond:** Reply with keep / defer / complete / supersede per `id`, or say "apply recommendations".

| id | status | kind | ROADMAP / issue | Suggested disposition | Rationale |
|----|--------|------|-----------------|----------------------|-----------|
| `kaggle-wandb-population` | executing | spec | **Now** [#97](https://github.com/jmduea/orbit_wars/issues/97) | **keep executing** | Matches ROADMAP Now; worker broken on Kaggle |
| `ralplan-kaggle-wandb-population` | executing | plan | **Now** [#97](https://github.com/jmduea/orbit_wars/issues/97) | **keep executing** | Paired plan for above |
| `feature-encoding-trace` | draft | trace | Done (v2 encoding shipped) | **complete** or **superseded** | Trace predates v2-only encoding; no open ROADMAP row |
| `alphazero-mcts-planning` | planned | spec | Later (M3) | **defer** | Notes: "do not start Phase 0 yet" |
| `ralplan-alphazero-mcts-planning` | planned | plan | Later (M3) | **defer** | Paired plan; awaiting execution approval |
| `ppo-gae-gradient-checkpoint` | planned | spec | — | **defer** | Follow-up optimization; not on ROADMAP |
| `ralplan-ppo-gae-gradient-checkpoint` | planned | plan | — | **defer** | Paired plan |
| `policy-advances-audit` | planned | spec | — | **complete** (audit) | Evidence says ralplan iter-1 audit done; implementation is separate milestones |
| `ralplan-policy-advances-audit` | planned | plan | — | **complete** (audit) | Consensus recorded; milestones not registered |
| `zero-launch-metrics-trace` | draft | trace | — | **defer** or **complete** | Shield-off zero-launch bug; verify if still reproduces |
| `ralplan-zero-launch-metrics` | planned | plan | — | **defer** | Fix plan; depends on trace |

## ROADMAP alignment

- **Human Now wins** over manifest (`scripts/roadmap.py agent`).
- Only **kaggle-wandb-population** (+ plan) maps directly to ROADMAP **Now** (#97).
- **#96** (docker validation) has manifest sibling `kaggle-docker-validation` (status **complete**).

## After your review

**Applied 2026-05-30** (user: apply recommended dispositions):

| id | new status |
|----|------------|
| `kaggle-wandb-population` | executing (unchanged) |
| `ralplan-kaggle-wandb-population` | executing (unchanged) |
| `feature-encoding-trace` | superseded |
| `alphazero-mcts-planning`, `ralplan-alphazero-mcts-planning` | deferred |
| `ppo-gae-gradient-checkpoint`, `ralplan-ppo-gae-gradient-checkpoint` | deferred |
| `policy-advances-audit`, `ralplan-policy-advances-audit` | complete |
| `zero-launch-metrics-trace`, `ralplan-zero-launch-metrics` | deferred |

Active manifest count: **2** (Kaggle W&B population only).

```bash
uv run python scripts/omg_workflow_manifest.py validate
uv run python scripts/roadmap.py agent
```
