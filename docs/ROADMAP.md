# Roadmap

> Human priority index. Details in GitHub issues. Suggested caps: **≤3** **Now**, **≤5** **Done** (not enforced).

**Phase:** submit-valid

## Now

| Item | Link |
| ---- | ---- |
| (empty — see Done for latest agent onboarding work) | — |

## Next

| Item | Link |
| ---- | ---- |
| (empty — see Done for latest agent onboarding work) | — |

## Later

| Item | Acceptance | Link |
| ---- | ------------ | ---- |
| Launch hygiene tier-2 throughput recovery (optional) | Only if a new rollout sampling design lands: re-gate vs baseline; tier-1 still green | [rollout design](plans/2026-06-01-launch-hygiene-rollout-throughput-design.md) · [ablation](benchmarks/launch-hygiene-ablation.json) |

## Done (last 5)

| Item | Link |
| ---- | ---- |
| Launch hygiene learner ablation + tier-2 assessment | Tier-2 failed (~4× sec/update); hot path exhausted; B (hygiene) wins learn-proof vs A (79162a); artifact [ablation.json](benchmarks/launch-hygiene-ablation.json) | [plan](plans/2026-06-02-013-feat-launch-hygiene-tier2-preflight-plan.md) · [runbook](operator-runbook.md) |
| Preflight learn-proof refresh (post-hygiene) | VERIFIED through `beat_random` vs `preflight-calibration.json` | [report](../outputs/preflight/factorized_post_hygiene_learn_proof.json) · [runbook](operator-runbook.md) |
| Cursor session-start hook: `docs/CURSOR.md` + project `.cursor/hooks.json` example | [plan](plans/2026-06-02-011-feat-cursor-session-start-hook-plan.md) · [Phase 3 §4](plans/2026-06-02-agent-native-phase3-refactors.md) |
| Observability debug bundle: `mean_ships_per_launch`, PPO `_2p`/`_4p` + update-time fractions gated behind `metric_groups.debug` | [plan](plans/2026-06-02-010-feat-observability-debug-metrics-plan.md) |
| Split decoder replay batch contracts (#167) | [plan](plans/2026-06-02-009-feat-split-decoder-replay-batch-contracts-plan.md) · Closes #167 |

_Last triaged: 2026-06-02_

## Maintenance

- Update on transition only — start, finish, or abandon work.
- Promote to Next/Now with a linked GitHub issue when useful.
- Agent setup: `docs/CURSOR.md`. Do not use `docs/brain_dump.md` (retired).
