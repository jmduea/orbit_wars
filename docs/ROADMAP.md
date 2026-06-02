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
| Launch hygiene tier-2 e2e gate pass on baseline GPU | `make test-launch-hygiene-e2e-throughput` exit 0 on RTX 5080 / same machine as baseline capture | [runbook](operator-runbook.md) · [plan](plans/2026-06-01-launch-hygiene-e2e-throughput-plan.md) |
| Launch hygiene Phase B (conditional) | Only if tier-2 fails: hot-path recovery per U7; re-gate until pass; tier-1 still green | [plan U7](plans/2026-06-01-launch-hygiene-e2e-throughput-plan.md#u7-conditional-phase-b--profile-driven-hot-path-recovery) |
| Preflight learn-proof refresh (post-hygiene) | `make preflight-learn-proof` through `beat_random` vs `preflight-calibration.json` after profile/calibration refresh | [runbook](operator-runbook.md) · [plan](plans/2026-06-02-005-feat-preflight-training-profiles-plan.md) |

## Done (last 5)

| Item | Link |
| ---- | ---- |
| Cursor session-start hook: `docs/CURSOR.md` + project `.cursor/hooks.json` example | [plan](plans/2026-06-02-011-feat-cursor-session-start-hook-plan.md) · [Phase 3 §4](plans/2026-06-02-agent-native-phase3-refactors.md) |
| Observability debug bundle: `mean_ships_per_launch`, PPO `_2p`/`_4p` + update-time fractions gated behind `metric_groups.debug` | [plan](plans/2026-06-02-010-feat-observability-debug-metrics-plan.md) |
| Split decoder replay batch contracts (#167) | [plan](plans/2026-06-02-009-feat-split-decoder-replay-batch-contracts-plan.md) · Closes #167 |
| Planet Flow queue residuals: shortlist verify, metric descriptors, PPO epoch driver, compiler-control tests (#166, #168–#170) | [plan](plans/2026-06-02-008-feat-planet-flow-queue-residuals-plan.md) · Closes #166, #168, #169, #170 |
| Submit-valid operator closure: status inlines `checkpoint_evals`, hybrid profile test | [plan](plans/2026-06-02-007-feat-submit-valid-operator-closure-plan.md) |

_Last triaged: 2026-06-02_

## Maintenance

- Update on transition only — start, finish, or abandon work.
- Promote to Next/Now with a linked GitHub issue when useful.
- Agent setup: `docs/CURSOR.md`. Do not use `docs/brain_dump.md` (retired).
