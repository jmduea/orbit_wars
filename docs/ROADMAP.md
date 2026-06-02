# Roadmap

> Human priority index. Details in GitHub issues. Suggested caps: **≤3** **Now**, **≤5** **Done** (not enforced).

**Phase:** submit-valid

## Now

| Item | Link |
| ---- | ---- |
| (empty — see Done for latest observability work) | — |

## Next

| Item | Link |
| ---- | ---- |
| Cursor session-start hook (document `make agent-context` in `docs/CURSOR.md`; optional `.cursor/hooks` example) | [Phase 3 plan §4](plans/2026-06-02-agent-native-phase3-refactors.md) |

## Later

| Item | Link |
| ---- | ---- |
| (empty — see Next for near-term backlog) | — |

## Done (last 5)

| Item | Link |
| ---- | ---- |
| Observability debug bundle: `mean_ships_per_launch`, PPO `_2p`/`_4p` + update-time fractions gated behind `metric_groups.debug` | [plan](plans/2026-06-02-010-feat-observability-debug-metrics-plan.md) |
| Split decoder replay batch contracts (#167) | [plan](plans/2026-06-02-009-feat-split-decoder-replay-batch-contracts-plan.md) · Closes #167 |
| Planet Flow queue residuals: shortlist verify, metric descriptors, PPO epoch driver, compiler-control tests (#166, #168–#170) | [plan](plans/2026-06-02-008-feat-planet-flow-queue-residuals-plan.md) · Closes #166, #168, #169, #170 |
| Submit-valid operator closure: status inlines `checkpoint_evals`, hybrid profile test | [plan](plans/2026-06-02-007-feat-submit-valid-operator-closure-plan.md) |
| CLI hardening: replay integration test, validate subcommand invariants (#160, #161) | [#160](https://github.com/jmduea/orbit_wars/issues/160) [#161](https://github.com/jmduea/orbit_wars/issues/161) · `30fc7a8` |
| Agent-native Phase 3: gate YAML, benchmark primitives, `ow sweep` (PR #175) | [plan](plans/2026-06-02-agent-native-phase3-refactors.md) · `7078c40` |

_Last triaged: 2026-06-02_

## Maintenance

- Update on transition only — start, finish, or abandon work.
- Promote to Next/Now with a linked GitHub issue when useful.
- Agent setup: `docs/CURSOR.md`. Do not use `docs/brain_dump.md` (retired).
