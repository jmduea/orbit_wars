# Roadmap

> Human priority index. Details in GitHub issues. Suggested caps: **≤3** **Now**, **≤5** **Done** (not enforced).

**Phase:** submit-valid

## Now

*None — pick from **Next** after planning.*

## Next

| Item | Link |
| ---- | ---- |
| Debug metric: average ships per fleet launch | — |

## Later

| Item | Link |
| ---- | ---- |
| ALL _2p and _4p specific telemetry should default off and be toggled with debug flag | — |
| update time fraction telemetry default off and be toggled with debug flag | — |

## Done (last 5)

| Item | Link |
| ---- | ---- |
| Tier 2: Cursor plugins + minimal agent config (#158) | [#158](https://github.com/jmduea/orbit_wars/issues/158) · archived OMG, `docs/CURSOR.md`, slim `AGENTS.md` |
| PPO health metrics: approx_kl_v2, first/last minibatch KL, parity (#157) | [#157](https://github.com/jmduea/orbit_wars/issues/157) · `ppo_update.py` `metric_registry.py` `make test-fast` |
| src audit phase 1: config required-key audit (#141) | [#141](https://github.com/jmduea/orbit_wars/issues/141) · `conf/` `test_config_consolidation.py` `make test-fast` |
| src audit phase 1: rollout metric contract + telemetry gating (#139–#140) | [#139](https://github.com/jmduea/orbit_wars/issues/139) [#140](https://github.com/jmduea/orbit_wars/issues/140) · `rollout/metrics.py` `metric_registry.py` |
| src audit phase 1: Kaggle P100 default + unified push (#138) | [#138](https://github.com/jmduea/orbit_wars/issues/138) · `src/orchestration/kaggle_*` `make test-fast` |

_Last triaged: 2026-05-31_

## Maintenance

- Update on transition only — start, finish, or abandon work.
- Promote to Next/Now with a linked GitHub issue when useful.
- Agent setup: `docs/CURSOR.md`. Do not use `docs/brain_dump.md` (retired).
