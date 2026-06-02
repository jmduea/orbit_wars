# Roadmap

> Human priority index. Details in GitHub issues. Suggested caps: **≤3** **Now**, **≤5** **Done** (not enforced).

**Phase:** submit-valid

## Now

| Item | Link |
| ---- | ---- |
| Planet Flow: sweep shortlist, decoder replay contracts, PPO epoch driver, metric descriptors, compiler-control tests (#166–#170) | [#166](https://github.com/jmduea/orbit_wars/issues/166) … [#170](https://github.com/jmduea/orbit_wars/issues/170) |

## Next

| Item | Link |
| ---- | ---- |
| Debug metric: average ships per fleet launch | — |
| Telemetry: `_2p`/`_4p` and update-time-fraction metrics default off behind debug flag | — |
| Cursor session-start hook (document `make agent-context` in `docs/CURSOR.md`; optional `.cursor/hooks` example) | [Phase 3 plan §4](plans/2026-06-02-agent-native-phase3-refactors.md) |

## Later

| Item | Link |
| ---- | ---- |
| (empty — see Next for near-term backlog) | — |

## Done (last 5)

| Item | Link |
| ---- | ---- |
| Submit-valid operator closure: status inlines `checkpoint_evals`, hybrid profile test | [plan](plans/2026-06-02-007-feat-submit-valid-operator-closure-plan.md) |
| CLI hardening: replay integration test, validate subcommand invariants (#160, #161) | [#160](https://github.com/jmduea/orbit_wars/issues/160) [#161](https://github.com/jmduea/orbit_wars/issues/161) · `30fc7a8` |
| Agent-native Phase 3: gate YAML, benchmark primitives, `ow sweep` (PR #175) | [plan](plans/2026-06-02-agent-native-phase3-refactors.md) · `7078c40` |
| src audit phase 4: checkpoint hooks, telemetry, promotion queue, CatalogView, parametric edge (#151–#155) | [#151](https://github.com/jmduea/orbit_wars/issues/151) … [#155](https://github.com/jmduea/orbit_wars/issues/155) · `jax/train/` `features/catalog/` `make test-fast` |
| src audit phase 3: shield relocation + action_sampling + constants + promotion writer (#147–#150) | [#147](https://github.com/jmduea/orbit_wars/issues/147) … [#150](https://github.com/jmduea/orbit_wars/issues/150) · `jax/shield/` `jax/action_sampling.py` `make test-fast` |

_Last triaged: 2026-06-02_

## Maintenance

- Update on transition only — start, finish, or abandon work.
- Promote to Next/Now with a linked GitHub issue when useful.
- Agent setup: `docs/CURSOR.md`. Do not use `docs/brain_dump.md` (retired).
