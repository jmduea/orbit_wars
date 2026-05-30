# Roadmap

> Human index only. Details live in linked issues/specs or [brain_dump.md](brain_dump.md).
> Caps: **≤3** in **Now**, **≤5** in **Done**. Inbox: brain_dump · Agent packages: `.omg/workflow-manifest.json`

**Phase:** submit-valid

## Now

| Item | Link |
|------|------|
| Kaggle submission fails docker validation | [#96](https://github.com/jmduea/orbit_wars/issues/96) |
| Kaggle population worker broken (W&B secret, kernel fails on Kaggle) | [#97](https://github.com/jmduea/orbit_wars/issues/97) |

## Next

| Item | Link |
|------|------|
| Verify seed scheduler swaps during training | [#99](https://github.com/jmduea/orbit_wars/issues/99) |
| JAX compile time: within expected bounds? | [#100](https://github.com/jmduea/orbit_wars/issues/100) |
| What is `survival_time` and how does it relate to performance? | [#101](https://github.com/jmduea/orbit_wars/issues/101) |

## Later

| Item | Link |
|------|------|
| W&B tags from Hydra config groups | [brain_dump.md#ideas](brain_dump.md#ideas) |
| Define `num_envs` via training weights instead of format YAML | [brain_dump.md#ideas](brain_dump.md#ideas) |
| Local tournament / ranking eval for best agents | [brain_dump.md#ideas](brain_dump.md#ideas) |
| VRAM profile from W&B run data | [brain_dump.md#ideas](brain_dump.md#ideas) |
| Debug metric: average ships per fleet launch | [brain_dump.md#ideas](brain_dump.md#ideas) |

## Done (last 5)

| Item | Link |
|------|------|
| Config cleanup: presets, legacy models, OMG clutter, Hydra launch recipes | [#98](https://github.com/jmduea/orbit_wars/issues/98) |
| Normalized ship differential terminal reward mode | — |
| ROADMAP system + GitHub issue workflow | — |

_Last triaged: 2026-05-30_

## Maintenance

- **Update on transition only** — when starting, finishing, or abandoning work; skip if nothing moved.
- **Weekly (≤5 min):** triage [brain_dump.md](brain_dump.md) → promote to **Next**/**Now** or delete stale **Later** rows; refresh _Last triaged_.
- **GitHub issues:** create for **Now**/**Next** blockers with labels `type:*` + `area:*` (see [.github/labels.yml](../.github/labels.yml)); replace brain_dump links with `#NNN`.
- **Agents:** run `uv run python scripts/roadmap.py agent` before planning; `validate` after editing this file. Human **Now** wins over manifest backlog for priority.
