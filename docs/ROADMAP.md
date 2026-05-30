# Roadmap

> Single priority index. Details in GitHub issues and `.omg/specs|plans/`.
> Caps: **≤3** **Now**, **≤5** **Done**. Agent packages: `.omg/workflow-manifest.json`

**Phase:** submit-valid

## Now

| Item | Link |
|------|------|
| Kaggle submission fails docker validation | [#96](https://github.com/jmduea/orbit_wars/issues/96) |

## Next

| Item | Link |
|------|------|
| Verify seed scheduler swaps during training | [#99](https://github.com/jmduea/orbit_wars/issues/99) |
| JAX compile time: within expected bounds? | [#100](https://github.com/jmduea/orbit_wars/issues/100) |
| What is `survival_time` and how does it relate to performance? | [#101](https://github.com/jmduea/orbit_wars/issues/101) |

## Later

| Item | Link |
|------|------|
| Unify outputs: promotion, W&B best artifacts, sweep naming | [output-storage plan](../.omg/plans/output-storage-roadmap.md) · [output-standardization spec](../.omg/specs/deep-interview-output-standardization.md) |
| W&B tags from Hydra config groups | — |
| Define `num_envs` via training weights instead of format YAML | — |
| Local tournament / ranking eval for best agents | — |
| VRAM profile from W&B run data | — |
| Debug metric: average ships per fleet launch | — |

## Done (last 5)

| Item | Link |
|------|------|
| Kaggle population worker (`ow train kaggle` standalone; P100 smoke) | [#97](https://github.com/jmduea/orbit_wars/issues/97) · `850568a` `e98f452` `20c8011` |
| Config cleanup: presets, legacy models, OMG clutter, Hydra launch recipes | [#98](https://github.com/jmduea/orbit_wars/issues/98) |
| Normalized ship differential terminal reward mode | — |
| ROADMAP system + GitHub issue workflow | — |

_Last triaged: 2026-05-30_

## Agent workflow (mandatory funnel)

Free-form chat is fine — agents run the funnel without slash commands. No implementation in `src/`, `conf/`, or `tests/` until gates pass (Cursor pre-tool hook + `approve-impl`).

| Phase | Action |
|-------|--------|
| **0 — Status** | `uv run python scripts/roadmap.py agent` · `uv run python scripts/omg_workflow_manifest.py active` |
| **1 — Begin** | `uv run python scripts/roadmap.py begin "<user message>"` — intake + gate + `work-session.json` |
| **2 — Planning** | `/deep-interview` → `/ralplan` (or `/omg-autopilot` through spec approval) for non-trivial work |
| **3 — Execution plan** | Chunk order, manifest register, create/update GitHub issues with AC, promote ROADMAP rows |
| **4 — Claim** | `roadmap.py claim --issue N --path src/...` (one claim per issue; no path overlap) |
| **5 — Approve impl** | `roadmap.py approve-impl --issue N` · branch `issue/N-short-slug` |
| **6 — Implement** | Code/tests; `roadmap.py gate --require-allowed` (`ORBIT_WARS_IMPL_GATE` on by default) |
| **7 — Wrap-up** | `gh issue close N --comment "…"` then `roadmap.py wrap-up --issue N --evidence "tests, commit, …"` |
| **8 — Done** | ROADMAP **Done**, manifest `complete`, `roadmap.py check-session` clean |

**Multi-agent:** `roadmap.py claims` before starting; `ORBIT_WARS_AGENT_ID` distinguishes owners. Session end must pass `check-session` (no open claims without wrap-up). See [OWNERSHIP.md](OWNERSHIP.md).

**New ideas:** add a **Later** row (no issue until phase 3). **Do not** use `docs/brain_dump.md`.

## Maintenance

- **Update on transition only** — start, finish, or abandon work.
- **Promote to Next/Now:** only after planning (phase 2–3); open or link GitHub issue with `type:*` + `area:*`.
- **Validate:** `uv run python scripts/roadmap.py validate` or `make roadmap-check`.
