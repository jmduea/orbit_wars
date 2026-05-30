# Roadmap

> Single priority index. Details in GitHub issues and `.omg/specs|plans/`.
> Caps: **≤3** **Now**, **≤5** **Done**. **Now** and **Next** may be empty while planning. Agent packages: `.omg/workflow-manifest.json`

**Phase:** submit-valid

## Now

_None — pick from **Next** after planning._

## Next

_None — pick from **Later** after planning._

## Later

| Item | Link |
|------|------|
| Define `num_envs` via training weights instead of format YAML | — |
| VRAM profile from W&B run data | — |
| Debug metric: average ships per fleet launch | — |

## Done (last 5)

| Item | Link |
|------|------|
| Local tournament / ranking eval for best agents | [#124](https://github.com/jmduea/orbit_wars/issues/124) · [#125](https://github.com/jmduea/orbit_wars/pull/125) · `docs/architecture/tournament-eval.md` `make test-domain-artifacts` |
| Define `survival_time` metric and relation to performance (research) | [#101](https://github.com/jmduea/orbit_wars/issues/101) · `docs/benchmarks/issue-101-survival-time.md` |
| Verify seed scheduler swaps during training | [#99](https://github.com/jmduea/orbit_wars/issues/99) · `test_seed_scheduler.py` `test_jax_seed_scheduler.py` |
| JAX compile time vs expected bounds (research) | [#100](https://github.com/jmduea/orbit_wars/issues/100) · `docs/benchmarks/issue-100-jax-compile-time.md` |
| Output hygiene v2: promotion, W&B best artifacts, sweep naming, Hydra tags | [#112](https://github.com/jmduea/orbit_wars/issues/112) [#113](https://github.com/jmduea/orbit_wars/issues/113) [#114](https://github.com/jmduea/orbit_wars/issues/114) [#115](https://github.com/jmduea/orbit_wars/issues/115) [#116](https://github.com/jmduea/orbit_wars/issues/116) [#117](https://github.com/jmduea/orbit_wars/issues/117) · `make test-domain-config` |

_Last triaged: 2026-05-30_

## Agent workflow (mandatory funnel)

Free-form chat is fine — agents run the funnel without slash commands. No implementation in `src/`, `conf/`, or `tests/` until gates pass (Cursor pre-tool hook + `approve-impl`).

| Phase | Action |
|-------|--------|
| **0 — Status** | `uv run python scripts/roadmap.py agent` · `uv run python scripts/omg_workflow_manifest.py active` |
| **1 — Begin** | `uv run python scripts/roadmap.py begin "<user message>"` — intake + gate + `work-session.json` |
| **2 — Planning** | `/deep-interview` → `/ralplan` (or `/omg-autopilot` through spec approval) for non-trivial work |
| **3 — Execution plan** | Chunk order, manifest register, create/update GitHub issues with AC, promote ROADMAP rows |
| **4 — Claim** | `claim --issue N --path … --setup-worktree` · set `ORBIT_WARS_ISSUE_ID=N` and unique `ORBIT_WARS_AGENT_ID` per parallel worker |
| **5 — Approve impl** | `approve-impl --issue N` · implement in `worktrees/issue-N/` on branch `issue/N-slug` (not `main`) |
| **6 — Implement** | Code/tests; `roadmap.py gate --require-allowed` (`ORBIT_WARS_IMPL_GATE` on by default) |
| **7 — ROADMAP Done** | Add **Done** row (≤5 cap), remove from **Now**/**Next**, run `make roadmap-check` |
| **8 — Wrap-up** | `gh issue close N --comment "…"` then `roadmap.py wrap-up --issue N --evidence "tests, commit, …"` (fails if issue not in **Done**) |
| **9 — Session end** | manifest `complete`, `roadmap.py check-session` clean |
| **Multitask** | Parent spawns one executor per issue with claim + worktree + env exports; parent runs `check-session` after workers finish |

**Multi-agent:** `roadmap.py claims` before starting; `ORBIT_WARS_AGENT_ID` distinguishes owners. Session end must pass `check-session` (no open claims without wrap-up). See [OWNERSHIP.md](OWNERSHIP.md).

**New ideas:** add a **Later** row (no issue until phase 3). **Do not** use `docs/brain_dump.md`.

## Maintenance

- **Update on transition only** — start, finish, or abandon work.
- **Promote to Next/Now:** only after planning (phase 2–3); open or link GitHub issue with `type:*` + `area:*`.
- **Validate:** `uv run python scripts/roadmap.py validate` or `make roadmap-check`.
