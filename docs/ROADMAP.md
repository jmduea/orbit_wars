# Roadmap

> Single priority index. Details in GitHub issues and `.omg/specs|plans/`.
> Caps: **≤3** **Now**, **≤5** **Done**. **Now** and **Next** may be empty while planning. Agent packages: `.omg/workflow-manifest.json`

**Phase:** submit-valid

## Now

_None — pick from **Next** after planning._

## Next

| Item | Link |
|------|------|
| Debug metric: average ships per fleet launch | — |

## Later

| Item | Link |
|------|------|
| _None_ | — |

## Done (last 5)

| Item | Link |
|------|------|
| Git landing: merge worktree to main, block issue/* push | [#135](https://github.com/jmduea/orbit_wars/issues/135) · `land-issue` `docs/MULTI_AGENT.md` |
| Multi-agent coordination (per-issue impl-gates, stale claims, playbook) | [#134](https://github.com/jmduea/orbit_wars/issues/134) · `docs/MULTI_AGENT.md` `make test-fast` |
| Telemetry cleanup phase 2: split logs, registry prune (#133) | [#133](https://github.com/jmduea/orbit_wars/issues/133) · `src/jax/train.py` `make test-fast` |
| Telemetry: gate heavy/sparse fields + PPO debug group (#131–#132) | [#131](https://github.com/jmduea/orbit_wars/issues/131) [#132](https://github.com/jmduea/orbit_wars/issues/132) · `metric_registry.py` |
| Telemetry: metric_groups runtime filtering (#130) | [#130](https://github.com/jmduea/orbit_wars/issues/130) · `filter_update_record` `make test-fast` |

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
| **Multitask** | Parent: `claims --stale`, `release-stale --apply`, spawn one executor per issue; `check-session --global` after workers |

**Multi-agent:** `roadmap.py claims` before starting; `ORBIT_WARS_AGENT_ID` distinguishes owners. Session end must pass `check-session` (no open claims without wrap-up). See [OWNERSHIP.md](OWNERSHIP.md).

**New ideas:** add a **Later** row (no issue until phase 3). **Do not** use `docs/brain_dump.md`.

## Maintenance

- **Update on transition only** — start, finish, or abandon work.
- **Promote to Next/Now:** only after planning (phase 2–3); open or link GitHub issue with `type:*` + `area:*`.
- **Validate:** `uv run python scripts/roadmap.py validate` or `make roadmap-check`.
