# Session handoff — Phase 2 env-parity rollback (continued)

**Date:** 2026-06-06  
**Audience:** Fresh agent continuing Phase 2 env-parity onto the integration worktree after pick #4 rollback.  
**Supersedes:** [2026-06-06-phase2-env-parity-picks-continued.md](./2026-06-06-phase2-env-parity-picks-continued.md) for *current* integration HEAD and pick #4 state.

---

## Start here

Integration worktree **`/home/jmduea/projects/orbit_wars-integration`** is at **`9db50f59f6f0d42d74b37cb1dbee373fc3ed6827`** (`9db50f5`).

### What landed

| Pick | SHA / label | Verdict | Scope |
| --- | --- | --- | --- |
| #1 | (in `3f88f1b` stack) | ADMIT | Game reference libs: `src/game/comet_generation.py`, `planet_generation.py`, `constants.py` |
| #2 | (in `3f88f1b` stack) | ADMIT | `encode_learner_turn` / learner feature path in `src/jax/features.py` + env call sites |
| 3b | `9db50f5` | ADMIT | Cheap mechanics hunks C–F: rotation index, `ship_speed`, first-hit combat, planet_id launch guard |
| #3 full | — | REJECT | Sequential `lax.scan` launch (~18× throughput) — do not replay |

Admission **VERIFIED** on picks #1–2 @ `3f88f1b` (locked recipe). Not re-run after 3b or pick #4 attempts.

### What rolled back and why

| Commit | Label | Fate | Reason |
| --- | --- | --- | --- |
| `75a7cf2` | Pick #4 greenfield pure JAX planet/comet | Removed from worktree | Validity defects found in review; superseded by fix attempt |
| `0eb349e` | Mechanical fidelity fix for pick #4 | **REJECTED** | `compile_time_regression` — **10m+ JAX compile** on smoke/light benchmark |

**Rollback (2026-06-06):** `git reset --hard 9db50f5` on integration. Pick #4 is **not** on the worktree until re-applied with acceptable compile cost.

**Read first:**

1. [phase2-pick4-jax-compile-rollback-criteria.md](../solutions/workflow-issues/phase2-pick4-jax-compile-rollback-criteria.md) — **ce-compound doc**: compile cliff, rollback criteria, mechanical parity framing
2. [phase2-env-parity-cherry-pick-integration-admission.md](../solutions/workflow-issues/phase2-env-parity-cherry-pick-integration-admission.md) — picks #1–3b, admission recipe, throughput reject
3. [cherry-pick-manifest.json](../benchmarks/cherry-pick-manifest.json) — `integration_state`, `pick_4_attempt_2026_06_06`, `candidates[]`
4. [2026-06-06-001-fix-pick4-jax-parity-plan.md](../plans/2026-06-06-001-fix-pick4-jax-parity-plan.md) — mechanical fidelity plan (status completed; fix rolled back)
5. [jax-no-kaggle-callbacks.md](../solutions/conventions/jax-no-kaggle-callbacks.md) — hard JAX-only constraint

---

## Hard constraints (do not violate)

1. **JAX-only hot path** — no `pure_callback`, `io_callback`, `_reference_*`, or `src/game/*` imports in `reset` / `step` / rollout / `jit` paths.
2. **No sequential `lax.scan` in `_launch_fleets`** — pick #3 full bundle rejected (~18× throughput).
3. **Mechanical parity goal** — valid rules-compliant JAX states; **not** bit-exact seed replay vs `kaggle_environments`. Maps, comet paths, comet RNG strings, home-group stream may differ per seed.
4. **Compile-time gate** — parity + trace green is insufficient if smoke/benchmark first-compile exceeds operator tolerance (session threshold: ~10m+ = fail).
5. **Operator runs admission** — agents run fast gates per pick; `make gate-admission` is milestone-only unless explicitly requested.

---

## Open decisions

| Decision | Options | Notes |
| --- | --- | --- |
| Pick #4 re-attempt | (A) Re-apply `75a7cf2` then fix defects incrementally with compile check per hunk; (B) Precompute planet/comet at `reset` only; (C) Deferred spawn `jit` with compile proof on 32-env vmap | `0eb349e` KTD7 (`_jit_spawn_comet_group` + `lax.cond`) passed parity but failed compile gate |
| Pick #5 | Deferred until pick #4 compile path acceptable | Remaining mechanics hunks; exclude pick #3 hunks A–B |
| Pick #6 | Pending on `main` | Callback teardown for tier-A static gate |
| Admission re-run | Operator discretion | Not required after rollback; re-run when parity stack + compile are green |

---

## Worktrees

| Role | Path | Branch | HEAD |
| --- | --- | --- | --- |
| Gate harness | `/home/jmduea/projects/orbit_wars` | `main` | Manifest, admission YAML, docs |
| Phase 1 anchor | `/home/jmduea/projects/orbit_wars-throughput-anchor` | `throughput-baseline` | Admission passed |
| **Phase 2 integration** | `/home/jmduea/projects/orbit_wars-integration` | `throughput-baseline-integration` | **`9db50f5`** |
| Pre-hygiene reference | `/home/jmduea/projects/orbit_wars-pre-hygiene` | detached @ `79162a2` | Baseline SHA only |

Integration branch base: anchor HEAD `52dfdb0` (seven Phase-1 commits past manifest `baseline_sha`).

---

## Reject criteria (manifest)

- `parity_fail` — `make test-kaggle-parity` red
- `trace_hygiene_fail` — tier-A `rg` or `make test-jax-trace-hygiene` fails
- `throughput_regression` — admission outside learning-first baseline band
- **`compile_time_regression`** — smoke/benchmark first-compile unacceptable (new for pick #4)
- `wrong_pick_granularity` — whole commit when split required
- JAX-only violation — reject under `user_constraint`

---

## First commands

```bash
cd /home/jmduea/projects/orbit_wars
make agent-context
git worktree list

# Integration head and history
cd /home/jmduea/projects/orbit_wars-integration
git log -5 --oneline
git rev-parse HEAD   # expect 9db50f5

# Fast gates (should be green at rollback HEAD)
make test-kaggle-parity

cd /home/jmduea/projects/orbit_wars
make test-jax-trace-hygiene

# Manifest path
cat docs/benchmarks/cherry-pick-manifest.json | jq '.integration_state, .pick_4_attempt_2026_06_06'

# Admission dry-run (operator runs full gate)
uv run ow benchmark gate run admission --dry-run --verbose \
  --repo-root /home/jmduea/projects/orbit_wars-integration \
  --output-root /home/jmduea/projects/orbit_wars-integration/outputs
```

---

## Handoff prompt (copy-paste for fresh agent)

```
You are continuing Orbit Wars Phase 2 env-parity on the integration worktree.

Current state:
- Integration path: /home/jmduea/projects/orbit_wars-integration
- Branch: throughput-baseline-integration
- HEAD: 9db50f5 (picks #1, #2, 3b only)
- Pick #4 greenfield (75a7cf2) and mechanical fix (0eb349e) were ROLLED BACK — compile_time_regression (10m+ JAX compile on smoke/benchmark)

Read first:
- docs/session-handoff/2026-06-06-phase2-env-parity-rollback-continued.md
- docs/solutions/workflow-issues/phase2-pick4-jax-compile-rollback-criteria.md
- docs/benchmarks/cherry-pick-manifest.json
- docs/plans/2026-06-06-001-fix-pick4-jax-parity-plan.md
- docs/solutions/conventions/jax-no-kaggle-callbacks.md

Hard constraints:
- JAX-only hot path (no pure_callback, no src/game imports in env/rollout hot path)
- No sequential lax.scan fleet launch (pick #3 rejected ~18x throughput)
- Mechanical parity: valid rules-compliant states, NOT bit-exact seed replay vs kaggle_environments
- Compile-time gate: parity green is not enough if smoke/benchmark compile exceeds operator tolerance

Your task: Re-attempt pick #4 (pure JAX planet/comet generation) on integration with compile-bounded approach. Options to evaluate: precompute at reset, incremental hunks with per-hunk compile check, deferred spawn jit with proof on production vmap geometry. Do NOT replay pick #3 full bundle.

Per-pick gates: make test-kaggle-parity (integration cwd) + make test-jax-trace-hygiene (main harness). Update manifest candidates[] after each admit/reject. Operator runs make gate-admission — do not run unless asked.

Do not commit unless explicitly requested.
```

---

## Escalation

- GPU contention: `make agent-context` / check terminals before long trains
- Do not pipe long `ow` commands through `tail`/`head` — use `--out`, `ow runs watch`, or `tail -f` on run logs
- If admission fails but parity is green: check recipe geometry (`admission.yaml` train_overrides) before reverting picks
- Human merges integration → `main`; agents update manifest and propose `ordered_shas` only

---

## Related handoffs

- Prior pick state: [2026-06-06-phase2-env-parity-picks-continued.md](./2026-06-06-phase2-env-parity-picks-continued.md)
- Phase 2 kickoff: [2026-06-06-phase2-env-parity-picks.md](./2026-06-06-phase2-env-parity-picks.md)
- Manifest program: [2026-06-05-cherry-pick-manifest.md](./2026-06-05-cherry-pick-manifest.md)
