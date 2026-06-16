---
title: JAX comet Kaggle parity and CI gate
date: 2026-06-03
last_updated: 2026-06-06
category: architecture-patterns
module: jax-env-parity
problem_type: architecture_pattern
component: testing_framework
severity: high
applies_when:
  - "Porting Kaggle orbit_wars.py comet mechanics (spawn steps, movement order, collision rules) into src/jax/env.py"
  - "Closing JAX/Kaggle env parity gaps after planet generation and combat parity"
  - "Adding or changing CI so env parity regressions fail on pull_request and push to main"
  - "Re-applying pick #4 greenfield pure JAX comet generation on orbit_wars-integration"
tags:
  - jax-env
  - kaggle-parity
  - comet-subsystem
  - mechanical-parity
  - compile-time-gate
  - test-kaggle-parity
  - ci-enforcement
related_components:
  - src/game/comet_generation.py
  - src/jax/comet_generation.py
  - src/jax/env.py
  - tests/test_jax_env_parity.py
  - tests/test_comet_generation.py
  - .github/workflows/kaggle-jax-parity.yml
---

# JAX comet Kaggle parity and CI gate

## Context

After planet generation and rotation parity, the JAX training env still reserved `TOTAL_COMETS` planet slots but never spawned or moved comets. Kaggle adds neutral capture targets at fixed spawn steps with elliptical paths, 4-fold symmetry, per-spawn RNG, first-tick collision skip, expiry, and `initial_planets` sync. Training rollouts after step 50 therefore diverged from tournament physics even when other parity tests passed.

Parity was verifiable locally via `make test-kaggle-parity`, but only a label-sync workflow ran on GitHub until PR [#188](https://github.com/jmduea/orbit_wars/pull/188) shipped the comet subsystem and `.github/workflows/kaggle-jax-parity.yml` together (plans `docs/solutions/architecture-patterns/jax-comet-kaggle-parity-ci-gate.md`, `docs/solutions/architecture-patterns/jax-comet-kaggle-parity-ci-gate.md`).

**Forward path (2026-06-06):** comet spawn belongs in **pure JAX** (`src/jax/comet_generation.py`), not `jax.pure_callback` + `src/game/comet_generation.py` in the hot path. See [jax-no-kaggle-callbacks.md](../conventions/jax-no-kaggle-callbacks.md). Phase 2 targets **mechanical fidelity** (valid rules-compliant states), not bit-exact seed replay against `kaggle_environments` — maps and comet paths may differ per seed.

**Worktree status:** `main` still uses `pure_callback` comet spawn (PR #188). Pick #4 greenfield on `orbit_wars-integration` was rolled back to **`9db50f5`** (picks #1–2 + 3b only) after a mechanical fix caused **10m+ compile** — see [phase2-pick4-jax-compile-rollback-criteria.md](../workflow-issues/phase2-pick4-jax-compile-rollback-criteria.md).

## Guidance

### Reference module (test-only) + fixed-shape JAX state

`src/game/comet_generation.py` (`generate_comet_paths`) is the **reference** for rules and test goldens — import from **test code only**, never from `reset` / `step` / rollout `jit` paths.

On `JaxGameState`, keep `JaxCometState` with fixed arrays for groups, paths, and activity masks so `jit` / `scan` / `vmap` stay valid (`src/jax/env.py`).

Spawn schedule and caps live in `src/game/constants.py`:

- `COMET_SPAWN_STEPS = [50, 150, 250, 350, 450]`
- `COMETS_PER_GROUP = 4`, `MAX_COMET_GROUPS = len(COMET_SPAWN_STEPS)`, `MAX_COMET_PATH_LEN = 40`
- `COMET_OFF_BOARD = -99.0` for pre-board placeholders

### Spawn via pure JAX (not `pure_callback`, not inlined step subgraph)

**Forward implementation:** `src/jax/comet_generation.py` — `generate_comet_paths`, `comet_rng_key`, `CometPathResult`. No `jax.pure_callback`, no `_reference_comet_paths`, no `src/game/*` imports in env hot path ([jax-no-kaggle-callbacks.md](../conventions/jax-no-kaggle-callbacks.md)).

Preferred patterns (pick #4 re-attempt):

1. **Precompute at `reset`** — `precompute_comet_schedule_at_reset` generates all waves once; `step` only activates precomputed groups on spawn steps (`lax.cond` over `COMET_SPAWN_STEPS`). Keeps per-step trace smaller than inline spawn search.
2. **Isolated `@jax.jit` spawn** — if spawn must run on a boundary, hoist to a dedicated jit with bounded static args; do not inline a large search subgraph into the main env `step` trace.

**Mechanical parity invariants** (Phase 2 scope — not coordinate goldens):

- `planets.active == initial_planets.active` through comet spawn and post-move expire (step 50+, step 200+)
- Comet IDs from reserved tail slots (not `max(active_id)+1`)
- Four-planet symmetry per group; paths on-board with valid bounds
- First board entry skips fleet collision (`check_collision` gated when `is_comet` and `old_px < 0`)

**Out of scope for Phase 2 proof:** bit-exact `Random(f"orbit_wars-comet-{episode_seed}-{step}")` coordinate match vs Kaggle; home-group PRNG stream coupling.

**Legacy on `main` (do not extend):** PR #188 spawn via `jax.pure_callback` + `_reference_comet_paths` — acceptable only until pure JAX ports land; new work must not add callbacks.

### Compile-time gate (independent from parity pytest)

Green `make test-kaggle-parity` does **not** bound training-loop compile. Before admitting pick #4 on integration, measure first-compile on a representative smoke path:

```bash
cd /home/jmduea/projects/orbit_wars
uv run ow benchmark training --preset primary --updates 3 --warmup 1 --label pick4-compile-check
```

Reject and roll back integration HEAD when compile exceeds operator tolerance (session: **10m+** after `0eb349e`). Record `compile_time_regression` in manifest — see [phase2-pick4-jax-compile-rollback-criteria.md](../workflow-issues/phase2-pick4-jax-compile-rollback-criteria.md).

### Movement order and collision rules

Per tick, keep Kaggle order: **expire comets → spawn → production → advance paths → fleet sweep**. Comets skip orbiting; on first board entry skip fleet collision. Update `initial_planets` in lockstep on spawn and expiry so rotation and feature encoding stay consistent.

### Tests and Makefile target

| Guard | Location |
|-------|----------|
| Path geometry golden (reference) | `tests/test_comet_generation.py` — e.g. `Random("orbit_wars-comet-0-50")`, 4 symmetric paths against `src/game/comet_generation.py` |
| End-to-end spawn sync | `tests/test_jax_env_parity.py` — 50 noop steps, seed 0, `group_count >= 1`, `planets.active == initial_planets.active` |
| Post-move expire sync | `tests/test_jax_env_parity.py` — expire through step 90+ keeps `initial_planets` aligned |
| Synthetic game states | Use `empty_comet_state()` in parity, golden, and shield fixtures |

```bash
make test-kaggle-parity
```

Runs `tests/test_jax_env_parity.py`, `tests/test_jax_env.py`, `tests/test_jax_env_dispatch.py` with `-m "jax and not slow"` (CPU JAX). Also: `make test-jax-trace-hygiene` (tier-A `rg` — no callbacks in `src/jax/` hot path).

### CI workflow

`.github/workflows/kaggle-jax-parity.yml` triggers on `pull_request` and `push` to `main` / `master`, runs `uv sync --group dev` then `make test-kaggle-parity` with `JAX_PLATFORMS: cpu`. Pass/fail is pytest exit code only — no invented thresholds.

## Why This Matters

Comets change mid-game capture and planet counts; wrong physics after step 50 poisons training metrics and tournament replay. A narrow CI job catches regressions in spawn timing, collision rules, and `initial_planets` sync without running full `test-premerge` or GPU tiers.

Callback-based spawn blocks trace hygiene and vmap rollouts; inlined pure-JAX spawn without compile gating can pass parity while blocking operator smoke for 10+ minutes. This slice is **env stepping parity** — distinct from the SSOT submit-valid spine (`docs/solutions/architecture-patterns/ssot-training-pipeline-config-to-kaggle-submission.md`) and legacy Gate 5 / bracket docs.

## When to Apply

- Any change to `src/jax/env.py`, `src/jax/comet_generation.py`, `src/game/comet_generation.py`, or combat/movement ordering
- Before trusting training or eval metrics in comet-heavy episodes (after step 50)
- When re-applying pick #4 on `orbit_wars-integration` — parity + trace + **compile** gates, not parity alone
- When adding new env parity tests — keep them under `-m "jax and not slow"` so `make test-kaggle-parity` and CI pick them up

## Examples

**Wrong → right**

- Wrong: reserve comet planet slots but never spawn — parity gap remains after planets/rotation fixes.
- Wrong: `jax.pure_callback` + `_reference_comet_paths` in env `step` / `reset` hot path — violates trace hygiene; legacy on `main` only.
- Wrong: inline full comet path search in main env `step` without compile check — can pass parity while causing 10m+ first-compile.
- Wrong: assert JAX comet coordinates == Kaggle `kaggle_environments` for fixed seeds — out of scope for Phase 2 mechanical parity.
- Right: pure JAX `src/jax/comet_generation.py`; reference `src/game/comet_generation.py` in tests only; spawn on `(step + 1) ∈ COMET_SPAWN_STEPS` with mechanical validity invariants.
- Right: precompute comet waves at `reset`, activate on spawn steps; measure compile on short benchmark before manifest admit.

**Targeted parity test (spawn sync guard):**

```bash
uv run pytest tests/test_jax_env_parity.py::test_comet_spawn_keeps_initial_planets_synced_after_forty_nine_steps -m "jax and not slow"
```

Same file set as CI also runs via `make test-domain-jax-env` (alias for the parity Makefile target).

## Related

- **JAX-only hot path convention:** [jax-no-kaggle-callbacks.md](../conventions/jax-no-kaggle-callbacks.md)
- **Pick #4 rollback + compile gate:** [phase2-pick4-jax-compile-rollback-criteria.md](../workflow-issues/phase2-pick4-jax-compile-rollback-criteria.md)
- **Phase 2 cherry-pick admission:** [phase2-env-parity-cherry-pick-integration-admission.md](../workflow-issues/phase2-env-parity-cherry-pick-integration-admission.md)
- Plans (implementation history): `docs/solutions/architecture-patterns/jax-comet-kaggle-parity-ci-gate.md`, `docs/solutions/architecture-patterns/jax-comet-kaggle-parity-ci-gate.md`, `docs/solutions/workflow-issues/phase2-pick4-jax-compile-rollback-criteria.md`
- PR [#188](https://github.com/jmduea/orbit_wars/pull/188) — squash merge on `main` (callback-era comet subsystem)
- **Canonical training spine (SSOT):** `docs/solutions/architecture-patterns/ssot-training-pipeline-config-to-kaggle-submission.md`
- Env invariant note (id↔row): `docs/solutions/logic-errors/planet-flow-catalog-reachability-mismatch.md` — comet spawn/expire should preserve the same invariant; refresh that doc if comet slot layout changes
- Operator pointers: `AGENTS.md` (Kaggle reference path, `make test-kaggle-parity`), `docs/ONBOARDING.md`
