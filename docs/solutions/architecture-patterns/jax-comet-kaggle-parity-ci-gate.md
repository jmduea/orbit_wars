---
title: JAX comet Kaggle parity and CI gate
date: 2026-06-03
category: architecture-patterns
module: jax-env-parity
problem_type: architecture_pattern
component: testing_framework
severity: high
applies_when:
  - "Porting Kaggle orbit_wars.py mechanics (comets, spawn steps, movement order) into src/jax/env.py"
  - "Closing remaining JAX/Kaggle env parity gaps after planet generation and combat parity"
  - "Adding or changing CI so env parity regressions fail on pull_request and push to main"
tags:
  - jax-env
  - kaggle-parity
  - comet-subsystem
  - pure-callback
  - test-kaggle-parity
  - ci-enforcement
related_components:
  - src/game/comet_generation.py
  - src/jax/env.py
  - tests/test_jax_env_parity.py
  - tests/test_comet_generation.py
  - .github/workflows/kaggle-jax-parity.yml
---

# JAX comet Kaggle parity and CI gate

## Context

After planet generation and rotation parity, the JAX training env still reserved `TOTAL_COMETS` planet slots but never spawned or moved comets. Kaggle adds neutral capture targets at fixed spawn steps with elliptical paths, 4-fold symmetry, per-spawn RNG, first-tick collision skip, expiry, and `initial_planets` sync. Training rollouts after step 50 therefore diverged from tournament physics even when other parity tests passed.

Parity was verifiable locally via `make test-kaggle-parity`, but only a label-sync workflow ran on GitHub until PR [#188](https://github.com/jmduea/orbit_wars/pull/188) shipped the comet subsystem and `.github/workflows/kaggle-jax-parity.yml` together (plans `docs/plans/2026-06-03-008-feat-jax-comet-subsystem-plan.md`, `docs/plans/2026-06-03-009-feat-ci-kaggle-jax-parity-plan.md`). Verification: 16 parity tests locally; CI job `kaggle-parity` SUCCESS on `main`.

## Guidance

### Reference module + fixed-shape JAX state

Port Kaggle path generation to `src/game/comet_generation.py` (`generate_comet_paths`). On `JaxGameState`, add `JaxCometState` with fixed arrays for groups, paths, and activity masks so `jit` / `scan` / `vmap` stay valid (`src/jax/env.py`).

Spawn schedule and caps live in `src/game/constants.py`:

- `COMET_SPAWN_STEPS = [50, 150, 250, 350, 450]`
- `COMETS_PER_GROUP = 4`, `MAX_COMET_GROUPS = len(COMET_SPAWN_STEPS)`, `MAX_COMET_PATH_LEN = 40`
- `COMET_OFF_BOARD = -99.0` for pre-board placeholders

### Spawn via `pure_callback` (not JIT-native search)

At spawn boundary when `(step + 1)` is in `COMET_SPAWN_STEPS`, call Python reference paths through `jax.pure_callback`, matching the existing `planet_generation` pattern:

```python
rng = random.Random(f"orbit_wars-comet-{episode_seed}-{step}")
paths = generate_comet_paths(rows, angular_velocity, step, excluded, comet_speed, rng=rng)
```

`episode_seed` comes from `reset()`; the RNG string must stay aligned with Kaggle per-spawn seeding.

### Movement order and collision rules

Per tick, keep Kaggle order: **expire comets → spawn → production → advance paths → fleet sweep**. Comets skip orbiting; on first board entry skip fleet collision (`check_collision` gated when `is_comet` and `old_px < 0`). Update `initial_planets` in lockstep on spawn and expiry so rotation and feature encoding stay consistent.

### Tests and Makefile target

| Guard | Location |
|-------|----------|
| Path geometry golden | `tests/test_comet_generation.py` — e.g. `Random("orbit_wars-comet-0-50")`, 4 symmetric paths |
| End-to-end spawn sync | `tests/test_jax_env_parity.py` — 50 noop steps, seed 0, `group_count >= 1`, `planets.active == initial_planets.active` |
| Synthetic game states | Use `empty_comet_state()` in parity, golden, and shield fixtures |

```bash
make test-kaggle-parity
```

Runs `tests/test_jax_env_parity.py`, `tests/test_jax_env.py`, `tests/test_jax_env_dispatch.py` with `-m "jax and not slow"` (CPU JAX).

### CI workflow

`.github/workflows/kaggle-jax-parity.yml` triggers on `pull_request` and `push` to `main` / `master`, runs `uv sync --group dev` then `make test-kaggle-parity` with `JAX_PLATFORMS: cpu`. Pass/fail is pytest exit code only — no invented thresholds.

## Why This Matters

Comets change mid-game capture and planet counts; wrong physics after step 50 poisons training metrics and tournament replay. A narrow CI job catches regressions in spawn timing, RNG seeds, collision rules, and `initial_planets` sync without running full `test-premerge` or GPU tiers. This slice is **env stepping parity** — distinct from Gate 5 Docker/tournament proof (`docs/solutions/architecture-patterns/gate5-unified-tournament-submit-valid-funnel.md`) and bracket μ/σ training (`docs/solutions/architecture-patterns/kaggle-bracket-ranking-foundational-slice.md`).

## When to Apply

- Any change to `src/jax/env.py`, `src/game/comet_generation.py`, `src/game/planet_generation.py`, or combat/movement ordering
- Before trusting training or eval metrics in comet-heavy episodes (after step 50)
- When adding new env parity tests — keep them under `-m "jax and not slow"` so `make test-kaggle-parity` and CI pick them up

## Examples

**Wrong → right**

- Wrong: reserve comet planet slots but never spawn — parity gap remains after planets/rotation fixes.
- Wrong: implement comet path search fully in JAX for spawn — hard to match Kaggle; use `pure_callback` + `generate_comet_paths` instead.
- Right: per-spawn RNG `orbit_wars-comet-{episode_seed}-{step}` and spawn when `(step + 1) ∈ COMET_SPAWN_STEPS`.

**Targeted parity test (spawn sync guard):**

```bash
uv run pytest tests/test_jax_env_parity.py::test_comet_spawn_keeps_initial_planets_synced_after_forty_nine_steps -m "jax and not slow"
```

Same file set as CI also runs via `make test-domain-jax-env` (alias for the parity Makefile target).

## Related

- Plans (implementation history): `docs/plans/2026-06-03-008-feat-jax-comet-subsystem-plan.md`, `docs/plans/2026-06-03-009-feat-ci-kaggle-jax-parity-plan.md`
- PR [#188](https://github.com/jmduea/orbit_wars/pull/188) — squash merge on `main`
- Env invariant note (id↔row): `docs/solutions/logic-errors/planet-flow-catalog-reachability-mismatch.md` — comet spawn/expire should preserve the same invariant; refresh that doc if comet slot layout changes
- Operator pointers: `AGENTS.md` (Kaggle reference path, comet RNG, `make test-kaggle-parity`), `docs/ONBOARDING.md`
