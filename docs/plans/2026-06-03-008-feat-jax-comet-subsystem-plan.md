---
date: 2026-06-03
topic: jax-comet-subsystem
status: completed
origin: AGENTS.md JAX/Kaggle env parity gap; review finding #5
---

# Plan: JAX comet subsystem parity with Kaggle

## Summary

Port Kaggle `orbit_wars.py` comet spawn, path movement, first-tick collision skip, expiry, and `initial_planets` sync into `src/jax/env.py` so training rollouts match tournament physics after step 50. Builds on merged env parity work (`planet_generation`, rotation `obs.step`, swept combat).

## Problem Frame

JAX env reserves `TOTAL_COMETS` planet slots but never spawns or moves comets. Kaggle adds neutral capture targets at `COMET_SPAWN_STEPS` with elliptical paths, symmetric 4-fold placement, RNG keyed off episode seed, and special collision rules on first board entry. Missing comets is the largest remaining gap vs `test_orbit_wars.py` (`test_comet_spawn_keeps_initial_planets_synced_across_players`).

## Requirements

| ID | Requirement |
|----|-------------|
| R1 | Spawn comet groups when `(step + 1) ∈ COMET_SPAWN_STEPS` using per-spawn RNG `Random(f"orbit_wars-comet-{episode_seed}-{step+1}")` |
| R2 | Port `generate_comet_paths` logic to `src/game/comet_generation.py` (Python reference, callable from JAX via `pure_callback` or precomputed at spawn) |
| R3 | Extend `JaxGameState` with fixed-shape comet state: group paths, `path_index`, `comet_planet_ids`, episode seed |
| R4 | Advance comet positions each step; skip fleet collision when `old_pos` off-board (`x < 0`); expire when path exhausted |
| R5 | Keep `initial_planets` in sync when comets spawn (append copies) and expire (remove by id) |
| R6 | Parity test port: `test_comet_spawn_keeps_initial_planets_synced_across_players` in `tests/test_jax_env_parity.py` |
| R7 | Existing parity tests remain green (`make test-kaggle-parity`) |
| R8 | Document comet parity in `AGENTS.md` (remove "Not in JAX" once shipped) |

## Key Technical Decisions

**KTD1 — Python path generation, JAX stepping.** Mirror `planet_generation`: implement `generate_comet_paths` in `src/game/comet_generation.py`; spawn step uses `jax.pure_callback` to produce padded path arrays for 4 symmetric comets per group.

**KTD2 — Fixed max path length.** Cap path length at 40 (Kaggle visible segment bound) × 4 groups max over episode; pad inactive slots.

**KTD3 — Planet table integration.** Comets occupy inactive reserved slots (`MAX_PLANETS - TOTAL_COMETS .. MAX_PLANETS-1` or dynamic append into active rows like Kaggle list append). Prefer activating reserved comet slots to keep `MAX_PLANETS` fixed shape.

**KTD4 — Movement order matches Kaggle.** Expire comets before launch → spawn at step boundary → production → compute planet paths (comet + regular) → fleet sweep → apply movement → combat → remove expired.

## Implementation Units

### U1. Reference comet generation module

**Files:** `src/game/comet_generation.py` (new), tests in `tests/test_comet_generation.py`

Port `generate_comet_paths` from Kaggle; unit test returns valid paths for fixed seed.

**Requirements:** R2

### U2. Comet state on `JaxGameState`

**Files:** `src/jax/env.py`, `src/game/constants.py` (reuse `COMET_*`)

Add fields: `episode_seed`, `comet_planet_ids` (int32 vector), `comet_path_index` (per group), padded path tensors, `comet_group_count`.

Wire `reset` to stash episode seed from PRNG key.

**Requirements:** R3, R5

### U3. Spawn and step integration

**Files:** `src/jax/env.py`

Implement spawn at `(step+1) in COMET_SPAWN_STEPS`, path advance, expiry, collision `check_collision` flag for first placement, sync `initial_planets`.

**Requirements:** R1, R4, R5, KTD4

### U4. Parity test + docs

**Files:** `tests/test_jax_env_parity.py`, `AGENTS.md`

Port Kaggle comet sync test (seed 0, 49 noop steps); update learned facts.

**Requirements:** R6, R7, R8

## Test Scenarios

| Unit | Scenario |
|------|----------|
| U1 | Fixed seed produces 4 paths, visible length 5–40, symmetric quadrants |
| U3 | After 49 steps at seed 0, comets non-empty; `len(initial_planets active)` matches planets |
| U3 | Expired comet ids removed from planets and `comet_planet_ids` |
| U4 | Full parity file passes under `make test-kaggle-parity` |

## Verification

```bash
make test-kaggle-parity
make test-domain-jax-env
```

## Out of Scope

- JIT-native comet path search (pure_callback acceptable for spawn-only)
- Feature encoding changes for comet-specific obs fields beyond existing planet table
