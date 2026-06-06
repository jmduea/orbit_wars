---
title: "fix: Pick #4 JAX planet/comet mechanical fidelity on integration"
type: fix
status: active
date: 2026-06-06
origin: docs/session-handoff/2026-06-06-phase2-env-parity-picks-continued.md
---

# fix: Pick #4 JAX planet/comet mechanical fidelity on integration

**Target repo:** `orbit_wars-integration` (branch `throughput-baseline-integration`)

## Summary

Pick #4 landed greenfield pure JAX planet and comet generation on the integration worktree. Adversarial re-review found **mechanical defects** (invalid boards, `initial_planets` desync, comet ID/compile issues) that block pick #5 and admission re-run. This plan fixes those in priority order under a **mechanical fidelity** gate: JAX must obey Orbit Wars rules and produce only valid game states — **not** bit-exact seed replay against `kaggle_environments`.

**Non-goals:** deterministic replay parity with `kaggle_environments` for the same seed; coordinate goldens vs reference on fixed seeds; Kaggle `Random(f"orbit_wars-comet-…")` string RNG bridge; home-group PRNG stream coupling to Python `random.Random`.

**Proof HITL Reference:** Python `src/game/planet_generation.py` and `src/game/comet_generation.py` anchor **rules** (validity bounds, spawn semantics, symmetry). Tests call these directly; we do **not** invoke live `kaggle_environments` during verification.

***

## Problem Frame

Phase 2 pick #4 wired `src/jax/planet_generation.py`, `src/jax/comet_generation.py`, and comet lifecycle into `src/jax/env.py` without callbacks or `src/game/*` imports on the hot path (see `docs/solutions/conventions/jax-no-kaggle-callbacks.md`). Fast gates passed on synthetic mechanics fixtures, but adversarial re-review found merge-blocking **validity** failures:

| Priority | Defect | Symptom |
| :------- | :----- | :------ |
| P0 | `_generation_done` exits after static phase | ~12 planets (3 groups) — below MIN groups (5–10) |
| P0 | Post-move expire drops `initial_planets` | Desync at step ~85 (`planets.active` ≠ `initial_planets.active`) |
| P1 | Comet IDs from `max(active_id)+1` | Breaks reserved-slot semantics and `_launch_fleets` id guard |
| P1 | General placement `px`/`py` share one key | Always `px == py`; reference uses independent draws |
| P1 | Spawn subgraph inlined in every `step` | Long compile at first spawn step |
| ~~P1~~ | ~~Comet RNG integer/string hash~~ | **Deprioritized** — JAX-local deterministic RNG is sufficient |
| ~~P1~~ | ~~Home group from planet-gen stream~~ | **Deprioritized** — valid `randint(0, num_groups-1)` only |

Pick #3 sequential `lax.scan` fleet launch remains **out of scope** (throughput regression).

***

## Requirements

**Mechanical fidelity and hot-path invariants**

* R1. After reset, JAX planet tables are **valid** per reference rules: group count in \[5, 10\] (20–40 active planets), four-planet symmetry, at least one orbiting group when general phase completes, no collisions, all planets in bounds. Maps **may differ** from reference on the same seed.

* R2. `planets.active` and `initial_planets.active` stay equal through comet spawn **and** post-move comet expiry for noop episodes through step 200+.

* R3. Comet spawn obeys reference **rules**: attempt limits (300), path length \[5, 40\], board bounds, collision checks against `initial_planets`, spawn steps 50/150/250/350/450, fourfold symmetry. Paths **may differ** from reference coordinates for the same seed.

* R4. Comet planet IDs at spawn match reserved tail-slot IDs from padded tables (not `max(active_id)+1` allocation).

* R5. Hot path remains JAX-only: no `pure_callback`, `_reference_*`, `env_parity_mode`, or `from src.game.(planet|comet)_generation` in `src/jax/` (tier-A trace hygiene).

* R6. No sequential `lax.scan` in `_launch_fleets` (pick #3 rejection stands).

**Verification**

* R7. Per-pick fast gates pass on integration cwd: `make test-kaggle-parity`; trace hygiene via tier-A `rg` on integration `src/jax/` + `tests/test_jax_trace_hygiene.py` from main harness with integration cwd.

* R8. Validity/invariant tests land on integration (`tests/test_planet_generation.py`, `tests/test_comet_generation.py`, extended `tests/test_jax_env_parity.py`). Optional spot-check seeds are regression anchors, not merge gates.

* R9. Manifest and session handoff updated to record mechanical fidelity green and fix commit SHA.

**Explicitly deferred**

* R10. Operator milestone `make gate-admission` / tier-2 e2e throughput — **not** part of agent verification; operator re-runs when parity stack is green.

***

## Key Technical Decisions

* **KTD1 — Two-phase planet loop termination:** Remove `(static_phase & static_done)` as terminal condition in `_generation_done`. Static completion flips `phase` to 1 and continues until `general_done` or `attempts_done`.

* **KTD2 — Thread `initial_planets` through post-move expire:** `_move_and_resolve` returns updated `initial_planets`; `_finish_step` persists into `next_game`.

* **KTD3 — Comet RNG: JAX-local only (not string replay):** `comet_rng_key(episode_seed, spawn_step)` uses a deterministic integer mix → `jax.random.key`. No Python `Random` string hash bridge.

* **KTD4 — Comet IDs from reserved slots:** Spawn uses `planets0.id[base_slot + i]` from padded tail rows.

* **KTD5 — Independent px/py draws:** Split RNG key before general-group `uniform` calls.

* **KTD6 — Home group: valid index only:** `randint(0, num_groups - 1)` after generation; no requirement to match reference stream coupling.

* **KTD7 — Spawn compile isolation:** `@jax.jit` `_jit_spawn_comet_group` invoked via `lax.cond(should_spawn, ...)`.

* **KTD8 — Validity tests, not coordinate goldens:** Invariant tests assert rules compliance; reference libs used for rule anchors and optional spot checks only.

***

## Scope Boundaries

**In scope**

* `src/jax/planet_generation.py`, `src/jax/comet_generation.py`, `src/jax/env.py`
* Tests: `tests/test_planet_generation.py`, `tests/test_comet_generation.py`, `tests/test_jax_env_parity.py`
* Manifest + handoff on `orbit_wars` main

**Out of scope**

* Seed/bit-exact replay parity with `kaggle_environments`
* Pick #5 mechanics hunks, pick #6 main callback teardown
* Sequential `lax.scan` `_launch_fleets`
* Operator `make gate-admission` (R10)

***

## Implementation Units

### U1. Fix planet generation two-phase loop (P0) — **done**

Two-phase `while_loop`; validity: 5–10 groups, orbiting present, symmetry, bounds, no collisions.

### U2. Persist `initial_planets` through post-move expire (P0) — **done**

`_move_and_resolve` + `_finish_step` wire post-move expire sync.

### U3. Independent general-phase px/py draws (P1) — **done**

Split key for independent x/y uniforms.

### U4. Comet planet IDs from reserved tail slots (P1) — **done**

Spawn uses pre-padded slot IDs.

### U5. Comet RNG string bridge — **skipped**

JAX-local `comet_rng_key` sufficient for mechanical fidelity.

### U6. Hoist comet spawn from unconditional step trace (P1) — **done**

`_jit_spawn_comet_group` + `lax.cond`.

### U6b. Home group PRNG stream coupling — **skipped**

Valid `randint(0, num_groups-1)` only.

### U7. Validity test suite (P1) — **done**

`test_planet_generation.py` (invariants), `test_comet_generation.py` (rules + JAX validity), no coordinate goldens as gates.

### U8. Long-run `initial_planets` sync through step 200+ — **done**

Parametrized noop stepping at milestones 50/85/150/200.

### U9. Manifest and handoff update — **pending operator merge**

Record mechanical fidelity green + integration HEAD SHA after commit.

***

## Sources & Research

* `docs/session-handoff/2026-06-06-phase2-env-parity-picks-continued.md`
* `docs/benchmarks/cherry-pick-manifest.json`
* `docs/solutions/conventions/jax-no-kaggle-callbacks.md`
* Reference: `src/game/planet_generation.py`, `src/game/comet_generation.py`
