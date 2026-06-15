---
title: "feat: Pick #4 pre-generated map pool env parity"
type: feat
status: active
date: 2026-06-06
origin: docs/brainstorms/2026-06-06-pick4-map-pool-requirements.html
supersedes: docs/plans/2026-06-06-002-feat-pick4-compile-bounded-plan.html
---

# feat: Pick #4 pre-generated map pool env parity

**Target repo:** `orbit_wars-integration` on branch `throughput-baseline-integration`

**Supersedes:** [2026-06-06-002-feat-pick4-compile-bounded-plan.html](docs/plans/2026-06-06-002-feat-pick4-compile-bounded-plan.html) (inline JAX generation — compile cliff proven)

## Summary

Replace inline JAX planet/comet generation in training-path `reset()` with a **500-entry pre-baked map pool** (planets + full comet wave schedules). Hot-path reset becomes constant-time gather from frozen tensors; round-robin map selection rotates across reseeds via per-env `episode_count`. Offline bake uses reference Python generators; eval/tournament paths stay on Kaggle observation replay. R1 single-map profile is **already satisfied** (~0.11 s/map, ~55 s extrapolated batch).

---

## Problem Frame

Pick 4a (inline `src/jax/planet_generation.py` in `reset()`) passed parity and trace hygiene but failed compile: cold-cache benchmark smoke hung 14+ minutes — same cliff as `0eb349e`. Rejection-sampling `while_loop` inlined into vmapped training trace is ruled out. Integration currently holds uncommitted 4a work on top of rollback substrate `9db50f5`; map-pool work must start from a clean substrate without inline generation modules in the hot path.

---

## Requirements

Traced from origin R1–R15.

| ID | Requirement | Plan coverage |
| --- | --- | --- |
| R1. | Single-map profile before 500-map bake | Satisfied (session); recorded in U4 |
| R2. | Operator accepts extrapolated batch cost | Satisfied (~1 min batch); U4 |
| R3. | Bake-time mechanical validity per entry | U2, U3, U4, U7 |
| R4. | 500-entry default pool | U4 |
| R5. | Round-robin map rotation on training reset | U6 |
| R6. | No generation/search in hot-path reset | U1, U6 |
| R7. | Step activates baked comet schedules only | U6 |
| R8. | Committed default pool + hash in manifest | U4, U8 |
| R9. | `ow` rebake + Hydra path override | U3, U5 |
| R10. | Eval/replay on Kaggle env — not pool reset | U7 (regression guard) |
| R11. | JAX-only hot path; tier-A trace hygiene | U1, U6, U8 |
| R12. | No sequential `lax.scan` in `_launch_fleets` | No change (invariant) |
| R13. | Fast gates: kaggle parity + trace hygiene | U1 (trace hygiene), U7 (kaggle parity), U8 (gate verdict) |
| R14. | Compile smoke ≤5 min cold cache | U8 |
| R15. | Manifest `env_parity_mode: map_pool` + metrics | U8 |

### Preconditions (R1–R2)

R1/R2 are **satisfied in the planning session** (~0.11 s/map, ~55 s extrapolated batch). U3 `profile` remains a rebake/audit primitive. U4 attaches session metrics to the bake manifest and runs the 500-map bake without re-gating unless profile metrics are stale. Implementer should still run `profile --out` once on the integration worktree and attach the artifact to the manifest before committing the npz.

---

## Key Technical Decisions

| ID | Decision | Rationale |
| --- | --- | --- |
| KTD1. | Pool artifact: versioned `.npz` + sidecar `manifest.json` under `data/jax_map_pool/` | NumPy stack is diff-friendly enough for hash pinning; loader maps directly to JAX constants. Sidecar holds sha256, count, profile metrics, generator version — not duplicated in Hydra defaults. |
| KTD2. | Map index at reset: `(episode_count + env_index) % pool_size` | `episode_count` increments on done in `collect.py` reset branch; adding `env_index` avoids all parallel envs sharing one map on first reset. (see origin: R5) |
| KTD3. | Each pool entry stores full comet wave tensors (inactive at reset); `step()` only toggles `group_active` and advances along pre-baked paths | Matches greenfield `75a7cf2` activation pattern without path search in trace. (see origin: R6, R7) |
| KTD4. | Offline bake only via `src/game/planet_generation.py` and `src/game/comet_generation.py`; delete `src/jax/planet_generation.py` from integration | Hot path must not import `src/game/*` per jax-no-kaggle-callbacks convention. Inline JAX gen proven to brick compile. |
| KTD5. | CLI: `ow benchmark map-pool` with `profile`, `bake`, `validate` subcommands | Follows existing `ow benchmark` registration pattern; rebake is operator primitive, not a standalone script. (see origin: R9) |

---

## High-Level Technical Design

### Data flow: offline bake → train reset

```mermaid
flowchart LR
  A[Reference Python game generators] --> B[ow map-pool bake / validate]
  B --> C[data/jax_map_pool .npz + manifest]
  C --> D[pool loader JAX constants]
  D --> E[reset gather + round-robin]
```

Offline stages (bake, validate, load at train init) stay outside the JIT trace. Only gather + home assignment run inside vmapped `reset()`.

### Eval carve-out

Training and benchmark smoke use pool gather reset. Tournament, Docker packaging validation, and `jax_game_from_observation_fast` replay paths are unchanged — they never call pool `reset()` for board truth. Eval/submit-valid paths use observation replay only. Training-path parity tests (`make test-kaggle-parity`) reset via the map pool and assert mechanical invariants — not bit-exact Kaggle seed replay.

---

## Output Structure

```text
data/jax_map_pool/
  default_v1.npz              # stacked pool tensors [500, ...]
  default_v1.manifest.json    # sha256, count, profile, bake metadata
src/jax/map_pool/
  home_assignment.py          # JIT-safe assign_home_planets (extracted from 4a)
  bake.py                     # offline entry builder (imports src/game/*)
  load.py                     # NPZ → frozen JAX arrays at train init
src/cli/benchmark/map_pool.py
conf/task/map_pool.yaml       # task=map_pool; path + size override on TaskConfig
tests/test_map_pool_bake.py
tests/test_map_pool_reset.py
```

---

## Implementation Units

### U1. Substrate cleanup — remove failed inline 4a

**Goal:** Restore integration hot path to rollback substrate without inline JAX generation imports.

**Requirements:** R6, R11, KTD4

**Dependencies:** None

**Files:**

- Modify: `src/jax/env.py`, `tests/test_jax_env_parity.py`, `tests/conftest.py`, `Makefile`
- Create: `src/jax/map_pool/home_assignment.py` (extract from 4a before delete)
- Delete: `src/jax/planet_generation.py`, `tests/test_planet_generation.py` (if present)

**Approach:** Discard uncommitted 4a work: remove `planet_generation` import from `env.py`, revert parity/Makefile/conftest changes tied to inline gen. Before deleting `src/jax/planet_generation.py`, extract JIT-safe `assign_home_planets` to `src/jax/map_pool/home_assignment.py` (needed by U6 reset-time home assignment). Stub `reset()` may remain until U6 — but must not call generation helpers. Verify tier-A `rg` clean on integration `src/jax/`. Parity suite may be partial between U1 and U6; do not treat U1 parity collect as R13 satisfaction until U7 completes.

**Patterns to follow:** `9db50f5` env reset substrate; `docs/solutions/conventions/jax-no-kaggle-callbacks.md`

**Test scenarios:**

- Happy path: tier-A forbidden-pattern scan returns zero matches on integration `src/jax/`.
- Integration: existing fast parity suite still collects (may still use stub boards until U6).

**Verification:** No `src/jax/planet_generation.py` in tree; no `generate_planet_tables` in `env.py`; trace hygiene gate unchanged green.

---

### U2. Pool schema and offline bake module

**Goal:** Define one pool entry's tensor layout and build entries from reference generators with bake-time validity.

**Requirements:** R3, KTD1, KTD4

**Dependencies:** U1

**Files:**

- Create: `src/jax/map_pool/bake.py`, `tests/test_map_pool_bake.py` (extract `schema.py` only if bake + load need shared constants without cross-imports)

**Approach:** `bake_one_entry(seed)` calls `generate_planets` then `generate_comet_paths` for all spawn waves at a fixed per-entry `angular_velocity` (sampled once per entry, stored in pool — comet path validity is av-coupled). Converts to fixed-shape arrays matching `JaxPlanetState` and full comet schedule tensors (all groups, paths, lengths). Run existing validity helpers from planet tests / game invariants at bake time; fail bake on invalid entry.

**Execution note:** Test-first: bake validity tests before wiring CLI.

**Patterns to follow:** `src/game/planet_generation.py`, `src/game/comet_generation.py`; greenfield `JaxCometState` from `75a7cf2`

**Test scenarios:**

- Covers R3. Happy path: 10 fixed seeds each produce valid entry (5–10 groups, symmetry, comet waves).
- Edge case: bake rejects entry when planet group count out of range (inject bad mock only in test).
- Integration: baked planet tensors round-trip through existing validity assertion helpers used in parity tests.

**Verification:** `tests/test_map_pool_bake.py` green; single-entry bake callable from Python without JAX compile.

---

### U3. `ow benchmark map-pool` CLI

**Goal:** Expose profile, batch bake, and validate as operator primitives with JSON stdout.

**Requirements:** R1, R2, R3, R9, KTD5

**Dependencies:** U2

**Files:**

- Create: `src/cli/benchmark/map_pool.py`
- Modify: `src/cli/benchmark/parser.py`, `src/cli/__init__.py`, `src/cli/benchmark/common.py`

**Approach:** Subcommands: `profile --repeats N --out path.json` (single-map timing), `bake --count 500 --out-dir data/jax_map_pool --label default_v1 --profile path.json` (requires prior profile or explicit `--accept-extrapolated-secs` — blocks batch bake when extrapolated time is unacceptable per origin AE2), `validate --pool path.npz`. Defer heavy JAX imports until subcommand runs. Write manifest sidecar with sha256 and profile stats on bake.

**Patterns to follow:** `src/cli/benchmark/parser.py` gate/training subcommands; AGENTS.md CLI policy

**Test scenarios:**

- Happy path: parser registers `map-pool`; dry `validate` on tiny 2-entry test artifact exits 0.
- Edge case: `bake --count 0` fails fast with actionable error.
- Covers AE2. Error path: `bake` without profile and without `--accept-extrapolated-secs` exits non-zero when profile JSON shows unacceptable 500× extrapolation.
- Integration: `uv run ow benchmark map-pool --help` lists subcommands (dispatch smoke).

**Verification:** CLI help and validate path work; profile JSON schema documented in sidecar example.

---

### U4. Production 500-map bake and committed artifact

**Goal:** Land default pool in-repo with recorded R1 profile and content hash.

**Requirements:** R1, R2, R3, R4, R8

**Dependencies:** U3

**Files:**

- Create: `data/jax_map_pool/default_v1.npz`, `data/jax_map_pool/default_v1.manifest.json`

**Approach:** Record completed R1 profile in manifest (mean ~0.11 s/map, 10/10 seeds valid, extrapolated ~55 s). Run `ow benchmark map-pool bake --count 500` with deterministic seed stream; validate before commit. Binary npz committed; manifest is human-readable audit trail.

**Test scenarios:**

- Happy path: committed npz loads; manifest sha256 matches file on disk; manifest records session R1 profile metrics (AE1 satisfied pre-implementation).
- Happy path: entry count == 500; validate subcommand passes on committed artifact.

**Verification:** Artifact pair exists; validate green; manifest documents profile metrics and bake timestamp.

---

### U5. JAX pool loader and Hydra config

**Goal:** Load pool once at train/env init into immutable JAX arrays; support path override.

**Requirements:** R8, R9, KTD1

**Dependencies:** U4

**Files:**

- Create: `src/jax/map_pool/load.py`, `conf/task/map_pool.yaml`
- Modify: `src/config/schema.py` (`TaskConfig.map_pool_path`, optional `map_pool_sha256` read from sidecar at load)

**Approach:** `load_map_pool(path) → MapPoolConstants` with leading dimension `[pool_size, ...]`. Hydra `task.map_pool_path` defaults to committed artifact via `task=map_pool` composition; override for experiments. Load in `init_rollout_groups` (before `_init_rollout_group` / `collect_fn` jit); pass `MapPoolConstants` into rollout group state and close into `collect_fn`. Populate read-only `map_pool_sha256` from sidecar for resolved-config audit (R8).

**Test scenarios:**

- Happy path: loader returns arrays with expected shapes for 2-entry test npz.
- Error path: missing file raises clear error before train loop starts.
- Integration: resolved config smoke shows default pool path when `print_resolved_config=true`.

**Verification:** Loader unit tests pass; config composes with default path.

---

### U6. Pool gather reset and comet step activation

**Goal:** Wire training-path `reset()` and `step()` to pool data with round-robin selection and baked comet schedules.

**Requirements:** R5, R6, R7, R11, R12, KTD2, KTD3

**Dependencies:** U5

**Files:**

- Modify: `src/jax/env.py`, `src/jax/rollout/collect.py` (`map_id`, `episode_count`, `env_index` on done-reset branch), `src/jax/train/rollout_groups.py` (pool + `map_ids` on initial `batched_reset` in `_init_rollout_group`)
- Create: `tests/test_map_pool_reset.py`

**Approach:** Extend `reset(key, cfg, map_pool, map_id)` and `batched_reset(keys, cfg, map_pool, map_ids)` with `in_axes=(0, None, None, 0)`. Compute `map_ids = (reset_episode_counts + env_indices) % pool_size` in collect reset branch (reuse existing `env_indices` / `env_index_offset`). Port `JaxCometState`, `empty_comet_state`, `_advance_comet_positions`, `_expire_comets_pre_launch`, and comet branches in `_move_and_resolve` from `75a7cf2`. **Replace** `_spawn_comet_group` with `_activate_baked_comet_group` (sets `group_active`, places pre-baked paths — no `generate_comet_paths`). Gather planets + comet schedule + per-entry `angular_velocity` from pool; assign homes via `home_assignment.assign_home_planets` (no rejection loop). `step()`: activate groups at `COMET_SPAWN_STEPS`, advance along baked paths only.

**Execution note:** Characterization-first: pool-reset validity in `tests/test_map_pool_reset.py` only; defer `test_jax_env_parity.py` changes to U7.

**Patterns to follow:** `75a7cf2:src/jax/env.py` comet types; `src/jax/rollout/collect.py` reset branch; `docs/solutions/conventions/jax-no-kaggle-callbacks.md`

**Test scenarios:**

- Covers R5 / R6. Happy path: two consecutive resets with incremented `episode_count` select different map indices mod `pool_size`.
- Covers R7. Happy path: after reset, stepping to first comet spawn step activates baked group without generating paths.
- Edge case: `pool_size` override with 2-entry test pool wraps `map_id` correctly.
- Integration: vmapped `batched_reset` produces valid `active_count` in 5–10 range for random map ids.
- Integration: initial group reset in `rollout_groups.py` selects distinct map indices per env (not only consecutive done-resets).

**Verification:** Pool reset tests green; tier-A scan still zero; no `while_loop` rejection in reset path.

---

### U7. Parity suite and gate test updates

**Goal:** Restore full kaggle parity coverage with pool-based training reset; remove comet deferral markers.

**Requirements:** R3, R10, R13

**Dependencies:** U6

**Files:**

- Modify: `tests/test_jax_env_parity.py`, `tests/conftest.py`, `Makefile`

**Approach:** Parity tests that stepped JAX env from pool reset assert mechanical invariants (not bit-exact Kaggle seed replay). Obs-replay tests unchanged for eval path. Remove `requires_comets` skip markers added for 4a deferral. Ensure `make test-kaggle-parity` exercises comet spawn steps where applicable.

**Test scenarios:**

- Covers R10. Happy path: tournament/eval code paths grep shows no pool loader import in eval modules.
- Covers R13. Integration: `make test-kaggle-parity` passes on integration worktree.
- Happy path: multi-step parity scenario with comet wave reaches valid terminal or mid-game state.

**Verification:** Kaggle parity Makefile target green; no deferred comet-only skips remain without justification.

---

### U8. Compile smoke and manifest admit

**Goal:** Prove ≤5 min cold-cache compile and record pick #4 map-pool admit on main manifest.

**Requirements:** R14, R15

**Dependencies:** U7

**Files:**

- Modify (main repo): `docs/benchmarks/cherry-pick-manifest.json`

**Approach:** Run cold-cache compile smoke from integration worktree with `env -u JAX_COMPILATION_CACHE_DIR ORBIT_WARS_PYTEST_JAX_CACHE=0` and `ow benchmark training --preset primary --updates 3 --warmup 1 --out /tmp/pick4_compile.json`. On pass, update manifest candidate `pick_4_map_pool` with `env_parity_mode: map_pool`, pool sha256, R1 profile JSON, compile verdict, integration HEAD. On fail: rollback per phase2-pick4-jax-compile-rollback-criteria — do not admit.

**Test scenarios:**

- Covers AE3. Happy path: compile smoke JSON reports completion within 300 s.
- Covers AE4. Regression: eval submit path documentation unchanged — Kaggle env only.

**Verification:** Compile ≤5 min; fast gates green; manifest updated on main with map_pool mode and metrics.

---

## Scope Boundaries

### Deferred for later

- Held-out map slice for admission overfitting checks.
- Operator `make gate-admission` tier-2 until parity + compile stack green.
- Bit-exact seed replay vs `kaggle_environments` for training maps.
- Pick #5 mechanics hunks until map-pool pick #4 admits.

### Outside this product's identity

- Inline JAX live generation in vmapped `reset()` / `step()`.
- Pool-based tournament boards.
- `pure_callback` reference generators on hot path.

### Deferred to Follow-Up Work

- Parallel/distributed bake worker if pool size grows beyond 500.
- Automated CI rebake — default pool is committed static artifact for this pick.

---

## Risks & Dependencies

| Risk | Mitigation |
| --- | --- |
| 500× stacked tensors inflate compile or HBM despite gather-only reset | U8 compile gate; ~5 MB NPZ is unlikely to replay inline-gen compile cliff — if fail, first inspect dynamic gather inside jitted `collect_fn` closure before shrinking pool. |
| Round-robin `map_id` needs `env_index` plumbed through batched reset | U6 explicitly wires from `collect.py`; test in U6 integration scenario. |
| Committed npz size bloats repo | Accept for reproducibility (R8); document size in manifest; Git LFS only if >50 MB in practice. |
| Uncommitted 4a work confuses implementer | U1 is mandatory first step on integration worktree. |

---

## Implementation-Time Unknowns

- Exact NPZ key naming (defer to U2 implementation).
- Home assignment: **reset-time** from gathered neutral planets + `assign_home_planets` (decided; preserves player RNG). Validate KTD2 `env_index` term in U6 integration tests; drop only if tests prove redundant.
- Final npz byte size and whether git commit needs LFS pointer.

---

## Sources & Research

- Origin: `docs/brainstorms/2026-06-06-pick4-map-pool-requirements.html`
- `docs/solutions/workflow-issues/phase2-pick4-jax-compile-rollback-criteria.md`
- `docs/solutions/conventions/jax-no-kaggle-callbacks.md`
- Greenfield reference: integration commits `75a7cf2`, `0eb349e` (rollback forensics)
- R1 profile (session): ~0.11 s/map mean, 500× extrapolation ~55 s, 10/10 seeds valid with comets
