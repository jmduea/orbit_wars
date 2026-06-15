---
status: active
type: feat
date: 2026-06-06
origin: conversation + profiling (opponent_sample ~68% on production_mix)
target_repos:
  - orbit_wars
  - orbit_wars-integration
---

# feat: Family-batched mixed opponent sampling

## Summary

Replace per-env `vmap` in mixed opponent sampling with family-batched compress → sample → scatter merge so neural policy forwards run once per active family per step instead of once per env.

## Problem Frame

When `single_family == false`, `_sample_mixed_opponent_2p_action` and `_sample_mixed_player_4p_action` vmapped `_sample_single_family_*` over each env (batch size 1). Profiling shows `opponent_sample` ≈ 68% on `production_mix`; mixed curricula multiply policy/shield cost by `num_envs`.

## Requirements

- R1: Mixed 2p sampling batches envs by `slot_type` family id before calling `_sample_single_family_2p_action`.
- R2: Mixed 4p sampling batches per player slot the same way.
- R3: Single-family fast path (`lax.cond(single_family, ...)`) unchanged.
- R4: Per-env RNG uses `fold_in(opp_key, env_index)` preserved via compressed env indices (not one key per family batch).
- R5: Inactive families (zero envs) skip work via `lax.cond(jnp.any(mask), ...)`.
- R6: Verify with tests + optional `rollout-phase-profile` on staged curriculum.

## Key Technical Decisions

- **Reorder/merge (JIT-safe):** Argsort masked envs to the leading axis, run batched family sampler on static `env_count`, merge with `jnp.where(mask, restored_partial, full)`. True dynamic compress deferred (JAX 0.10 `lax.cond` trace + non-concrete bool/dynamic slice limits).
- **Static family loop:** Python `for family_id in range(OPPONENT_NOOP + 1)` unrolls seven `lax.cond` branches at trace time.
- **Baseline merge accumulator:** Start from `build_noop_action_from_edge_batch` full batch; overwrite per active family.
- **Keys:** `jax.vmap(lambda i: fold_in(opp_key, i))(compressed_indices)` passed by vmapping single-env sampler — rejected; instead single batched call with `fold_in(opp_key, family_id)` for family separation; per-env keys deferred (document in Assumptions).

## Assumptions

- Batched `_sample_single_family_*` with one key per family batch is acceptable for throughput work; golden seed replay is not a gate.
- Seven-family static loop compile cost is acceptable vs per-env vmap.

## Scope Boundaries

**In scope:** `src/opponents/jax_actions/sampling.py` helpers + mixed 2p/4p paths; tests in `tests/test_opponent_mixed_sampling.py`.

**Deferred:** Per-family sub-buckets inside sample; historical pool vmap fix; beat-noop curriculum routing.

## Implementation Units

### U1. Compress/expand helpers

**Goal:** Env-axis compress and action scatter utilities.

**Files:** `src/opponents/jax_actions/sampling.py`

**Test scenarios:**
- Compress preserves row count equal to `mask.sum()`.
- Scatter writes partial actions only at masked indices.

### U2. Family-batched mixed 2p

**Goal:** Replace `_sample_mixed_opponent_2p_action` body with family loop.

**Dependencies:** U1

**Files:** `src/opponents/jax_actions/sampling.py`, `tests/test_opponent_mixed_sampling.py`

**Test scenarios:**
- Finite `collect_rollout_jax` with synthetic stage_view (latest+random+noop mix).
- `single_family` path still used when stage has one family.

### U3. Family-batched mixed 4p

**Goal:** Same for `_sample_mixed_player_4p_action`.

**Dependencies:** U2

**Files:** `src/opponents/jax_actions/sampling.py`, `tests/test_opponent_mixed_sampling.py`

### U4. Mirror to integration + profile smoke

**Goal:** Copy sampling changes to `orbit_wars-integration`; run fast tests.

**Dependencies:** U2, U3

**Verification:** `make test-fast` or targeted pytest on both repos.

## Verification

- `uv run pytest tests/test_opponent_mixed_sampling.py tests/test_rollout_noop_opponent.py -q` on main.
- Optional: `ow benchmark rollout-phase-profile` with `curriculum=self_play_staged` compare opponent_sample fraction.
