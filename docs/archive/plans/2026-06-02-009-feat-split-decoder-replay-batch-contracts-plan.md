---
title: "feat: Split decoder replay batch contracts (#167)"
date: 2026-06-02
status: completed
type: feat
origin: "GitHub #167; docs/ROADMAP.md Now"
---

# feat: Split Decoder Replay Batch Contracts (#167)

## Summary

Replace optional Planet Flow pressure fields and fabricated factorized sequence fields on a flat `JaxTransitionBatch` with explicit `FactorizedActionReplay` and `PlanetFlowActionReplay` variants nested under `action_replay`. Rollout constructs the matching variant; PPO dispatches on replay type instead of defending against alien optional fields.

---

## Problem Frame

Planet Flow rollout zeroes factorized sequence tensors while populating `planet_flow_target_*` optional fields on the same batch type. That implies both action contracts can coexist and forces factorized and Planet Flow PPO paths to guard against mismatched payloads (#167).

---

## Requirements

- **R1:** Introduce `FactorizedActionReplay` and `PlanetFlowActionReplay` NamedTuples in `src/jax/rollout/types.py`.
- **R2:** `JaxTransitionBatch` holds shared observation/GAE fields plus `action_replay: FactorizedActionReplay | PlanetFlowActionReplay` (no top-level optional replay fields).
- **R3:** Planet Flow collect path stops fabricating factorized sequence fields; factorized path stops carrying Planet Flow pressure fields.
- **R4:** `ppo_update_jax` dispatches on `isinstance(batch.action_replay, …)` and removes cross-decoder optional-field guards.
- **R5:** Tests cover replay variant construction, dispatch mismatch errors, and existing Planet Flow / factorized PPO contracts.
- **R6:** Update `docs/ROADMAP.md`; close #167 in PR.

---

## Key Technical Decisions

- **Nested union on batch** — keeps one rollout batch type for concat/normalize while making replay contracts mutually exclusive at the type level.
- **Scan dict stays flat for metrics** — rollout metrics continue keying off `planet_flow_target_bucket` presence in scan `data`; batch assembly nests into `action_replay` at collect exit.
- **Env row count from observation** — `_flatten_transition_to_turn_batch` uses `planet_features` time×env shape, not fabricated `target_index`.

---

## Scope Boundaries

### Non-goals

- Changing PPO numerics, metric descriptors, or decoder policy architectures.
- Splitting `collect_jax_rollout` into wholly separate modules per decoder.

---

## Implementation Units

### U1. Replay variant types

**Goal:** Define explicit replay NamedTuples and refactor `JaxTransitionBatch`.

**Files:** `src/jax/rollout/types.py`

**Test scenarios:** Import and construct both variants; batch exposes exactly one replay kind.

### U2. Rollout collect assembly

**Goal:** Build the correct `action_replay` variant; drop Planet Flow zeroed factorized tensors from scan output.

**Files:** `src/jax/rollout/collect.py`

**Test scenarios:** Planet Flow policy rollout transitions use `PlanetFlowActionReplay` without factorized sequence fields on batch.

### U3. PPO dispatch and helpers

**Goal:** Read replay fields from nested variant; dispatch on `isinstance`.

**Files:** `src/jax/ppo_update.py`

**Test scenarios:** Factorized config rejects Planet Flow replay; Planet Flow rejects factorized replay; existing invalid-bucket test still fails.

### U4. Test and ROADMAP updates

**Goal:** Update helpers/tests; move #167 to Done.

**Files:** `tests/test_planet_flow_action_contract.py`, `tests/test_ppo_update.py`, `tests/test_planet_flow_policy.py`, `docs/ROADMAP.md`

**Verification:** `make test-fast` and planet-flow domain tests pass.

---

## Assumptions

- Decoder type is fixed per training run, so scan pytree shape may differ between compiled factorized and Planet Flow collect paths.
- `jax.tree.map` concatenation remains valid when all batches in a group share the same replay variant.
