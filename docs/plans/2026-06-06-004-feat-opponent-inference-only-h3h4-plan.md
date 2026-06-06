---
status: completed
---

# Opponent inference-only K-step/shield (H3+H4)

## Summary

Land opponent neural rollout optimizations: skip critic/logprob replay (H3) and use unshielded K-step masks with pointwise cheap-shield validation (H4). Address code-review gaps before merge. H1/H2 historical dispatch stays reverted.

## Problem Frame

Opponent sampling dominates rollout collect (~68–70%). H3+H4 targets the K-step factorized decoder + shield lattice. Prior measurement showed ~2–3% opponent-fraction drop (below 10% keep bar) but correct direction. Review flagged safety gaps (`shield_off`), tiered double-validation, missing tests, and `remaining_ships` semantics in pointwise cheap check.

## Requirements

- **R1:** Neural opponent paths call `_sample_opponent_policy_action*` only; learner collect never passes `inference_only=True`.
- **R2:** `inference_only` factorized path skips critic forward, logprob replay, and full lattice shield per K-step; uses unshielded masks + pointwise validation for `cheap`/`tiered`.
- **R3:** `shield_off` opponents still get pointwise cheap validation (or documented assert that neural opponents require non-off shield).
- **R4:** Tiered `inference_only` does not run redundant cheap pointwise when exact post-sample validation runs.
- **R5:** `selected_factored_launch_passes_cheap_shield_jax` uses `remaining_ships` for ship availability, aligned with lattice cheap shield.
- **R6:** Unit tests cover pointwise cheap vs lattice legality and at least one `inference_only` opponent sample smoke.
- **R7:** Targeted tests pass (`test_trajectory_shield_factorized`, opponent ladder, scripted opponents).

## Key Technical Decisions

- **KTD1:** Keep H1/H2 out of scope — historical pool vmap cost accepted for this PR.
- **KTD2:** Opponent `shield_off` runs pointwise cheap validation (same as cheap mode) rather than fully unvalidated launches.
- **KTD3:** Tiered inference_only skips cheap pointwise block; exact `selected_factored_launch_is_exact_safe_jax` remains when `trajectory_shield_final_validate_selected`.
- **KTD4:** Add `remaining_planet_ships` optional arg to pointwise cheap helper; default to `game.planets.ships` for backward compat.

## Implementation Units

### U1. Review autofixes in shield/sampling path

Files: `src/jax/action_sampling.py`, `src/jax/shield/trajectory.py`

- Tiered gate: skip cheap pointwise when tiered exact validation active under inference_only.
- `shield_off` inference_only: apply pointwise cheap validation.
- Pass `remaining_ships[source_row]` into pointwise cheap check.

### U2. Tests

Files: `tests/test_trajectory_shield_factorized.py`, `tests/test_opponent_inference_only.py` (new)

- Pointwise cheap matches lattice bucket legality on existing fixtures.
- Opponent wrapper smoke: `inference_only` produces finite action, zero value, no replay logprob correction.

### U3. Verification

- `make test-fast` domain targets or targeted pytest on U2 files + opponent ladder compose.

## Scope Boundaries

- No H1/H2 historical batched dispatch.
- No port of main `factorized_decode_step` / launch hygiene.
- No full-geometry throughput re-profile in this PR.
- No learner `selected_validate` mode changes.

## Test Scenarios

| ID | Scenario | Expected |
|----|----------|----------|
| T1 | Safe two-planet launch, pointwise vs lattice | Same bucket legal |
| T2 | Sun-cross fixture, pointwise cheap | Rejects unsafe bucket |
| T3 | `_sample_opponent_policy_action_with_params` smoke | Finite JaxAction, value zeros |
| T4 | Tiered inference_only with final validate | Only exact block runs (no double cheap+exact redundancy in behavior) |
