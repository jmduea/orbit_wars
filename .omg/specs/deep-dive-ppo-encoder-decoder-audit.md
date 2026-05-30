# Deep Dive Spec: PPO Encoder/Decoder Audit

## Trace Findings (injected)
Factorized PPO was broken at `mask_factored_policy_output_for_shield` when rollout supplied 5D `ship_bucket_mask`; joint-flat path was fine. PPO hyperparameters under `training.*` compose correctly; `epochs` is applied in `train.py`, not inside `ppo_update_jax`. Advantages use Monte Carlo returns minus turn-level value (no GAE). `enable_gradient_checkpointing` is unused in JAX training.

## Goal
Confirm PPO correctness after encoder/decoder changes, ensure hyperparameters are exposed, validate loss sources, and add regression tests.

## Acceptance Criteria
- [x] Joint-flat and factorized PPO update paths run without shape errors
- [x] PPO loss uses shield-masked log-probs consistent with rollout semantics
- [x] Hydra overrides for core PPO knobs resolve to `TrainingConfig`
- [x] Unit tests prove returns math, on-policy KL≈0, coef scaling, and param updates
- [x] Fix factorized shield masking bug blocking factorized training updates

## Implementation Notes
- Fix: `src/game/trajectory_shield.py` — collapse planet/bucket axes when building target/ship step masks for 5D bucket tensors
- Tests: `tests/test_ppo_update.py` (12 CPU Hydra cases + 4 JAX behavioral tests)

## Open Questions (for follow-up)
1. Should we add `gae_lambda` to `TrainingConfig` or document Monte Carlo-only design?
2. Wire or remove `enable_gradient_checkpointing`?
3. Should joint-flat PPO honor `step_mask` when early-stop is added to joint path?

## Execution Bridge
Recommended: treat as complete for audit scope; optional `ralplan` if adding GAE or gradient checkpointing.
