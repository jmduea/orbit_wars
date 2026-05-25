# Deep Interview Spec: PPO GAE + Gradient Checkpointing

## Goal
Wire standard PPO Generalized Advantage Estimation (`gae_lambda`) and connect `training.enable_gradient_checkpointing` to JAX encoder forward passes.

## Parent Context
Follow-up from `.omg/specs/deep-dive-ppo-encoder-decoder-audit.md`:
- Advantages today: Monte Carlo returns minus turn-level value (no GAE)
- `enable_gradient_checkpointing` exists in schema/YAML but is unused in JAX training

## Scope
**In:**
- Add `training.gae_lambda` to `TrainingConfig`, Hydra YAML, runtime validation
- Compute GAE at rollout timestep granularity in `collect_rollout_jax`
- Broadcast step-level advantages/returns to K launch steps (preserve current transition layout)
- Apply Flax `nn.remat` to encoder layer blocks when flag is true (GNN + planet graph transformer)
- CPU/JAX-light tests for GAE math and config compose; smoke that remat flag toggles without shape errors

**Out:**
- Decoder/value-head rematerialization (encoder-only first)
- Changing PPO loss structure or adding separate value normalization
- Full slow-tier rollout/training benchmark gate (user approval only)

## Acceptance Criteria
1. `training.gae_lambda` composes via Hydra and affects computed advantages
2. `gae_lambda=1.0` reproduces current Monte Carlo advantage semantics (within float tolerance) on fixed toy tensors
3. `enable_gradient_checkpointing=true` wraps encoder layer forwards with remat; `false` preserves current graph
4. Policy init/apply and `ppo_update_jax` succeed for both joint-flat and factorized decoders with flag on/off
5. `tests/test_ppo_update.py` extended; `make test-fast` green

## Constraints
- Preserve existing transition batch shapes and PPO update API
- Default `gae_lambda` must not silently change production behavior for existing YAML unless explicitly chosen
- Remat must not alter checkpoint param tree structure (same module names)
