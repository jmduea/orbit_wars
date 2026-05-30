# Deep Dive Trace: PPO After Encoder/Decoder Changes

## Problem
Verify JAX PPO still works after planet-edge encoder/decoder refactors (joint-flat + factorized top-K), that tunable hyperparameters are wired, and losses are computed from the correct policy/value heads.

## Lane A — Code Path (most likely)
**Hypothesis:** PPO dispatch and loss math are structurally sound but factorized shield masking had shape bugs blocking the factorized update path.

**Evidence for:**
- `ppo_update_jax` cleanly dispatches on `is_factorized_pointer_decoder`.
- Joint-flat path applies `mask_policy_output_for_shield_v2` before `action_log_prob_and_entropy`.
- Factorized path mirrors PPO clipped objective with `factored_action_log_prob_and_entropy` and `step_mask`.
- `train.py` applies `cfg.training.epochs` by repeating jitted `ppo_update_jax`; optimizer uses `lr` + `max_grad_norm`.

**Evidence against:**
- `mask_factored_policy_output_for_shield` used `bucket_step_mask.any(axis=-1)` on 5D masks, producing `(batch, seq, P, k)` masks incompatible with `(batch, seq, k)` target logits — **fixed**.

**Critical unknown:** Whether turn-level scalar value broadcast to all K launch steps is intentional (Monte Carlo advantage without GAE).

**Discriminating probe:** Synthetic factorized transition batch through `ppo_update_jax` (now passes).

## Lane B — Config / Hyperparameters
**Hypothesis:** All PPO knobs live under `training.*` in Hydra and compose to `TrainingConfig`.

**Evidence for:** `gamma`, `clip_coef`, `ent_coef`, `vf_coef`, `lr`, `max_grad_norm`, `minibatch_size`, `epochs`, `update_chunk_rows_{min,max}` all compose via `compose_hydra_train_config`.

**Evidence against:** `enable_gradient_checkpointing` is schema-only (not referenced in JAX training path) — dead config.

**Critical unknown:** Whether users expect GAE (`gae_lambda`) — not implemented; advantages are `returns - V(s)`.

## Lane C — Measurement / Tests
**Hypothesis:** Existing tests were joint-flat rollout smokes; factorized PPO was untested end-to-end.

**Evidence for:** `test_jax_ppo.py` smokes joint-flat collect+update; no prior factorized PPO unit tests.

**Evidence against:** New `tests/test_ppo_update.py` covers returns math, Hydra knobs, on-policy KL (joint + factorized), coef scaling, optimizer step.

**Convergence:** Primary actionable defect was factorized shield mask broadcasting; hyperparameter wiring is correct except dead gradient-checkpoint flag and missing GAE.
