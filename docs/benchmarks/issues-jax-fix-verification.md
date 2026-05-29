# Issues.md JAX fix — verification report

Worktree: `issues-jax-14f07ba2` vs baseline `e7f8e4c`. Evidence is file:line or test name unless noted.

## Summary

| Issue (Issues.md) | Verdict | Evidence |
|-------------------|---------|----------|
| Continuous ship-fraction log-prob | **FIXED** | `ship_action.py:45-61`, `action_codec.py` uses `continuous_fraction_log_prob_at_action`; `builders.py:388`; tests `test_continuous_ship_logprob_depends_on_policy_loc`, `test_rollout_replay_logprob_parity_continuous_fraction` |
| Factorized K+1 policy forwards | **FIXED** | `factored_sequence_scan.py:236` single `policy.apply`; mask-only `lax.scan` in `_replay_masks_and_logprobs_from_output` |
| Parity metrics gated | **FIXED** | `conf/training/base.yaml` `debug_replay_parity: false`; `ppo_update.py:565` |
| Scalar returns / advantages | **FIXED** | `rollout/collect.py:372-373` stores `[T,N]`; `ppo_update.py:247-249` `_flatten_state_scalars` + `_actor_advantages_from_state`; `_value_loss_per_state` |
| C51 dense targets | **FIXED** | `distributional_value.py:71+` `sparse_categorical_value_cross_entropy`; used in `_value_loss_per_state`; tests `test_sparse_categorical_value_cross_entropy_matches_dense`, `test_value_loss_per_state_uses_cross_entropy_for_distributional_head` |
| Missing value-only path | **NOT FIXED** | No `value_only` in `policy.py` (advisory; mitigated by merged replay forward) |
| Padded minibatch trees | **NOT FIXED** | `_reshape_minibatches` still used in `ppo_update.py` |
| Geometry + mixed precision | **NOT FIXED** (advisory) | No bf16 rollout/network policy added |

## Detail

### 1. Continuous ship-fraction log-prob — FIXED

Replay and rollout evaluate `log p(fraction | policy_loc)` via `continuous_fraction_log_prob_at_action` (`src/jax/ship_action.py:45-61`). Factorized step path calls it when `ship_fraction` is set (`src/jax/action_codec.py` ~336-341). Rollout builder uses the same helper (`src/opponents/jax_actions/builders.py:388`). No `_logit_from_fraction` overwrite before density in replay.

**Test:** `test_continuous_ship_logprob_depends_on_policy_loc` — non-zero `jax.grad` w.r.t. `policy_loc` with fixed fraction.

### 2. Single-pass factorized replay — FIXED

`replay_factored_sequence_logprob` performs one teacher-forced `policy.apply` (`factored_sequence_scan.py:228-236`), then `_replay_masks_and_logprobs_from_output` scans only masks/log-probs without additional forwards.

**Test:** `test_factored_sequence_scan` rollout↔replay parity (buckets + continuous).

### 3. Parity metrics gated — FIXED

`TrainingConfig.debug_replay_parity` default `False` (`schema.py`, `conf/training/base.yaml`). `factored_logprob_parity_metrics` only when flag true (`ppo_update.py:565-587`).

### 4. Scalar returns / advantages — FIXED

Rollout stores scalar `returns_step` / `advantages_step` without broadcast to sequence shape (`collect.py:372-373`). PPO flattens state scalars and late-broadcasts for actor (`ppo_update.py` `_flatten_state_scalars`, `_actor_advantages_from_state`). Critic uses `_value_loss_per_state` once per env row.

### 5. Sparse C51 — FIXED

`sparse_categorical_value_cross_entropy` in `distributional_value.py:71+`; wired in `_value_loss_per_state` (`ppo_update.py:223-225`). Equivalence test vs dense projection.

### 6. Not implemented (expected)

- **Dynamic minibatch gather:** still materializes padded minibatch trees.
- **value_only:** not added; single forward supplies replay + value in post-fix path.
- **Mixed precision:** advisory only; no global bf16 enablement.

## Regressions / gaps

- Hydra default `opponents/default` still references missing `self_play_curriculum` — breaks some compose tests; benchmarks use `opponents=self_play_only`.
- `make test-fast` may still hit unrelated worktree import/Hydra issues (documented in benchmark notes).
- GPU benchmark used `format=2p_4p_8env` (16 envs) to avoid OOM vs production 32+32 format.

## Tests run

```
uv run --group dev pytest tests/test_factored_sequence_scan.py \
  tests/test_distributional_value.py tests/test_action_codec.py -m "not slow"
```

Result: **17 passed, 1 failed** (Hydra missing config; not introduced by CUDA deps).
