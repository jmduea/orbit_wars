# Deep Dive: Zero Launch Metrics During Mixed Train

## Status
Interview complete — ready for ralplan → omg-autopilot.

## Problem Statement
During `mixed_train_param_search.yaml` with **no trajectory shield**, W&B shows `entropy_move` and `entropy_stop` changing while `mean_active_launches_per_turn` and `stop_utilization_ratio` remain **0**. Game outcome metrics suggest the agent is not launching fleets at all.

## Trace Findings
See `.omg/specs/deep-dive-trace-zero-launch-metrics.md`.

**Root cause (high confidence):** Shield-disabled factorized sampling cannot produce real launches.

1. **`apply_trajectory_shield_factorized_topk`** when `trajectory_shield_enabled=false` sets `ship_bucket_mask[..., 0] = True` only — no buckets > 0 allowed.
2. **`factorized_source_mask_from_shield`** requires `ship_bucket_mask[..., 1:].any()` → `source_mask` is all **False** with shield off.
3. **`_sample_factored_step_from_logits`** applies `jnp.where(selected_bucket_mask, selected_ship_logits, illegal_logit)` where continuous ship logits have shape `(1,)` and bucket mask shape `(bucket_count,)`. JAX broadcasts to `(bucket_count,)`, failing the `continuous_ship` check and forcing the **discrete bucket path** with only bucket 0 legal → `ship_bucket=0`, `ship_fraction=0`.
4. Same broadcast pattern exists in PPO replay (`_factored_step_log_prob_entropy`).
5. **Entropy metrics diverge** because PPO update replays policy logits (`entropy_stop`, `entropy_move`) independent of whether rollout stored launch actions.

**Local repro:** `model=transformer_factorized`, `task.trajectory_shield_enabled=false`, `ship_action_mode=continuous_fraction` → `source_mask any=False`, `mean_active_launches_per_turn=0`, `bucket>0=0`, `frac>0=0`, `stop_rate≈0.33`.

## Interview Answers

| Question | Answer |
|----------|--------|
| Shield-off intent | Skip trajectory simulation only; launches must remain fully legal (all buckets / continuous fraction) |
| Execution path | ralplan → omg-autopilot |
| Metric scope | Fix sampler/replay **and** update launch metric for `continuous_fraction` (`ship_fraction > 0`) |

## Requirements

### R1 — No-shield legality
When `trajectory_shield_enabled=false`, factorized action sampling and PPO replay must allow real launches equivalent to unfiltered edge legality (respect `edge_mask`, owned planets with ships), without trajectory bucket simulation.

### R2 — Continuous ship head correctness
Continuous ship mode (`ship_action_mode=continuous_fraction`) must not fall through to discrete bucket sampling due to `(1,)` vs `(bucket_count,)` mask broadcast.

### R3 — Launch metrics
`mean_active_launches_per_turn` and derived `stop_utilization_ratio` must count active non-stop launch steps:
- Discrete buckets: `ship_bucket > 0`
- Continuous fraction: `ship_fraction > 0`

### R4 — Tests
Add regression tests covering shield-off + continuous_fraction rollout producing non-zero launch metrics and valid env actions.

### R5 — Sweep compatibility
Fix must work for `mixed_train_param_search.yaml` composition without requiring shield re-enable.

## Non-Goals
- Re-enabling trajectory shield in sweep (user wants shield off)
- Thin trajectory shield / M4 intercept work
- Changing entropy metric definitions

## Affected Modules
- `src/game/trajectory_shield.py` — `apply_trajectory_shield_factorized_topk` disabled path
- `src/opponents/jax_actions/builders.py` — `_sample_factored_step_from_logits`
- `src/jax/action_codec.py` — `_factored_step_log_prob_entropy`
- `src/jax/rollout/metrics.py` — `_apply_factorized_metrics`
- Tests: `tests/test_trajectory_shield_factorized.py`, new rollout metric test

## Acceptance Criteria
1. With shield off + continuous_fraction, single rollout shows `mean_active_launches_per_turn > 0` (random init policy, non-zero with high probability over a few seeds).
2. `source_mask` is non-empty when owned planets have legal edges.
3. `transitions.ship_fraction > 0` on non-stop active steps.
4. `make test-fast` passes including new regression tests.
5. Existing shield-on behavior unchanged (parity test).

## Workflow Gates
- [x] Trace complete
- [x] Interview complete
- [ ] ralplan consensus
- [ ] omg-autopilot execution
- [ ] Verification evidence
