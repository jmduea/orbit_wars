# Deep Dive Trace: Zero Launch Metrics During Mixed Train

## Problem
During `mixed_train_param_search.yaml` (no trajectory shield), W&B shows `entropy_move` and `entropy_stop` changing while `mean_active_launches_per_turn` and `stop_utilization_ratio` stay at 0.

## Lane A — Code Path / Action Generation (most likely)
**Hypothesis:** Shield-disabled factorized sampling cannot produce real launches because the no-shield bucket mask is noop-only and continuous ship logits are broadcast into the discrete bucket path.

**Evidence for:**
- `apply_trajectory_shield_factorized_topk` with `trajectory_shield_enabled=false` sets `default_bucket_mask[..., 0] = True` only (no buckets > 0).
- `factorized_source_mask_from_shield` requires `ship_bucket_mask[..., 1:].any()` → `source_mask` is all False when shield is off (`tests/test_trajectory_shield_factorized.py::test_factorized_source_mask_requires_ships_and_buckets`).
- In `_sample_factored_step_from_logits`, `jnp.where(selected_bucket_mask, selected_ship_logits, illegal_logit)` broadcasts `(bucket_count,)` mask against `(1,)` continuous logits → shape `(8,`) → `continuous_ship` check fails → discrete bucket path with only bucket 0 legal → `ship_bucket=0`, `ship_fraction=0`.
- Local repro with sweep-like config (`transformer_factorized`, shield off, continuous_fraction): `source_mask any=False`, `mean_active_launches_per_turn=0`, `bucket>0=0`, `frac>0=0`, but `stop_rate≈0.33`.
- PPO update computes `entropy_stop` / `entropy_move` from replayed logits (`src/jax/ppo_update.py`) independent of whether rollout produced launches.

**Evidence against:**
- `noop_percent` is not 100% (~21% in local repro), so some non-noop target indices are still emitted via pointer indices even when launch counts are zero.
- Decoder-carry sampler bug was fixed recently; this appears unrelated to carry mishandling.

**Critical unknown:** Whether env `valid` launch slots are also always zero (likely yes via `build_action_from_factored_batch` requiring `launch_fraction > 0` or `bucket > 0`).

**Discriminating probe:** Compare `transitions.ship_fraction` and `transitions.ship_bucket` with shield on vs off for identical seed; inspect `selected_ship_logits.shape[-1]` inside sampler for continuous config.

## Lane B — Config / Environment
**Hypothesis:** The sweep config composes a path that intentionally disables shield and uses continuous ship mode, exposing an unimplemented no-shield legality path.

**Evidence for:**
- `conf/task/default.yaml` sets `trajectory_shield_enabled: false`; sweep does not override it.
- Sweep selects `model=transformer_factorized` (factorized decoder, `decoder_carry: true`, `max_moves_k: 8`).
- `task.ship_action_mode: continuous_fraction` is active in default task config.

**Evidence against:**
- Shield-off is an explicit config choice, not accidental Hydra miscomposition.
- Training loop, curriculum, and opponent wiring appear unaffected; issue localizes to factorized action legality when shield disabled.

**Critical unknown:** Whether the user intended shield-off to mean "no trajectory filtering" vs "no legality filtering at all".

## Lane C — Measurement / Telemetry
**Hypothesis:** Metrics under-report launches due to aggregation or continuous-mode accounting mismatch.

**Evidence for:**
- `_apply_factorized_metrics` counts launches only when `ship_bucket > 0`, not `ship_fraction > 0` (`src/jax/rollout/metrics.py`).
- `_finalize_cross_chunk_rate_metrics` does not recompute `stop_rate` / `mean_active_launches_per_turn` after microbatch sum (would distort non-zero values, not create zeros from nothing).

**Evidence against:**
- With shield off, stored `ship_bucket` and `ship_fraction` are both zero in rollout transitions (root cause upstream).
- `entropy_*` metrics come from PPO update path, explaining divergence from rollout launch counters without telemetry bug.

**Critical unknown:** Whether M1 gate scripts should treat continuous mode launch detection differently even after sampler fix.

## Convergence (pre-interview)
Most likely explanation: **Lane A**, triggered by **Lane B** config (shield disabled + continuous_fraction). Entropy metrics change because PPO replays policy distributions; launch metrics stay zero because the sampler never produces legal launch actions when shield is off.

Recommended fix direction (for interview/planning, not executed here):
1. When shield disabled, emit legality masks that still allow real launches (all buckets or continuous-compatible mask).
2. Fix continuous/discrete ship head masking to avoid `(1,)` vs `(bucket_count,)` broadcast in sampler and PPO replay.
3. Optionally align launch metric with continuous mode (`ship_fraction > 0`).

## Local Verification
Command (single rollout, CPU/GPU JIT):
```
uv run python -c "..."  # compose_hydra_train_config with model=transformer_factorized, task.trajectory_shield_enabled=false
```
Observed: `mean_active_launches_per_turn=0.0`, `stop_rate≈0.33`, `source_mask any=False`, all `ship_fraction==0`.
