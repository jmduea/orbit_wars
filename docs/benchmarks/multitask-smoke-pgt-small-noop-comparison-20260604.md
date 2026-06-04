# PGT-small noop throughput comparison (2026-06-04)

## Baseline

- Run: `outputs/campaigns/multitask_smoke/runs/20260604T151112Z-s42-bed91b60`
- Snapshot: `multitask-smoke-pgt-small-noop-baseline-20260604.json`
- Update 20 (steady): **54.95** `env_steps_per_sec`, **55.21** `rollout_env_steps_per_sec`, **4.66s** `update_seconds`
- Steady mean updates 11–19: **63.98** `env_steps_per_sec` (compile skew on updates 1–10)

## After (noop opponent encode skip)

- Change: skip per-step and initial `encode_turn` for opponent cache when `opponents.mode.opponent` is `noop`/`no_op` in 2p rollouts (`collect_rollout_jax`).
- Benchmark: `ow benchmark training` with matching Hydra overrides (see `/tmp/pgt_small_benchmark_after.json`).
- Aggregate over 20 measured updates (warmup=2): **75.44** `env_steps_per_sec`, **3.39s** `seconds_per_update_mean`
- Delta vs baseline update-20 point: **+37%** env steps/s; vs steady u11–19 mean: **+18%**

## Caveats

- Baseline log per-update rate ≠ benchmark total-time aggregate (benchmark includes compile amortization differently).
- Same seed (42) and GPU; commit SHA differs (`5911320` at benchmark time).
