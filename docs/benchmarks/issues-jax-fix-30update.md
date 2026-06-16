# Issues.md JAX fix — 30-update GPU benchmark

## Commits

| Label | Commit / tree |
|-------|----------------|
| Baseline (micro) | `e7f8e4c2c45881e0e9db6c81f172b5800605e74f` — 16-env micro profile |
| Post-fix (micro) | archived `docs/benchmarks/postfix-30update.json` |
| **Workstation production** | `dcafdc827a9202152b27e3676edfa9e715ecc0f7` — `docs/benchmarks/workstation-30update.json` |

## CUDA / JAX environment

| Item | Value |
|------|--------|
| GPU | NVIDIA GeForce RTX 5080 (driver 610.x, CUDA UMD 13.3) |
| `nvidia-smi` | OK |
| JAX | `0.10.0` with `jax[cuda13]` in `pyproject.toml` (linux x86_64) |
| `jax.devices()` | `[CudaDevice(id=0)]`, default backend `gpu` |

**WSL note:** `pin_jax_platform_from_kaggle()` may set `JAX_PLATFORMS=cpu` when `/dev/nvidia*` is absent. Local benchmarks call `configure_jax_runtime_for_host()` and force `cuda,cpu` when `nvidia_gpu_present()` before importing JAX.

## Benchmark protocol

- Script: `ow benchmark training` (`src/benchmark/training.py` presets)
- 30 measured updates after 2 warmup (JIT compile counted through update 3)
- JSON includes `compile_seconds_to_update_3`, `rollout_seconds_mean`, `update_seconds_mean`, `tier`, `format`, `rollout_microbatch_envs`, `rollout_groups`

### Micro profile (historical comparability)

Hydra overrides: `model=transformer_factorized`, `opponents=self_play_only`, `seed=42`, `format=2p_4p_8env`, `training.rollout_steps=16`, `training.rollout_microbatch_envs=8` — **16 total envs**, short rollouts.

Artifact: `docs/benchmarks/baseline-30update.json`

### Workstation production profile (primary gate)

```bash
uv run ow benchmark training \
  --label workstation-production-30u \
  --tier workstation \
  --out docs/benchmarks/workstation-30update.json \
  --overrides \
    model=transformer_factorized \
    format=2p_4p_16env \
    training=workstation \
    opponents=self_play_curriculum \
    curriculum=self_play_staged \
    telemetry.wandb.enabled=false \
    artifacts.artifact_pipeline.enabled=false \
    seed=42
```

| Field | Workstation production |
|-------|----------------------|
| `num_envs` | 32 (16×2p + 16×4p) |
| `rollout_steps` | 128 |
| `rollout_microbatch_envs` | 16 |
| `compile_seconds_to_update_3` | **221.3 s** |
| `seconds_per_update_mean` | **0.703 s** |
| `rollout_seconds_mean` | 0.585 s |
| `update_seconds_mean` | 0.118 s |
| `env_steps_per_sec` | **5826** |
| `mean_active_launches_per_turn` | 0.475 |
| `overall_win_rate` | 0.319 |
| `approx_kl` | 0.00035 |

## Micro vs workstation (not apples-to-apples)

| Metric | Baseline micro (`baseline-30update.json`) | Workstation production |
|--------|------------------------------------------|------------------------|
| Total envs | 16 | 32 |
| `rollout_steps` | 16 | 128 |
| Curriculum | default `sp_2p` | `self_play_staged` |
| Opponents | `self_play_only` | `self_play_curriculum` |
| `seconds_per_update_mean` | 0.300 | 0.703 |
| `env_steps_per_sec` | 854 | 5826 |
| `compile_seconds_to_update_3` | (in warmup) | 221.3 |

Workstation uses **8× longer rollouts** and **2× env parallelism**; higher env-steps/sec is expected. Use this table for throughput regression on the **same** profile, not to compare against the 16-env micro run.

## Cloud stretch tier (documented, not run here)

| Tier | Profile | Where |
|------|---------|-------|
| **Colab / large VRAM** | `format=2p_4p_32env` (64 envs), same training/opponents/curriculum | Run when GPU headroom allows; OOM → bisect `training.rollout_microbatch_envs` (32 → 16) |

Alias: `format=mix_2p_4p_32env` composes `2p_4p_32env` (deprecated `mix_*` naming).

## Tests (worktree `dcafdc8`)

```bash
make test-domain-config
uv run --group dev pytest tests/test_kaggle_jax_backend.py tests/test_kaggle_wandb_population.py -m "not slow and not jax"
make test-fast
```

**228 passed** (`make test-fast`), domain-config green after Hydra hygiene fixes.

## Not in this milestone

- Dynamic minibatch row-id gather
- `value_only` policy path
- Mixed precision (advisory only)

See `docs/benchmarks/issues-jax-fix-verification.md` for issue-by-issue code verification.
