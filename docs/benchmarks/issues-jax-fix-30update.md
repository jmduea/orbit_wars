# Issues.md JAX fix — 30-update GPU benchmark

## Commits

| Label | Commit / tree |
|-------|----------------|
| Baseline | `e7f8e4c2c45881e0e9db6c81f172b5800605e74f` (clean tree before Issues.md fixes) |
| Post-fix | worktree `issues-jax-14f07ba2` (uncommitted JAX PPO fixes on same base) |

## CUDA / JAX environment

| Item | Value |
|------|--------|
| GPU | NVIDIA GeForce RTX 5080 (driver 610.x, CUDA UMD 13.3) |
| `nvidia-smi` | OK |
| JAX | `0.10.0` with `jax[cuda13]` in `pyproject.toml` (linux x86_64) |
| `jax.devices()` | `[CudaDevice(id=0)]`, default backend `gpu` |

**Root cause of prior “no CUDA” runs:** WSL2 has no `/dev/nvidia*` nodes, so `pin_jax_platform_from_kaggle()` set `JAX_PLATFORMS=cpu` unless overridden. `configure_jax_runtime_for_host()` only pinned `cuda,cpu` for Kaggle `KAGGLE_ACCELERATOR_ID=nvidia*`. **Fix:** pin `cuda,cpu` whenever `nvidia_gpu_present()` (see `src/jax/device.py`).

**Dependency:** `jax[cuda13]` already declared; `uv sync --group dev` in the worktree is sufficient (no extra `uv add` required on this host).

## Benchmark protocol

- Script: `scripts/issues_jax_30update_benchmark.py`
- Hydra overrides: `model=transformer_factorized`, `opponents=self_play_only`, `seed=42`, `format=2p_4p_8env`, `training.rollout_steps=16`, `training.rollout_microbatch_envs=8`
- 30 measured updates after 2 warmup (JIT compile included in warmup)
- **Note:** Default `format` (32+32 envs) OOMs on 16 GB GPU during XLA compile; paired runs use 16 total envs (8+8) for comparability.

Artifacts: `docs/benchmarks/baseline-30update.json`, `docs/benchmarks/postfix-30update.json`

## Before / after (GPU)

| Metric | Baseline | Post-fix | Δ |
|--------|----------|----------|---|
| Wall-clock total (30 updates, s) | 8.99 | 6.94 | **−22.9%** |
| Mean s / update | 0.300 | 0.231 | **−22.9%** |
| Env steps / s | 854.1 | 1107.0 | **+29.6%** |
| `policy_loss` | −0.00113 | 0.02592 | — (correctness change) |
| `value_loss` | 0.4010 | 0.4160 | +3.8% |
| `approx_kl` | 0.000194 | 0.000862 | +344% (still tiny) |
| `entropy` | 3.614 | 4.528 | +25.3% |
| `total_loss` | 0.1813 | 0.2113 | +16.5% |
| `mean_active_launches_per_turn` | 1.490 | 2.520 | +69.2% |

Post-fix includes: continuous ship `log p(fraction|μ)`, single-pass factorized replay (one `policy.apply` per replay), gated parity metrics, scalar GAE storage, sparse C51 CE, WSL CUDA platform pin.

**Interpretation:** Throughput improves ~23–30% on this GPU with fewer decoder forwards. Loss magnitudes and launch activity shift as expected when the ship head receives PPO gradients; not a numeric match to baseline.

## Tests (post-fix worktree)

```bash
uv run --group dev pytest tests/test_factored_sequence_scan.py \
  tests/test_distributional_value.py tests/test_action_codec.py -m "not slow"
```

**17 passed, 1 failed** — `test_compose_hydra_train_config_accepts_distributional_value_head` (missing `opponents/self_play_curriculum` in default Hydra group; pre-existing worktree config gap, unrelated to CUDA).

## Not in this milestone

- Dynamic minibatch row-id gather
- `value_only` policy path
- Mixed precision (advisory only)

See `docs/benchmarks/issues-jax-fix-verification.md` for issue-by-issue code verification.
