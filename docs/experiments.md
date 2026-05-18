# Running Orbit Wars experiments

This guide explains how to launch, evaluate, and compare Orbit Wars training
experiments from the checked-in configuration files. The repository supports two
independent backend choices:

- `env_backend`: `kaggle` for the Kaggle/Python environment, or `jax` for the
  fixed-shape JAX environment.
- `rl_backend`: `torch` for the existing Torch PPO loop, or `jax` for the
  end-to-end JAX PPO loop.

The default configuration keeps the established Kaggle/Torch training path.
Enable JAX explicitly in a YAML config or with a temporary copied config.

## Setup

Install dependencies with `uv` and run commands through the managed environment:

```bash
uv sync
```

All examples below assume commands are run from the repository root.

## Quick smoke runs

Use the default config for a short Kaggle/Torch run:

```bash
uv run python -m src.train --config default_cfg.yaml
```

For a short end-to-end JAX smoke run, copy `configs/jax_training.yaml` and lower
the rollout budget:

```yaml
# Keep both backend fields set to jax.
env_backend: jax
rl_backend: jax
ppo:
  rollout_steps: 64
  num_envs: 2
  total_updates: 10
```

Then launch it with:

```bash
uv run python -m src.train --config path/to/jax_smoke.yaml
```

The checked-in end-to-end JAX config can be launched directly with:

```bash
uv run python -m src.train --config configs/jax_training.yaml
```

For self-play plus conservative reward shaping on the end-to-end JAX stack, use:

```bash
uv run python -m src.train --config configs/jax_self_play_shaped_reward_training.yaml
```

## Reproducible training configs

The `configs/` directory contains longer-running experiment presets:

| Config | Purpose |
| --- | --- |
| `configs/full_training.yaml` | MLP baseline using the standard PPO budget. |
| `configs/attention_training.yaml` | Attention-policy baseline. |
| `configs/attention_shaped_reward_training.yaml` | Attention policy with conservative reward shaping. |
| `configs/attention_self_play_pool.yaml` | Attention policy trained against the self-play opponent pool. |
| `configs/attention_candidates_16.yaml` | Attention policy with 15 real target slots plus no-op. |
| `configs/attention_candidates_24.yaml` | Attention policy with 23 real target slots plus no-op. |
| `configs/jax_training.yaml` | End-to-end JAX environment plus JAX PPO training. |
| `configs/jax_self_play_shaped_reward_training.yaml` | JAX self-play with conservative reward shaping. |

Launch any preset with:

```bash
uv run python -m src.train --config configs/attention_training.yaml
```

To make runs easier to compare, keep the seed, PPO budget, opponent settings,
and evaluation seed range fixed when comparing one experimental variable.

## Logs and checkpoints

Torch PPO writes JSONL metrics to:

```text
artifacts/rl_template/logs/<run_name>.jsonl
```

JAX PPO writes JSONL metrics to:

```text
artifacts/rl_template/logs/<run_name>_jax.jsonl
```

Checkpoints are saved under:

```text
<save_dir>/<run_name>/
```

Torch checkpoints use `ckpt_last.pt` and numbered `ckpt_*.pt` files. JAX
checkpoints use `jax_ckpt_last.pkl` and numbered `jax_ckpt_*.pkl` files.

## Resuming training from a checkpoint

Use `--resume-checkpoint` with the same config family that created the checkpoint.
The config's `ppo.total_updates` is interpreted as the final update to run, not
as an additional number of updates. For example, resuming a JAX run from update
50 with `total_updates: 2000` starts at update 51 and stops after update 2000:

```bash
uv run python -m src.train \
  --config configs/jax_training.yaml \
  --resume-checkpoint /artifacts/jax_training/orbit_wars_ppo_jax_training/jax_ckpt_000050.pkl
```

The same flag works for Torch PPO checkpoints when using a Torch training config:

```bash
uv run python -m src.train \
  --config configs/attention_training.yaml \
  --resume-checkpoint /artifacts/attention_training/orbit_wars_ppo_attention_training/ckpt_000050.pt
```

When resuming JAX checkpoints produced by older code that did not include Optax
optimizer state or RNG state, training still loads the policy parameters and
reinitializes any missing state from the current config. New JAX checkpoints save
policy parameters, optimizer state, RNG key, update number, environment-step
counter, and completed-episode counter.

## Evaluation protocol

Evaluate Torch checkpoints against a fixed benchmark set with identical seeds for
every checkpoint:

```bash
uv run python evaluate.py \
  --config configs/attention_training.yaml \
  --checkpoint /artifacts/attention_training/orbit_wars_ppo_attention_training/ckpt_last.pt \
  --games 100 \
  --opponents sniper,random,self_play_snapshot \
  --seeds 0:99 \
  --deterministic \
  --run-name attention_training_ckpt_last
```

When comparing shaped vs. unshaped rewards, candidate counts, or self-play
settings, keep `--games`, `--opponents`, and `--seeds` unchanged. This makes the
reported win rates and rewards directly comparable across checkpoints.

## Benchmarking environment and JAX throughput

Compare Kaggle/Python and JAX environment stepping with:

```bash
uv run python scripts/benchmark_env.py --backend both --rollout-steps 200 --num-envs 8
```

Measure the end-to-end JAX rollout plus PPO update stack with:

```bash
uv run python scripts/benchmark_jax_rl.py \
  --updates 10 \
  --rollout-steps 128 \
  --num-envs 16 \
  --architecture attention
```

The first JAX update includes compilation overhead. For throughput comparisons,
run enough updates to separate compile time from steady-state execution.

## Configuration tips

- `model.architecture` accepts `mlp`, `attention`, and `transformer`. The
  `transformer` keyword is an alias for the attention implementation.
- `env.candidate_count` includes no-op slot `0`; the number of real targets is
  `candidate_count - 1`.
- `env.max_planets` and `env.max_fleets` control fixed JAX array shapes. Larger
  values support more game objects but increase compile time and memory use.
- The JAX PPO path currently supports `opponent: self` and `opponent: random`.
  Use the Kaggle/Torch path for opponents that depend on the Kaggle observation
  API, such as sniper benchmark play.
- Use a new `run_name` for every experiment you want to preserve separately.
