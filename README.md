# orbit_wars

## Reinforcement learning tutorial implementation

The Orbit Wars reinforcement-learning implementation that was previously generated from
`orbit-wars-reinforcement-learning-tutorial.ipynb` now lives as versioned repository files:

- `default_cfg.yaml` for quick notebook/demo runs
- `configs/full_training.yaml` for longer reproducible MLP baseline training runs
- `configs/attention_training.yaml` for longer reproducible attention-policy training runs
- `configs/attention_shaped_reward_training.yaml` for attention-policy training with conservative reward shaping
- `configs/attention_self_play_pool.yaml` for attention training against a self-play opponent pool
- `configs/jax_training.yaml` for end-to-end JAX environment plus JAX PPO training
- `configs/jax_self_play_shaped_reward_training.yaml` for JAX self-play training with conservative reward shaping
- `src/`
- `evaluate.py` for checkpoint evaluation across multiple opponents
- `eval_vs_sniper.py` as a backwards-compatible sniper-only wrapper
- `play_vs_sniper.py`

The notebook should be treated as a tutorial wrapper around this checked-in implementation.
For code changes, update the repository files first; do not treat notebook `%%writefile`
cells as the canonical source of the implementation.

For launch commands, backend selection, logging locations, checkpoint naming,
evaluation protocol, and benchmark usage, see [`docs/experiments.md`](docs/experiments.md).

## Attention self-play-pool experiment

Train the attention policy against the self-play opponent pool with:

```bash
uv run python -m src.train --config configs/attention_self_play_pool.yaml
```

Evaluate each checkpoint against the fixed benchmark set instead of only the
current training opponent. Use the same seed range for every checkpoint so the
`sniper`, `random`, and `self_play_snapshot` results are directly comparable:

```bash
uv run python evaluate.py \
  --config configs/attention_self_play_pool.yaml \
  --checkpoint /artifacts/attention_self_play_pool/orbit_wars_ppo_attention_self_play_pool/ckpt_000050.pt \
  --games 100 \
  --opponents sniper,random,self_play_snapshot \
  --seeds 0:99 \
  --deterministic \
  --run-name attention_self_play_pool_ckpt_000050
```

Repeat the command for later `ckpt_*.pt` files while keeping `--games`,
`--opponents`, and `--seeds` unchanged.

## Dependency management

This repository uses [`uv`](https://docs.astral.sh/uv/) for Python dependency management.
Install the runtime dependencies into a local `.venv` with:

```bash
uv sync
```

On Linux x86_64, the project depends on `jax[cuda13]` so `uv sync` installs
JAX's CUDA 13 plugin instead of a CPU-only JAX stack. The JAX training and
benchmark entry points fail fast when NVIDIA hardware is visible but JAX only
initializes CPU devices; set `ORBIT_WARS_ALLOW_CPU_JAX_ON_NVIDIA=1` only when
you intentionally want to debug on CPU. Non-Linux or non-x86_64 machines keep
the standard CPU JAX dependency.

Run the extracted package and scripts through `uv run`, for example:

```bash
uv run python -m src.train --config default_cfg.yaml
uv run python -m src.train --config configs/full_training.yaml
uv run python -m src.train --config configs/attention_training.yaml
uv run python -m src.train --config configs/attention_shaped_reward_training.yaml
uv run python -m src.train --config configs/attention_self_play_pool.yaml
uv run python -m src.train --config configs/jax_training.yaml
uv run python -m src.train --config configs/jax_self_play_shaped_reward_training.yaml

uv run python -m src.train --config configs/jax_entity_transformer_500k.yaml
uv run python -m src.train --config configs/jax_entity_transformer_700k.yaml
uv run python -m src.train --config configs/jax_entity_transformer_1m.yaml
uv run python evaluate.py --config default_cfg.yaml --games 100 --opponents sniper,random,self_play_snapshot --seeds 0:99 --deterministic
uv run python eval_vs_sniper.py --config default_cfg.yaml --deterministic
uv run python play_vs_sniper.py --config default_cfg.yaml --deterministic --output result.html
```

## JAX environment/RL training

Launch the end-to-end JAX backend with the checked-in JAX config:

```bash
uv run python -m src.train --config configs/jax_training.yaml
```

This config sets both `env_backend: jax` and `rl_backend: jax`, writes metrics to
`artifacts/rl_template/logs/orbit_wars_ppo_jax_training_jax.jsonl`, and saves
checkpoints under `/artifacts/jax_training/orbit_wars_ppo_jax_training/` as
`jax_ckpt_last.pkl` and numbered `jax_ckpt_*.pkl` files. Resume from any saved
JAX checkpoint with:

```bash
uv run python -m src.train \
  --config configs/jax_training.yaml \
  --resume-checkpoint /artifacts/jax_training/orbit_wars_ppo_jax_training/jax_ckpt_000050.pkl
```

`ppo.total_updates` remains the final target update number, so resuming from
`jax_ckpt_000050.pkl` with `total_updates: 2000` continues at update 51 and stops
after update 2000.

For JAX self-play with the same conservative reward-shaping values used by the
attention shaped-reward experiment, launch:

```bash
uv run python -m src.train --config configs/jax_self_play_shaped_reward_training.yaml
```

The JAX self-play shaped config keeps `env_backend: jax`, `rl_backend: jax`, and
`opponent: self`, sets `self_play_enabled: true` to identify the experiment, and
uses `reward_capture_planet: 0.1`, `reward_ship_delta: 0.001`,
`reward_production_delta: 0.02`, and `reward_terminal_scale: 1.0`.

## Shaped-reward attention experiment

Train the conservative shaped-reward attention run with the same PPO budget as
`configs/attention_training.yaml`:

```bash
uv run python -m src.train --config configs/attention_shaped_reward_training.yaml
```

Compare it against the unshaped attention run by evaluating both checkpoints
with identical opponents and seeds:

```bash
uv run python evaluate.py \
  --config configs/attention_training.yaml \
  --checkpoint /artifacts/attention_training/orbit_wars_ppo_attention_training/ckpt_last.pt \
  --games 100 \
  --opponents sniper,random,self_play_snapshot \
  --seeds 0:99 \
  --deterministic \
  --run-name attention_unshaped_ckpt_last

uv run python evaluate.py \
  --config configs/attention_shaped_reward_training.yaml \
  --checkpoint /artifacts/attention_shaped_reward_training/orbit_wars_ppo_attention_shaped_reward/ckpt_last.pt \
  --games 100 \
  --opponents sniper,random,self_play_snapshot \
  --seeds 0:99 \
  --deterministic \
  --run-name attention_shaped_reward_ckpt_last
```

Keep `--games`, `--opponents`, and `--seeds` unchanged for any earlier
checkpoint pair so the shaped and unshaped metrics remain directly comparable.

## Attention candidate-count experiments

Candidate index `0` is reserved for the no-op action, so an `env.candidate_count`
of `8`, `16`, or `24` gives the policy `7`, `15`, or `23` real target slots.
The attention configs below keep the same seed and PPO settings so their logs can
be compared directly:

```bash
uv run python -m src.train --config configs/attention_training.yaml
uv run python -m src.train --config configs/attention_candidates_16.yaml
uv run python -m src.train --config configs/attention_candidates_24.yaml
uv run python scripts/compare_attention_candidates.py
```

Training logs include `candidate_valid_avg`, `candidate_enemy_share`,
`candidate_neutral_share`, and `candidate_friendly_share` to diagnose whether the
candidate builder is giving the policy enough real targets from each ownership
class.


## JAX entity-transformer size sweep (≈500k / ≈700k / ≈1M params)

Use these configs to run end-to-end JAX training with progressively larger
entity-transformer policies:

- `configs/jax_entity_transformer_500k.yaml` (`hidden_size: 192`, `attention_heads: 6`)
- `configs/jax_entity_transformer_700k.yaml` (`hidden_size: 224`, `attention_heads: 7`)
- `configs/jax_entity_transformer_1m.yaml` (`hidden_size: 272`, `attention_heads: 8`)

Train each run with:

```bash
uv run python -m src.train --config configs/jax_entity_transformer_500k.yaml
uv run python -m src.train --config configs/jax_entity_transformer_700k.yaml
uv run python -m src.train --config configs/jax_entity_transformer_1m.yaml
```

All three are configured for the mixed 2p/4p self-play curriculum from
`configs/jax_mixed_2p_4p_training.yaml` (equal 2p/4p mix, 32 total envs,
rollout length 500) so comparisons isolate model size changes under the target
throughput profile.
