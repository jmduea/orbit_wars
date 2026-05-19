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

The JAX trainer uses **Option A: separate compiled rollout/update functions per
format** for mixed 2-player/4-player experiments. Each configured rollout group
gets its own environment state, turn batch, and statically compiled collector
(`player_count: 2` or `player_count: 4`). The trainer collects those groups
independently, concatenates their compatible transition tensors along the
environment axis, and then runs a shared PPO update on the combined batch. This
avoids recompilation or shape errors from switching a single jitted collector
between player formats while preserving one policy and optimizer.

To exercise both formats in one run, keep `training_format.rollout_groups`
declared with separate 2p and 4p entries, as in `configs/jax_training.yaml` and
`configs/jax_mixed_2p_4p_training.yaml`:

```yaml
training_format:
  rollout_groups:
    - name: two_player
      player_count: 2
      num_envs: 4
    - name: four_player
      player_count: 4
      num_envs: 4
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

| `configs/jax_entity_transformer_500k.yaml` | JAX entity transformer sweep point (~500k params). |
| `configs/jax_entity_transformer_700k.yaml` | JAX entity transformer sweep point (~700k params). |
| `configs/jax_entity_transformer_1m.yaml` | JAX entity transformer sweep point (~1M params). |
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

Mixed-format checkpoints should be evaluated in both canonical Orbit Wars match
formats. Use `--formats 2p,4p` (equivalently `--player-counts 2,4`) so the
evaluation runner calls Kaggle with the matching player count and reports
format-specific summaries:

```bash
uv run python evaluate.py \
  --config configs/attention_training.yaml \
  --checkpoint /artifacts/attention_training/orbit_wars_ppo_attention_training/ckpt_last.pt \
  --games 100 \
  --opponents sniper,random,self_play_snapshot \
  --formats 2p,4p \
  --learner-seats all \
  --seeds 0:99 \
  --deterministic \
  --run-name attention_training_ckpt_last_mixed_formats
```

The 2-player summary reports `win_rate_2p`. The 4-player summary constructs
three opponent slots for each game and reports `first_place_rate_4p`,
`average_placement_4p`, and `per_seat` metrics for each learner seat. Keep
`--learner-seats all` for canonical reports; use a comma-separated subset such
as `--learner-seats 0,2` only for targeted debugging.

When comparing shaped vs. unshaped rewards, candidate counts, or self-play
settings, keep `--games`, `--opponents`, and `--seeds` unchanged. This makes the
reported win rates and rewards directly comparable across checkpoints.


## Hydra multirun sweep recipes

Hydra sweeps are the easiest way to run controlled ablations while keeping one
canonical base config. The examples below use:

- `-m` (Hydra multirun) to launch one run per value combination.
- A fixed experiment backbone (for example `experiment=jax_training`) so only
  one independent variable changes.
- Explicit `seed`, `--games`, `--opponents`, and `--seeds` settings during
  evaluation to keep comparisons fair.

### 1) Candidate-count sweep (`8/16/24`)

```bash
uv run python -m src.train -m \
  experiment=attention_training \
  env.candidate_count=8,16,24 \
  run_name=sweep_candidates
```

### 2) Shaped vs unshaped reward sweep

```bash
uv run python -m src.train -m \
  experiment=attention_training \
  env.reward_capture_planet=0.0,0.02 \
  env.reward_ship_delta=0.0,0.01 \
  env.reward_production_delta=0.0,0.005 \
  run_name=sweep_reward_shape
```

Tip: keep terminal objective settings unchanged (`env.terminal_reward_mode` and
`env.reward_terminal_scale`) so the shaped/unshaped comparison isolates only the
dense shaping terms.

### 3) Model-size sweep (`500k/700k/1m`)

You can sweep by experiment name (recommended because each file already pins a
known architecture width/head combination):

```bash
uv run python -m src.train -m \
  experiment=jax_entity_transformer_500k,jax_entity_transformer_700k,jax_entity_transformer_1m \
  run_name=sweep_model_size
```

### 4) Mixed 2p/4p format settings sweep

For JAX mixed-format training, sweep only one ratio at a time while keeping
per-format rollout groups fixed:

```bash
uv run python -m src.train -m \
  experiment=jax_mixed_2p_4p_training \
  training_format.format_mix='[{player_count:2,weight:0.75},{player_count:4,weight:0.25}]','[{player_count:2,weight:0.5},{player_count:4,weight:0.5}]','[{player_count:2,weight:0.25},{player_count:4,weight:0.75}]' \
  run_name=sweep_format_mix
```

### Keep sweep outputs comparable (naming + directory convention)

Use a stable naming pattern that encodes the sweep family and let Hydra append
parameter overrides:

- Base `run_name`: `sweep_<factor>` (for example `sweep_candidates`).
- Output root per study: `save_dir=artifacts/sweeps/<family>`
- Include date only if running repeated studies, for example
  `save_dir=artifacts/sweeps/candidates_2026-05-19`.

Recommended multirun invocation pattern:

```bash
uv run python -m src.train -m \
  experiment=attention_training \
  env.candidate_count=8,16,24 \
  run_name=sweep_candidates \
  save_dir=artifacts/sweeps/candidates
```

## Optional Hydra sweeper plugins

The repository works with Hydra's default basic sweeper out of the box. If you
want a checked-in config stub, use `conf/hydra/sweeper/basic.yaml` and include
it from your active config defaults.

```yaml
# conf/hydra/sweeper/basic.yaml
_target_: hydra._internal.core_plugins.basic_sweeper.BasicSweeper
max_batch_size: null
params: {}
```

Later, you can optionally switch to search plugins such as Optuna or Ax once
those dependencies are installed in your environment:

- Optuna plugin target: `hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper`
- Ax plugin target: `hydra_plugins.hydra_ax_sweeper.ax_sweeper.AxSweeper`

Keep the same naming convention (`run_name`, `save_dir`, fixed eval seeds) when
moving from grid sweeps to adaptive sweeps so historical results remain
comparable.

## Fair-evaluation checklist for any sweep

For every checkpoint family being compared, keep all of the following fixed:

1. **Training seed:** set explicit `seed=<N>` in every sweep launch.
2. **Opponent recipe:** same `opponent`, `opponent_mix`, and self-play settings.
3. **Training budget:** same `ppo.total_updates`, `ppo.rollout_steps`, and
   effective environment count.
4. **Evaluation games:** same `--games` value for all checkpoints.
5. **Evaluation opponents:** identical `--opponents` list and ordering.
6. **Evaluation seeds:** identical fixed range such as `--seeds 0:99`.
7. **Evaluation formats:** identical `--formats` and `--learner-seats` policy.

Example fair-eval command template:

```bash
uv run python evaluate.py \
  --config configs/attention_training.yaml \
  --checkpoint <checkpoint_path> \
  --games 100 \
  --opponents sniper,random,self_play_snapshot \
  --formats 2p,4p \
  --learner-seats all \
  --seeds 0:99 \
  --deterministic \
  --run-name <eval_run_name>
```

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


## JAX entity-transformer size sweep

To reproduce a model-size sweep near 500k, 700k, and 1M parameters in the JAX
training stack, run:

```bash
uv run python -m src.train --config configs/jax_entity_transformer_500k.yaml
uv run python -m src.train --config configs/jax_entity_transformer_700k.yaml
uv run python -m src.train --config configs/jax_entity_transformer_1m.yaml
```

These configs all use `env_backend: jax` + `rl_backend: jax`, enable self-play,
use the mixed 2p/4p curriculum (`format_mix` 50/50), and set `ppo.num_envs: 32`
with `ppo.rollout_steps: 500`. They only vary model width/head count so results
are directly comparable at the target throughput settings.


## Complete tweakable configuration reference

The fields below are the complete set of user-tweakable training config values
from `src/config.py` (top-level and nested dataclasses). Recommended defaults
are the repository defaults unless noted otherwise.

### Top-level `TrainConfig`

| Key | Default | Recommended default | What it controls | Why this default is recommended |
| --- | --- | --- | --- | --- |
| `seed` | `42` | `42` | Global RNG seed for training/eval reproducibility. | Stable run-to-run comparisons require deterministic seeds. |
| `run_name` | `orbit_wars_template_ppo` | Unique per experiment, e.g. `attention_ablation_a` | Name prefix for logs/checkpoints. | Prevents overwriting prior runs and makes artifact lineage obvious. |
| `device` | `auto` | `auto` | Torch device selection policy. | Uses GPU automatically when available without hardcoding platform-specific values. |
| `save_dir` | `artifacts/rl_template` | Keep default, or one folder per study | Root output directory for checkpoints/metrics. | Predictable artifact layout simplifies resume/eval scripts. |
| `checkpoint_every` | `10` | `10` | Save interval in updates. | Balances recovery safety with I/O overhead for typical PPO budgets. |
| `log_every` | `1` | `1` | Metrics logging cadence in updates. | Per-update logging makes debugging learning instabilities much easier. |
| `opponent` | `random` | `random` (baseline) or `self` (self-play studies) | Primary training opponent policy selection. | `random` gives a low-variance baseline; `self` is best for co-evolution studies. |
| `env_backend` | `kaggle` | `kaggle` for legacy/Torch, `jax` for end-to-end JAX | Environment implementation backend. | Explicit backend choice avoids accidental mixed-stack behavior. |
| `rl_backend` | `torch` | `torch` for baseline, `jax` for JAX throughput studies | RL training loop backend. | Keeps default path robust while allowing explicit JAX experiments. |
| `self_play_update_interval` | `10` | `10` | Update cadence for refreshing self-play opponent state. | Frequent enough to track learner improvement without excessive churn. |
| `self_play_deterministic` | `False` | `False` | Whether self-play action selection is deterministic. | Stochastic opponents improve robustness and reduce exploit overfitting. |
| `self_play_enabled` | `False` | `False` unless self-play experiment | Master switch for self-play flow. | Keeps baseline behavior simple; enable only when intentionally studying self-play. |
| `self_play_pool_size` | `5` | `5` | Number of historical snapshots retained in pool. | Provides diversity without large memory/management overhead. |
| `self_play_snapshot_interval` | `25` | `25` | How often learner snapshots are added to pool. | Captures policy progression while limiting near-duplicate snapshots. |
| `self_play_latest_probability` | `0.5` | `0.5` | Probability of sampling latest policy vs pool sample. | Even split balances recency pressure and anti-forgetting diversity. |
| `multi_opponent_mode` | `mixed` | `mixed` | Strategy for selecting opponents in multi-opponent setups. | Mixed sampling reduces over-specialization to a single opponent style. |
| `alternate_player_sides` | `True` | `True` | Rotates learner seat/side assignment. | Reduces seat-bias confounding in win-rate measurements. |
| `reseed_every_updates` | `0` | `0` | Periodic RNG reseed interval (`0` disables). | Fixed seed stream is best for controlled ablations unless exploring variance. |
| `reseed_on_plateau` | `False` | `False` | Whether to reseed when plateau detector triggers. | Avoids hiding optimization issues behind stochastic resets. |
| `plateau_metric` | `episode_reward_mean` | `episode_reward_mean` | Metric monitored for plateau logic. | Reward mean is always present and aligns with training objective. |
| `plateau_window` | `10` | `10` | Window length for plateau detection. | Short enough to react, long enough to smooth noisy updates. |
| `plateau_delta` | `0.0` | `0.0` | Minimum improvement threshold for plateau checks. | Zero threshold keeps logic simple; tune only if metric noise causes false positives. |
| `heldout_eval_seed_set` | `[]` | Fixed explicit set for paper-grade evals (e.g. `0..99`) | Optional fixed seeds reserved for held-out evaluation. | Enforces reproducible benchmark slices and prevents train/eval leakage. |

### `env` (`EnvConfig`)

| Key | Default | Recommended default | What it controls | Why this default is recommended |
| --- | --- | --- | --- | --- |
| `env.board_size` | `100.0` | `100.0` | Spatial map scale for positions/distances. | Matches tuned game dynamics and travel-time assumptions in existing configs. |
| `env.episode_steps` | `500` | `500` | Max steps before forced terminal. | Allows comebacks and strategic depth without unbounded episodes. |
| `env.candidate_count` | `8` | `8` (or explicit sweep: `16`, `24`) | Action candidate slots (including no-op at slot 0). | Good baseline action branching; larger counts increase compute and exploration burden. |
| `env.ship_bucket_count` | `8` | `8` | Discretization buckets for ship-related features. | Reasonable resolution with manageable observation dimensionality. |
| `env.max_planets` | `48` | `48` | Fixed JAX shape bound for planets. | Covers typical maps while controlling compile time and memory footprint. |
| `env.max_fleets` | `256` | `256` | Fixed JAX shape bound for fleets. | Prevents frequent truncation while remaining practical for accelerator memory. |
| `env.player_count` | `2` | `2` unless explicitly training/evaluating 4p | Default format player count. | 2p is simplest controlled baseline and fastest per environment step. |
| `env.ship_speed` | `6.0` | `6.0` | Movement speed scaling for fleets. | Preserves established pacing for combat and expansion timing. |
| `env.max_ships` | `400.0` | `400.0` | Normalization/clamp scale for ship counts. | Keeps feature magnitudes in stable range for policy/value learning. |
| `env.max_production` | `5.0` | `5.0` | Normalization scale for planet production rates. | Aligns observation scales with common map generation ranges. |
| `env.reward_capture_planet` | `0.0` | `0.0` (baseline), small positive for shaping studies | Dense reward bonus for captures. | Zero preserves sparse-objective baseline; nonzero can speed early learning but biases behavior. |
| `env.reward_ship_delta` | `0.0` | `0.0` (baseline), conservative values for shaping studies | Dense reward on ship differential changes. | Avoids reward hacking in baseline; conservative shaping reduces distortion risk. |
| `env.reward_production_delta` | `0.0` | `0.0` (baseline), conservative values for shaping studies | Dense reward on production differential changes. | Same rationale as ship-delta shaping: use only in controlled ablations. |
| `env.reward_terminal_scale` | `1.0` | `1.0` | Multiplier on terminal outcome reward. | Keeps win/loss target dominant without destabilizing value scale. |
| `env.terminal_reward_mode` | `binary_win` | `binary_win` | Terminal reward schema. | Binary objective best matches win-rate optimization and cross-run comparability. |

### `model` (`ModelConfig`)

| Key | Default | Recommended default | What it controls | Why this default is recommended |
| --- | --- | --- | --- | --- |
| `model.architecture` | `mlp` | `mlp` baseline; `attention` for stronger policy studies | Policy/value network architecture family. | MLP is fast/control baseline; attention better handles entity interactions. |
| `model.hidden_size` | `128` | `128` baseline (increase only with matching compute budget) | Core hidden width across supported architectures. | Good quality/throughput trade-off; larger widths can overfit or slow updates. |
| `model.attention_heads` | `4` | `4` | Number of attention heads for attention models. | Sufficient relational capacity without significant kernel overhead. |
| `model.normalize_observations` | `True` | `True` | Enables running observation normalization. | Strongly improves optimization stability across mixed feature scales. |
| `model.obs_norm_clip` | `10.0` | `10.0` | Clip magnitude for normalized observations. | Guards against outliers without excessively flattening informative variation. |

### `ppo` (`PPOConfig`)

| Key | Default | Recommended default | What it controls | Why this default is recommended |
| --- | --- | --- | --- | --- |
| `ppo.rollout_steps` | `32` | `32` baseline; larger (e.g. `128-500`) only with larger batches | Steps collected per env before update. | Baseline keeps iteration latency low and stabilizes debugging. |
| `ppo.num_envs` | `4` | `4` baseline; scale up on stronger hardware | Number of parallel environments. | Small default works broadly; higher values improve throughput if memory allows. |
| `ppo.num_envs_2p` | `null` | `null` unless using format-specific overrides | Optional 2p env-count override. | Avoids hidden asymmetry unless explicitly needed for mixed-format balancing. |
| `ppo.num_envs_4p` | `null` | `null` unless using format-specific overrides | Optional 4p env-count override. | Same rationale as `num_envs_2p`. |
| `ppo.rollout_groups` | `[]` | `[]` unless using explicit grouped collection | Optional per-group rollout allocation metadata. | Keep empty for simple runs; define explicitly for mixed-format JAX collectors. |
| `ppo.phases` | `[]` | `[]` unless curriculuming PPO hyperparams | Optional PPO phase schedule. | Flat hyperparams are easier to compare and reproduce. |
| `ppo.total_updates` | `200` | `200` smoke/baseline; increase for long runs | Number of PPO updates to run. | Useful quick default for CI/smoke; longer studies should set explicit larger budgets. |
| `ppo.epochs` | `4` | `4` | Optimization epochs per rollout batch. | Standard PPO setting balancing sample efficiency and overfitting risk. |
| `ppo.minibatch_size` | `512` | `512` | Minibatch size for PPO optimization passes. | Works with default batch sizes and keeps gradient noise moderate. |
| `ppo.gamma` | `0.99` | `0.99` | Discount factor for returns. | Conventional long-horizon credit assignment for strategy games. |
| `ppo.clip_coef` | `0.2` | `0.2` | PPO clipping epsilon. | Proven stable default preventing destructive policy jumps. |
| `ppo.ent_coef` | `0.01` | `0.01` | Entropy regularization strength. | Encourages exploration while still allowing policy convergence. |
| `ppo.vf_coef` | `0.5` | `0.5` | Value-loss weighting in total loss. | Balances policy and value learning in common PPO regimes. |
| `ppo.lr` | `3e-4` | `3e-4` | Optimizer learning rate. | Robust starting point across architectures/backends in this project. |
| `ppo.max_grad_norm` | `0.5` | `0.5` | Global gradient clipping threshold. | Protects against occasional gradient spikes and training divergence. |

### `training_format` (`TrainingFormatConfig`)

| Key | Default | Recommended default | What it controls | Why this default is recommended |
| --- | --- | --- | --- | --- |
| `training_format.format_schedule` | `[]` | `[]` unless running explicit curriculum | Time-based format curriculum definitions. | Empty keeps one stationary distribution for clean comparisons. |
| `training_format.format_mix` | `[]` | `[]` for single-format; explicit 50/50 for mixed 2p/4p studies | Weighted sampling across formats. | Makes format balancing explicit and reproducible when needed. |
| `training_format.rollout_groups` | `[]` | Define per-format groups for JAX mixed runs | Separate rollout collectors per format. | Required to avoid JAX recompilation/shape switching issues in mixed 2p/4p. |
| `training_format.phases` | `[]` | `[]` unless using staged curricula | Additional staged format metadata. | Avoids unnecessary schedule complexity in baseline experiments. |

### `opponent_mix` (`OpponentMixConfig`)

| Key | Default | Recommended default | What it controls | Why this default is recommended |
| --- | --- | --- | --- | --- |
| `opponent_mix.weights.latest` | `1.0` | `1.0` | Weight for newest self-play snapshot opponent. | Ensures training remains anchored to current policy strength. |
| `opponent_mix.weights.historical` | `0.0` | `0.0` baseline; raise for anti-forgetting studies | Weight for older snapshot opponents. | Off by default to keep baseline simple and isolate historical-mix effects. |
| `opponent_mix.weights.scripted_sniper` | `0.0` | `0.0` unless scripted-opponent robustness study | Weight for scripted sniper opponent. | Keeps baseline unbiased toward scripted exploit patterns. |
| `opponent_mix.weights.random` | `0.0` | `0.0` baseline; raise for exploration regularization | Weight for random opponent. | Useful knob for robustness, but default stays focused on primary opponent regime. |
| `opponent_mix.temperature` | `1.0` | `1.0` | Sampling temperature over opponent weights. | Neutral temperature reflects declared weights directly and is easiest to reason about. |
| `opponent_mix.curriculum` | `[]` | `[]` unless explicitly scheduling mixture changes | Time-varying opponent-mixture curriculum. | Static mixes are easier to reproduce and compare. |

### `wandb` (`WandBConfig`)

| Key | Default | Recommended default | What it controls | Why this default is recommended |
| --- | --- | --- | --- | --- |
| `wandb.enabled` | `False` | `False` locally; `True` for team tracking | Enables Weights & Biases logging. | Off by default keeps local runs dependency-light and private. |
| `wandb.project` | `null` | Set when `enabled: true` | W&B project namespace. | Explicit naming prevents logging to unintended projects. |
| `wandb.entity` | `null` | Set when collaborating | W&B user/org owner. | Avoids access/visibility confusion in shared environments. |
| `wandb.group` | `null` | Set per experiment family | Groups related runs in W&B UI. | Makes multi-run comparison dashboards cleaner. |
| `wandb.tags` | `[]` | Add architecture/format tags when enabled | Free-form run tags. | Tags speed filtering and slice analysis later. |
| `wandb.log_artifacts` | `False` | `False` unless artifact lineage is required | Upload checkpoints/artifacts to W&B. | Saves bandwidth/storage unless remote artifact tracking is needed. |
| `wandb.log_model_every` | `100` | `100` | Model artifact logging interval (updates). | Coarse cadence limits storage churn while preserving snapshots. |
| `wandb.watch_model` | `False` | `False` | Enables gradient/parameter watching. | Useful diagnostically but adds overhead/noise for routine runs. |

### `replay` (`ReplayConfig`)

| Key | Default | Recommended default | What it controls | Why this default is recommended |
| --- | --- | --- | --- | --- |
| `replay.enabled` | `False` | `False` unless validating determinism/regressions | Toggle replay export on checkpoints. | Off by default avoids extra I/O in normal training. |
| `replay.every_n_checkpoints` | `1` | `1` when enabled | Replay frequency relative to checkpoint saves. | Guarantees consistent replay coverage for saved milestones. |
| `replay.opponent` | `random` | `random` | Opponent used in replay games. | Stable, inexpensive baseline for deterministic regression checks. |
| `replay.seed_policy` | `update` | `update` | Strategy for replay seed selection. | Ties replay deterministically to update index for reproducible diffs. |
| `replay.max_steps` | `500` | `500` | Max replay episode length. | Matches default episode horizon for comparable trajectories. |
| `replay.output_dir` | `replays` | `replays` | Replay output subdirectory name. | Predictable location simplifies tooling and artifact pickup. |

### `checkpoint_retention` (`CheckpointRetentionConfig`)

| Key | Default | Recommended default | What it controls | Why this default is recommended |
| --- | --- | --- | --- | --- |
| `checkpoint_retention.keep_last_n` | `5` | `5` | Always keep N most recent checkpoints. | Protects recent recovery points without runaway disk growth. |
| `checkpoint_retention.keep_every_n_updates` | `0` | `0` baseline; set for sparse long-term archive | Periodic archival keep interval (`0` disables). | Disabled by default to avoid extra disk usage unless explicitly needed. |
| `checkpoint_retention.keep_best_k_by_metric` | `0` | `0` baseline; set >0 for leaderboard workflows | Number of top checkpoints retained by metric. | Keeps behavior simple until best-model selection is required. |
| `checkpoint_retention.best_metric_name` | `episode_reward_mean` | `episode_reward_mean` | Metric used for best-K retention. | Available in standard logs and aligned with objective optimization. |
| `checkpoint_retention.best_metric_mode` | `max` | `max` | Whether higher/lower metric is better. | Correct mode for reward-like metrics used by defaults. |
| `checkpoint_retention.min_update_for_pruning` | `0` | `0` | Update floor before any pruning begins. | Immediate pruning keeps disk bounded from run start. |
| `checkpoint_retention.dry_run_pruning` | `False` | `False` | Simulate retention decisions without deleting files. | Set `True` only when validating a new retention policy safely. |


### Allowed values for constrained keys

For keys that accept a constrained set of string values, use the following:

| Key | Allowed values | Notes |
| --- | --- | --- |
| `device` | `auto`, or any explicit `torch.device` string (for example `cpu`, `cuda`, `cuda:0`) | `auto` resolves to `cuda` when available, otherwise `cpu`. |
| `opponent` | `self`, `random`, `sniper` | End-to-end JAX PPO currently supports only `self` and `random`; use Kaggle/Torch for `sniper`. |
| `env_backend` | `kaggle`, `jax` (`python` is also accepted as alias for Kaggle path) | Selects environment implementation backend. |
| `rl_backend` | `torch`, `jax` | Selects PPO implementation backend. |
| `multi_opponent_mode` | `shared_current`, `sampled_pool`, `mixed` | Controls how current vs historical self-play opponents are sampled. |
| `env.terminal_reward_mode` | `binary_win`, `ranked`, `score_share`, `survival_plus_rank` | Terminal reward shaping strategy. |
| `model.architecture` | `mlp`, `attention`, `transformer` | `transformer` is an alias for the attention implementation. |
| `replay.seed_policy` | `update`, `constant` | `update` ties replay seeds to update index; `constant` uses a fixed seed. |
| `checkpoint_retention.best_metric_mode` | `max`, `min` | `max` for reward/win metrics, `min` for loss/error metrics. |

For structured list/dict fields (for example `training_format.format_mix`,
`training_format.rollout_groups`, `opponent_mix.curriculum`, and
`checkpoint_retention` schedules), refer to the checked-in config examples under
`configs/` and keep schemas consistent with existing entries.

### Practical default profiles

- **Baseline reproducible run**: keep all defaults, set only unique `run_name` and
  desired `ppo.total_updates`.
- **Self-play study**: set `self_play_enabled: true`, keep
  `self_play_latest_probability: 0.5`, and introduce historical/random
  opponent weights only in controlled ablations.
- **JAX mixed 2p/4p run**: set `env_backend: jax`, `rl_backend: jax`, and define
  separate `training_format.rollout_groups` for `player_count: 2` and `4`.
