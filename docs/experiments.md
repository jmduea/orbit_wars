# Running Orbit Wars experiments with Hydra

This guide is the canonical reference for launching, sweeping, and resuming Orbit Wars training with Hydra.

## 1) Hydra basics for this repo

Training entrypoint:

```bash
uv run python -m src.train experiment=attention_training
```

How config composition works:

- `conf/config.yaml` is the root config.
- `experiment=<name>` selects a preset from `conf/experiment/<name>.yaml`.
- Additional CLI overrides patch specific keys.

Examples:

```bash
uv run python -m src.train experiment=attention_training
uv run python -m src.train experiment=attention_training env.player_count=4
uv run python -m src.train experiment=attention_training ppo.total_updates=5000
```

### Override patterns

- Existing key override: `key=value`
- Nested override: `section.key=value`
- Optional append-only override for missing keys: `+new_key=value`

```bash
uv run python -m src.train experiment=attention_training +notes=hydra_test
```

If Hydra reports that a key is not in the struct/schema, either:

1. Use the correct existing key name, or
2. Intentionally add with `+...` when dynamic keys are supported for your workflow.

## 2) Multirun usage and output layout

Use `-m` to sweep one or more values:

```bash
uv run python -m src.train -m \
  experiment=attention_training \
  env.player_count=2,4 \
  ppo.total_updates=1000,2000
```

This runs one job per Cartesian-product combination.

Hydra outputs:

- Single run: `outputs/<YYYY-MM-DD>/<HH-MM-SS>/`
- Multirun: `multirun/<YYYY-MM-DD>/<HH-MM-SS>/<job_id>/`
- Each job includes `.hydra/` metadata (`config.yaml`, `hydra.yaml`, `overrides.yaml`).

Training artifacts/checkpoints still follow the project artifact paths (for example `/artifacts/...`) based on run configuration.

## 3) Resume behavior with checkpoints

Use `resume_checkpoint=<path>` in Hydra form:

```bash
uv run python -m src.train \
  experiment=attention_training \
  resume_checkpoint=/artifacts/attention_training/orbit_wars_ppo_attention_training/jax_ckpt_000050.pkl
```

Behavior details:

- `ppo.total_updates` is the absolute final update target.
- Resuming from update `N` continues at `N+1`.
- Keep architecture- and shape-compatible configs when resuming.

## 4) Backend notes (JAX-only)

- Training uses the JAX environment, JAX policy, and JAX PPO implementation.
- Checkpoints: `jax_ckpt_last.pkl`, `jax_ckpt_*.pkl`
- Use JAX-compatible experiment presets (for example `jax_training`, `jax_mixed_2p_4p_training`, `jax_entity_transformer_*`, `attention_training`, `full_training`).
- Do not change shape-defining settings between save and resume/eval unless intentionally starting a fresh run.

## 5) Migration: old `--config` commands → Hydra commands

For the complete migration matrix, timeline, and troubleshooting, see [`docs/hydra_migration.md`](hydra_migration.md).

Quick examples:

```bash
uv run python -m src.train experiment=attention_training
```

```bash
uv run python -m src.train experiment=jax_training resume_checkpoint=/path/to/jax_ckpt_000050.pkl
```

## Canonical experiment authoring policy

- Canonical experiment editing and sweeps happen only in `conf/` (`conf/config.yaml`, `conf/experiment/*.yaml`, and config groups).
- `configs/` has been removed; use Hydra experiment selection from `conf/experiment/` for all authoring and execution.

## 6) Canonical opponent profiles, curricula, and sweep-safe knobs

Use `curriculum=<profile>` to choose progressive difficulty and `opponent_mix=<profile>` for static compatibility settings. Progressive opponent schedules live under `curriculum.stages`; `opponent_mix.curriculum` and `training_format.phases` are legacy schedule surfaces and are rejected when the new curriculum is enabled.

Canonical profiles:

- `curriculum/latest_only`: No staged curriculum; sample latest policy through static opponent settings.
- `curriculum/self_play_staged`: Metric-gated staged self-play with random/bootstrap opponents, frozen historical snapshots, and scripted exploiters.
- `opponent_mix/latest_only`: Static latest-only opponent compatibility profile.
- `opponent_mix/self_play_curriculum`: Enables self-play compatibility flags while leaving staged scheduling to `curriculum.self_play_staged`.

### Sweep-safe knobs matrix

| Field | Sweep-safe? | Notes |
| --- | --- | --- |
| `curriculum` (config group choice) | **Yes** | Primary high-level sweep axis for staged difficulty. |
| `curriculum.stages[*].opponent_families.*` | **Yes** | Safe for intentional staged opponent-mixture ablations. |
| `curriculum.stages[*].promote_if.*` | **Yes** | Safe when testing promotion thresholds/windows. |
| `curriculum.snapshot.*` | **Usually** | Keep `pool_size` and `interval_updates` coherent with historical weights. |
| `opponent_mix` (config group choice) | **Yes** | Primary high-level sweep axis for opponent behavior. |
| `opponent_mix.weights.*` | **Limited** | Static compatibility values; prefer stage weights for progressive training. |
| `opponent_mix.temperature` | **Yes** | Safe to sweep sampling sharpness. |
| `ppo.*`, `env.reward_*` | **Yes** | Typical training/reward sweeps. |
| `self_play_enabled` | **No** | Fixed by opponent profile; avoid overriding directly in sweeps. |
| `self_play_pool_size` | **No** | Fixed by profile and validated against `self_play_enabled`. |
| `self_play_snapshot_interval` | **No** | Fixed by profile and validated against `self_play_enabled`. |
| `opponent_mix.curriculum` | **No** | Deprecated schedule surface; use `curriculum.stages`. |

### Copy/paste multirun examples (canonical fields only)

Sweep profile + mixture weights:

```bash
uv run python -m src.train -m \
  experiment=jax_training \
  curriculum=latest_only,self_play_staged \
  opponent_mix=latest_only,self_play_curriculum
```

Sweep profile + temperature:

```bash
uv run python -m src.train -m \
  experiment=jax_training \
  curriculum=self_play_staged \
  curriculum.stages.1.opponent_families.historical=0.15,0.25
```

Reward + opponent-mixture sweep:

```bash
uv run python -m src.train -m \
  experiment=jax_training \
  curriculum=latest_only,self_play_staged \
  opponent_mix=latest_only,self_play_curriculum \
  env.reward_capture_planet=0.05,0.1 \
  curriculum.stages.1.opponent_families.nearest_sniper=0.0,0.1
```

## 7) Experiment tuning playbook (what to change for each goal)

Use this section as the authoritative “which knob for which goal” map. Each goal lists:

- **Primary keys to edit** (the actual knobs for that goal).
- **Nearby keys to avoid** for that specific goal (easy-to-confuse controls that change something else).

### I want to change model capacity

#### Model capacity primary keys

- `model=<group_name>` (preferred): pick a model profile such as `entity_transformer_500k`, `entity_transformer_700k`, `entity_transformer_1m`, or `attention`.
- `model.hidden_size`: primary width/capacity knob.
- `model.attention_heads`: attention partitioning/capacity coupling knob (keep compatible with hidden size).

#### Model capacity nearby keys to avoid

- `ppo.minibatch_size`, `ppo.rollout_steps`, `ppo.num_envs`: these change optimization/runtime budget, not architecture capacity.
- `env.*` keys like `env.player_count` or `env.max_planets`: these change task difficulty/distribution, not model size.
- `model.normalize_observations`, `model.obs_norm_clip`: normalization behavior, not capacity.

### I want to change training budget

#### Training budget primary keys

- `ppo.total_updates`: top-level budget horizon.
- `ppo.num_envs`: controls parallel sample collection rate.
- `ppo.rollout_steps`: controls samples per update.
- `ppo.epochs`: optimization work per update.
- `ppo.minibatch_size`: optimization granularity and throughput.

#### Training budget nearby keys to avoid

- `model.hidden_size`, `model.attention_heads`: capacity knobs (architecture changes).
- `env.reward_*`: objective shaping, not budget.
- `self_play_enabled`, `opponent_mix.curriculum`: opponent-distribution controls, not core budget.

### I want to change opponent curriculum

#### Opponent curriculum primary keys

- `curriculum=<profile>`: select staged difficulty (`latest_only` or `self_play_staged`).
- `curriculum.stages[*].opponent_families.latest`
- `curriculum.stages[*].opponent_families.historical`
- `curriculum.stages[*].opponent_families.nearest_sniper`
- `curriculum.stages[*].opponent_families.turtle`
- `curriculum.stages[*].opponent_families.opportunistic`
- `curriculum.stages[*].opponent_families.random`
- `curriculum.stages[*].opponent_families.noop`
- `curriculum.stages[*].promote_if.*`
- `curriculum.snapshot.pool_size` and `curriculum.snapshot.interval_updates`

#### Opponent curriculum nearby keys to avoid

- `opponent`, `multi_opponent_mode`: legacy/low-level behavior toggles that can conflict with canonical profile intent.
- `self_play_enabled`, `self_play_pool_size`, `self_play_snapshot_interval`, `self_play_latest_probability`: profile-owned fields; do not sweep ad hoc when using canonical opponent profiles.
- `opponent_mix.curriculum`, `training_format.phases`: deprecated schedule surfaces; use `curriculum.stages`.
- `env.reward_*`: reward shaping changes learning target, not opponent sampling policy.

### I want to change environment difficulty

#### Environment difficulty primary keys

- `env.player_count`: major difficulty and multi-agent interaction shift.
- `env.max_planets`, `env.max_fleets`, `env.candidate_count`: state/action complexity knobs.
- `env.MAX_STEPS`: horizon/difficulty knob.
- `env.max_ships`, `env.max_production`, `env.ship_speed`: game dynamics difficulty knobs.
- `training_format=<group_name>` and/or `training_format.schedule`: use curated player-count schedules for curriculum-like environment shifts.

#### Environment difficulty nearby keys to avoid

- `ppo.*`: training compute/budget knobs; can mask real difficulty effects if changed at the same time.
- `model.*`: capacity knobs; can compensate for difficulty changes and confound attribution.
- `opponent_mix.*`: opponent distribution knobs; keep fixed when isolating environment-difficulty effects.

## 8) Sweep templates and output-directory hygiene

### Cartesian sweeps (`-m`) template

```bash
uv run python -m src.train -m \
  experiment=<experiment_name> \
  <axis_a>=<a1>,<a2> \
  <axis_b>=<b1>,<b2>,<b3>
```

Example (capacity × budget):

```bash
uv run python -m src.train -m \
  experiment=attention_training \
  model.hidden_size=128,192 \
  ppo.total_updates=2000,5000
```

### Staged sweeps (`-m`) template

Stage 1 (coarse scan):

```bash
uv run python -m src.train -m \
  experiment=jax_training \
  ppo.total_updates=1000,2000,4000 \
  ppo.lr=0.0001,0.0003
```

Stage 2 (refine around best Stage-1 region):

```bash
uv run python -m src.train -m \
  experiment=jax_training \
  ppo.total_updates=3000,4000,5000 \
  ppo.lr=0.0002,0.0003,0.0004
```

### Output-directory hygiene

Keep runs discoverable by assigning explicit Hydra directories per sweep campaign:

```bash
uv run python -m src.train -m \
  experiment=attention_training \
  model.hidden_size=128,192 \
  hydra.sweep.dir=multirun/capacity_sweeps \
  hydra.sweep.subdir='${hydra.job.num}_${model.hidden_size}'
```

Single-run convention:

```bash
uv run python -m src.train \
  experiment=attention_training \
  hydra.run.dir=outputs/capacity_debug/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

Recommended hygiene conventions:

- Use campaign prefixes in `hydra.sweep.dir` (for example `multirun/budget_sweeps`, `multirun/opponent_curriculum`).
- Encode the key varied axis in `hydra.sweep.subdir`.
- Keep checkpoint/artifact destinations stable while changing Hydra metadata directories only.

## 9) Common anti-patterns

- **Duplicate knobs for the same intent**: changing both `model=<group>` and manual `model.hidden_size`/`model.attention_heads` overrides without documenting intent.
- **Conflicting overrides**: mixing canonical `opponent_mix=<profile>` with ad hoc overrides to profile-owned `self_play_*` fields in the same sweep.
- **Too many moving parts per sweep**: changing `model.*`, `ppo.*`, and `env.*` simultaneously in early sweeps, making attribution impossible.
- **Legacy `configs/` edits**: editing or relying on removed/non-canonical config paths instead of `conf/` and Hydra overrides.
- **Output sprawl**: running large `-m` jobs without explicit `hydra.sweep.dir` conventions, making comparison and cleanup harder.
