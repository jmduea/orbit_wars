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
  resume_checkpoint=/artifacts/attention_training/orbit_wars_ppo_attention_training/ckpt_000050.pt
```

```bash
uv run python -m src.train \
  experiment=jax_training \
  resume_checkpoint=/artifacts/jax_training/orbit_wars_ppo_jax_training/jax_ckpt_000050.pkl
```

Behavior details:

- `ppo.total_updates` is the absolute final update target.
- Resuming from update `N` continues at `N+1`.
- Keep architecture- and backend-compatible configs when resuming.

## 4) Backend notes (JAX-only)

- Setup: `env_backend=jax`, `rl_backend=jax`
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

## 6) Canonical opponent profiles and sweep-safe knobs

Use `opponent_mix=<profile>` to choose a canonical opponent profile. These profiles own the brittle self-play/opponent-mixture combinations.

Canonical profiles:
- `opponent_mix/latest_only`: No self-play pool; sample only latest policy.
- `opponent_mix/self_play_curriculum`: Self-play enabled with historical snapshots and curriculum progression.

### Sweep-safe knobs matrix

| Field | Sweep-safe? | Notes |
|---|---|---|
| `opponent_mix` (config group choice) | **Yes** | Primary high-level sweep axis for opponent behavior. |
| `opponent_mix.weights.*` | **Yes** | Safe to sweep for mixture ablations. |
| `opponent_mix.temperature` | **Yes** | Safe to sweep sampling sharpness. |
| `ppo.*`, `env.reward_*` | **Yes** | Typical training/reward sweeps. |
| `self_play_enabled` | **No** | Fixed by opponent profile; avoid overriding directly in sweeps. |
| `self_play_pool_size` | **No** | Fixed by profile and validated against `self_play_enabled`. |
| `self_play_snapshot_interval` | **No** | Fixed by profile and validated against `self_play_enabled`. |
| `opponent_mix.curriculum` | **No** | Profile-owned schedule; edit profile file intentionally instead of CLI sweeps. |

### Copy/paste multirun examples (canonical fields only)

Sweep profile + mixture weights:

```bash
uv run python -m src.train -m \
  experiment=jax_training \
  opponent_mix=latest_only,self_play_curriculum \
  opponent_mix.weights.latest=0.5,0.7 \
  opponent_mix.weights.random=0.0,0.2
```

Sweep profile + temperature:

```bash
uv run python -m src.train -m \
  experiment=jax_training \
  opponent_mix=self_play_curriculum \
  opponent_mix.temperature=0.8,1.0,1.2
```

Reward + opponent-mixture sweep:

```bash
uv run python -m src.train -m \
  experiment=jax_training \
  opponent_mix=latest_only,self_play_curriculum \
  env.reward_capture_planet=0.05,0.1 \
  opponent_mix.weights.scripted_sniper=0.0,0.1
```

## 7) Experiment tuning playbook (what to change for each goal)

Use this section as the authoritative “which knob for which goal” map. Each goal lists:
- **Primary keys to edit** (the actual knobs for that goal).
- **Nearby keys to avoid** for that specific goal (easy-to-confuse controls that change something else).

### I want to change model capacity

**Primary keys to edit**
- `model=<group_name>` (preferred): pick a model profile such as `entity_transformer_500k`, `entity_transformer_700k`, `entity_transformer_1m`, or `attention`.
- `model.hidden_size`: primary width/capacity knob.
- `model.attention_heads`: attention partitioning/capacity coupling knob (keep compatible with hidden size).

**Nearby keys to avoid for this goal**
- `ppo.minibatch_size`, `ppo.rollout_steps`, `ppo.num_envs`: these change optimization/runtime budget, not architecture capacity.
- `env.*` keys like `env.player_count` or `env.max_planets`: these change task difficulty/distribution, not model size.
- `model.normalize_observations`, `model.obs_norm_clip`: normalization behavior, not capacity.

### I want to change training budget

**Primary keys to edit**
- `ppo.total_updates`: top-level budget horizon.
- `ppo.num_envs`: controls parallel sample collection rate.
- `ppo.rollout_steps`: controls samples per update.
- `ppo.epochs`: optimization work per update.
- `ppo.minibatch_size`: optimization granularity and throughput.

**Nearby keys to avoid for this goal**
- `model.hidden_size`, `model.attention_heads`: capacity knobs (architecture changes).
- `env.reward_*`: objective shaping, not budget.
- `self_play_enabled`, `opponent_mix.curriculum`: opponent-distribution controls, not core budget.

### I want to change opponent curriculum

**Primary keys to edit**
- `opponent_mix=<profile>`: select canonical policy (`latest_only` or `self_play_curriculum`).
- `opponent_mix.weights.latest`
- `opponent_mix.weights.historical`
- `opponent_mix.weights.scripted_sniper`
- `opponent_mix.weights.random`
- `opponent_mix.temperature`
- `opponent_mix.curriculum` (intentional schedule edits only; usually by editing the profile file).

**Nearby keys to avoid for this goal**
- `opponent`, `multi_opponent_mode`: legacy/low-level behavior toggles that can conflict with canonical profile intent.
- `self_play_enabled`, `self_play_pool_size`, `self_play_snapshot_interval`, `self_play_latest_probability`: profile-owned fields; do not sweep ad hoc when using canonical opponent profiles.
- `env.reward_*`: reward shaping changes learning target, not opponent sampling policy.

### I want to change environment difficulty

**Primary keys to edit**
- `env.player_count`: major difficulty and multi-agent interaction shift.
- `env.max_planets`, `env.max_fleets`, `env.candidate_count`: state/action complexity knobs.
- `env.episode_steps`: horizon/difficulty knob.
- `env.max_ships`, `env.max_production`, `env.ship_speed`: game dynamics difficulty knobs.
- `training_format=<group_name>` and/or `training_format.schedule`: use curated player-count schedules for curriculum-like environment shifts.

**Nearby keys to avoid for this goal**
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
