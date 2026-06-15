# Training Configs

PPO and rollout hyperparameters, including mixed 2p/4p rollout allocation.

## Rollout allocation

`training.num_envs` is the env-parallelism budget. Per-group env counts are derived from `training.format_weights` (and curriculum stage weights for scheduling), not duplicated in YAML.

| `rotate_format_rollouts` | Meaning of the number in preset names | Example |
| --- | --- | --- |
| `false` (`*_split`) | **Total** parallel envs per update (split 50/50 → half per format) | `2p4p_32_split` → 16+16 = **32** active |
| `true` (`*_rotate`) | **Active** envs on the single format collecting this update | `2p4p_16_rotate` → **16** envs per update |

Static mix when curriculum stages omit `format_weights` uses `training.format_weights`. Curriculum stages may override mix at runtime via `format_weights`.

## `rollout_microbatch_envs`

PPO rollout collection shards each group's env axis into microbatches. At compose time:

1. `training.rollout_microbatch_envs` must be **≤ each** rollout group's `num_envs`.
2. Each group's `num_envs` must be **evenly divisible** by the microbatch size.

For **split** presets with 50/50 weights, each format gets `total_envs / 2` parallel envs — set microbatch to that half (e.g. `2p4p_32_split` → 16+16 envs → microbatch **16**, not 32). For **rotate** presets, each active group uses the full `num_envs` budget. Single-format presets (`2p_32`, `4p_16`, …) set microbatch to match that group's env count when maximizing throughput.

`training=default` selects `2p4p_32_split` (canonical mixed production geometry).

## Presets

| Preset | Parallel envs per update | Notes |
| --- | --- | --- |
| `smoke` | 2 (2p-only) | Fast pipeline check |
| `smoke_2p_16` | 16 (2p-only) | Smoke launch recipe |
| `workstation` | 32 (16+16 split) | Even 2p/4p mix + workstation PPO knobs |
| `2p_16`, `2p_32`, `4p_16`, `4p_32` | name = total | Single-format runs |
| `2p4p_16_split` | 16 (8+8) | Split mode |
| `2p4p_32_split` | 32 (16+16) | Split mode; old `format=2p_4p_16env` equivalent |
| `2p4p_64_split` | 64 (32+32) | Split mode; old `format=2p_4p_32env` equivalent |
| `2p4p_16_rotate` | 16 active | One format per update |

`update_chunk_rows` sets rows per PPO `lax.scan` step (capped by rollout batch size); minibatch count is `ceil(total_rows / update_chunk_rows)`.

**Seed scheduler:** `reseed_every_updates: -1` (default) auto-scales to `max(25, total_updates // 10)`. Use `0` to disable periodic reseed. Calibrate with `uv run ow benchmark calibrate-seed-scheduler`.

Typical fields:

```yaml
rollout_steps: 500
num_envs: 32
format_weights:
  2: 0.5
  4: 0.5
total_updates: 100
epochs: 2
update_chunk_rows: 1024
gamma: 0.99
gae_lambda: 0.95
lr: 0.0003
ent_coef: 0.005
```

Examples:

```bash
uv run ow train training=smoke_2p_16 curriculum=noop_only
uv run ow train training=workstation
uv run ow train training=2p4p_32_split training.total_updates=2
```
