# Training Configs

PPO and rollout hyperparameters, including mixed 2p/4p rollout allocation.

## Rollout allocation

`training.num_envs` is the env-parallelism budget. Per-group env counts are derived from `training.format_weights` (and curriculum stage weights for scheduling), not duplicated in YAML.

| `rotate_format_rollouts` | Meaning of the number in preset names | Example |
| --- | --- | --- |
| `false` (`*_split`) | **Total** parallel envs per update (split 50/50 → half per format) | `2p4p_32_split` → 16+16 = **32** active |
| `true` (`*_rotate`) | **Active** envs on the single format collecting this update | `2p4p_16_rotate` → **16** envs per update |

Static mix when curriculum stages omit `format_weights` uses `training.format_weights`. Curriculum stages may override mix at runtime via `format_weights`.

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

Typical fields:

```yaml
rollout_steps: 500
num_envs: 32
format_weights:
  2: 0.5
  4: 0.5
total_updates: 100
epochs: 2
minibatch_size: 512
gamma: 0.99
gae_lambda: 0.95
lr: 0.0003
ent_coef: 0.005
```

Examples:

```bash
uv run ow train training=smoke_2p_16 curriculum=off opponents=noop_only
uv run ow train training=workstation
uv run ow train training=2p4p_32_split training.total_updates=2
```
