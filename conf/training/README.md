# Training Configs

PPO and rollout hyperparameters, including mixed 2p/4p rollout allocation.

## Rollout allocation

`training.num_envs` is the single env-parallelism budget. Per-group env counts are derived from `training.format_weights` (and curriculum stage weights for scheduling), not duplicated in YAML.

| `rotate_format_rollouts` | Meaning of `num_envs` | Example |
| --- | --- | --- |
| `false` (split mode) | Total envs split across active formats by weight | `mixed_2p4p_16_total` → 16+16 from 32 |
| `true` (per-group mode) | Env count for whichever format collects this update | `mixed_2p4p_16_rotating` → 16 per group when active |

Static mix when curriculum stages omit `format_weights` uses `training.format_weights`. Curriculum stages may override mix at runtime via `format_weights`.

## Presets

| Preset | Use |
| --- | --- |
| `smoke` | Fast pipeline check (2 envs, 2p-only) |
| `smoke_2p_16` | Smoke launch recipe with 16 parallel 2p envs |
| `workstation` | Rotating mixed 2p/4p, 16 envs per active format |
| `2p_16`, `2p_32`, `4p_16`, `4p_32` | Single-format runs |
| `mixed_2p4p_8_total`, `mixed_2p4p_16_total`, `mixed_2p4p_32_total` | Parallel mixed collection (split mode) |
| `mixed_2p4p_16_rotating` | Workstation-style rotating mixed collection |

Typical fields:

```yaml
rollout_steps: 500
num_envs: 32
format_weights:
  2: 1.0
  4: 0.0
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
uv run ow train training=mixed_2p4p_16_total training.total_updates=2
```
