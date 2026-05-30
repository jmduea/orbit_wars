# Practical guidelines

## Use a config group when the value should define one normal run

Examples:

```bash
task/shield_cheap.yaml
training/workstation.yaml
format/2p_4p_16env.yaml
```

## Launch recipes (multi-group overrides)

When several groups should change together, pass explicit Hydra overrides:

| Recipe | Overrides |
| --- | --- |
| smoke | `training=smoke format=2p_16env curriculum=off opponents=noop_only telemetry=throughput_only artifacts=disabled` |
| shield_cheap | `task=shield_cheap telemetry=default` |
| shield_tiered | `task=shield_tiered telemetry=default` |

## Use `wandb_sweep/fixed` for values that stay fixed across a sweep

Example:

```yaml
parameters:
  task:
    value: shield_cheap

  output.campaign:
    value: shield_cheap_sweep
```

## Use `wandb_sweep/space` for values W&B should vary

Example:

```yaml
parameters:
  training.lr:
    values: [0.0001, 0.0003, 0.0006]
```

## Debugging

### Print resolved config

```bash
uv run ow train task=shield_cheap print_resolved_config=true
```

### Generate a sweep and inspect it

```bash
uv run ow make wandb_sweep=shield_cheap_history
cat artifacts/sweeps/shield_cheap_history.yaml
```

### Run one manual version of a sweep trial

```bash
uv run ow train \
  task=shield_cheap \
  training.lr=0.0003 \
  task.feature_history_steps=10 \
  task.trajectory_shield_horizon=50
```

## Recommended Development Flow

### 1. Create or update normal configs in `task/`, `training/`, `model/`, etc

### 2. Verify with direct overrides

```bash
uv run ow train task=shield_cheap print_resolved_config=true
```

### 3. Run a smoke test

```bash
uv run ow train training=smoke format=2p_16env curriculum=off opponents=noop_only telemetry=throughput_only artifacts=disabled
```

### 4. If running many trials, define

```bash
wandb_sweep/fixed/*.yaml
wandb_sweep/space/*.yaml
wandb_sweep/<recipe>.yaml
```

### 5. Generate the W&B sweep YAML

### 6. Register with W&B

### 7. Run the W&B agent
