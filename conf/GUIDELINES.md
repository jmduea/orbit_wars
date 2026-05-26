# Practical guidelines

## Use a config group when the value should define one normal run

Examples:

```bash
task/shield_cheap.yaml
training/workstation.yaml
format/mix_2p_4p_16env.yaml
```

## Use a preset when multiple groups should change together

Particularly useful for making sure that logging/telemetry stays on when it should and gets disabled when it's not needed.

Examples:

```bash
preset/shield_off.yaml
preset/smoke.yaml
```

## Use `wandb_sweep/fixed` for values that stay fixed across a sweep

Example:

```yaml
parameters:
  preset:
    value: shield_cheap
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
uv run ow train preset=shield_cheap print_resolved_config=true
```

### Check a preset

```bash
uv run ow train preset=smoke print_resolved_config=true
```

### Generate a sweep and inspect it

```bash
uv run ow make wandb_sweep=shield_cheap_history
cat artifacts/sweeps/shield_cheap_history.yaml
```

### Run one manual version of a sweep trial

```bash
uv run ow train \
  preset=shield_cheap \
  training.lr=0.0003 \
  task.feature_history_steps=10 \
  task.trajectory_shield_horizon=50
```

## Recommended Development Flow

### 1. Create or update normal configs in `task/`, `training/`, `model/`, etc

### 2. Compose them into a `preset/*.yaml`

### 3. Verify with

```bash
uv run ow train preset=<preset_name> print_resolved_config=true
```

### 4. Run a smoke test

```bash
uv run ow train preset=smoke
```

### 5. If running many trials, define

```bash
wandb_sweep/fixed/*.yaml
wandb_sweep/space/*.yaml
wandb_sweep/<recipe>.yaml
```

### 6. Generate the W&B sweep YAML

### 7. Register with W&B

### 8. Run the W&B agent
