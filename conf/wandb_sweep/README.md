# **W&B Sweeps**

## What is a sweep?

A normal Hydra run executes one config:

```bash
uv run ow train task=shield_cheap
```

A W&B sweep executes ***many*** configs:

```bash
Run 1: task=shield_cheap training.lr=0.0001
Run 2: task=shield_cheap training.lr=0.0003
Run 3: task=shield_cheap training.lr=0.0006
...
```

W&B chooses the values. The local wandb agent executes them using your training loop.

## Sweep folders

```bash
conf/wandb_sweep/
  base.yaml

  # How W&B chooses values: grid, random, bayes
  method/ 
    grid.yaml
    random.yaml
    bayes.yaml

  # What W&B optimizes for
  metric/
    overall_win_rate.yaml
    episode_reward_mean.yaml
    samples_per_sec.yaml

  # Parameters that stay fixed during the sweep
  fixed/
    *.yaml

  # Parameters W&B varies
  space/
    *.yaml

  # Final sweep recipe
  <sweep_recipe>.yaml
```

Metric and method are pretty well covered by W&B documentation, so if you need to know more about how those work, check out the official [Weights & Biases documentation](https://docs.wandb.ai/models/sweeps)

## Fixed Block example (fixed/*.yaml)

```yaml
# conf/wandb_sweep/fixed/shield_cheap_train.yaml
# @package _global_

parameters:
  task:
    value: shield_cheap

  output.campaign:
    value: shield_cheap_sweep 

  telemetry.wandb.group:
    value: shield_cheap_sweep

  training.total_updates: # make every run in the sweep run 100 updates
    value: 100
```

By creating a fixed block you create a reusable component you can re-use in multiple sweeps without having to define it in seperate sweep .yamls

## Search-space block example

```yaml
# conf/wandb_sweep/space/pgt_model_capacity.yaml
# @package _global_

parameters:
  model.hidden_size:
    values: [128, 192, 224]
  model.planet_transformer_layers:
    values: [1, 2, 3]
  model.max_moves_k:
    values: [3, 5, 8]
```

This means W&B will vary these values, in this example, using this block in a sweep means W&B will spin up training runs varied across these parameters meaning we get to test different size architectures against each other easily.

## Full sweep recipe example

```yaml
# conf/wandb_sweep/2p_only_throughput.yaml
# @package _global_

defaults:
  - base
  - command: ow_train
  - method: grid
  - metric: samples_per_sec
  - fixed: throughput_2p
  - override wandb_sweep/fixed@fixed: no_artifacts
  - space: throughput_2p
  - _self_

name: 2p_only_throughput
run_cap: 20
```

This composes:

```bash
base sweep settings
+ grid search
+ optimize for samples_per_sec
+ fixed throughput_2p params           # fixed/throughput_2p.yaml
+ fixed no_artifacts params            # fixed/no_artifacts.yaml
+ variable throughput_2p params        # space/throughput_2p.yaml
```

## Generated W&B YAML

```yaml
method: grid
metric:
  name: samples_per_sec
  goal: maximize
command:
- ${env}
- uv
- run
- ow
- train
- ${args_no_hyphens}
parameters:
  artifacts:
    value: disabled
  format:
    values:
    - 2p_16env
    - 2p_32env
  training.rollout_steps:
    values:
    - 250
    - 500
  training.minibatch_size:
    values:
    - 256
    - 512
    - 1024
  training.rollout_microbatch_envs:
    values:
    - 4
    - 8
    - 16
    - 32
  training.update_chunk_rows_min:
    values:
    - 2048
    - 4096
    - 8192
run_cap: 20
```

## Sweep generation workflow

### 1. Generate sweep YAML

```bash
uv run ow make wandb_sweep={your_sweep_name}
```

Expected output:

```bash
Wrote artifacts/sweeps/{your_sweep_name}.yaml
```

### 2. Register sweep with W&B

```bash
uv run wandb sweep artifacts/sweeps/{your_sweep_name}.yaml
```

W&B prints an agent command:

```bash
wandb agent <entity>/<project>/<sweep_id>
```

### 3. Run the agent and profit $$$

```bash
uv run wandb agent <entity>/<project>/<sweep_id>
```
