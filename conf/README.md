# **Orbit Wars Configuration Guide**

Config tests verify Hydra composition and set membership for command-critical values—not brittle equality against full resolved configs.

This folder contains the Hydra configuration system for Orbit Wars training, evaluation, W&B sweeps, artifacts, telemetry, and experiment overrides.

The intended workflow is:

```bash
uv run ow train training=smoke format=2p_16env curriculum=off opponents=noop_only telemetry=throughput_only artifacts=disabled
uv run ow train task=shield_cheap
uv run ow train print_resolved_config=true
```

For W&B sweeps:

```bash
uv run ow make wandb_sweep=2p_only_throughput
uv run wandb sweep outputs/_meta/sweeps/2p_only_throughput.yaml
uv run wandb agent <entity>/<project>/<sweep_id>
```

---

## **Launch recipes**

Common multi-group launches use direct Hydra overrides instead of preset bundles:

| Recipe | Overrides |
| --- | --- |
| smoke | `training=smoke format=2p_16env curriculum=off opponents=noop_only telemetry=throughput_only artifacts=disabled` |
| shield_cheap | `task=shield_cheap telemetry=default` |
| shield_tiered | `task=shield_tiered telemetry=default` |

Examples:

```bash
uv run ow train training=smoke format=2p_16env curriculum=off opponents=noop_only telemetry=throughput_only artifacts=disabled
uv run ow train task=shield_cheap
uv run ow train task=shield_tiered telemetry=shield_debug
```

---

## **Run flow**

A normal run:

```bash
# Pick your poison through yaml configs/hydra overrides:
conf/
    task/shield_cheap.yaml
    training/default.yaml
    model/transformer_factorized.yaml
    artifacts/default.yaml

uv run ow train \
    task=shield_cheap \
    training=default \
    model=transformer_factorized \
    artifacts=default
```

A W&B sweep:

```bash
# Pick your poison to turn into a sweep
conf/wandb_sweep/*.yaml

# Compose the peices parts together
uv run ow make wandb_sweep=2p_only_throughput

# sweep file generated at outputs/_meta/sweeps/

# let wandb handle the rest
wandb sweep outputs/_meta/sweeps/2p_only_throughput.yaml
wandb agent ...
```

## **Root Config (The default run config)**

```bash
conf/config.yaml
```

Composes the default config groups together:

```yaml
defaults:
  - model: default
  - task: default
  - reward: default
  - training: default
  - format: default
  - curriculum: default
  - opponents: default
  - telemetry: default
  - artifacts: default
  - _self_
```

Check out the resolved config:

```bash
uv run ow train print_resolved_config=true
```

## **Config Organization Standards**

Every config group should follow this structure:

```bash
conf/<group>/base.yaml          # shared concrete defaults
conf/<group>/default.yaml       # what the root config selects by default
conf/<group>/<variant>.yaml     # meaningful override profiles
```

Sweep-only metadata belongs in:

```bash
conf/wandb_sweep/space/*.yaml
conf/wandb_sweep/fixed/*.yaml
```

Runtime configs and sweep configs should be kept seperate:

Runtime configs should contain concrete values:

```yaml
rollout_steps: 500
```

Sweep configs may contain search spaces:

```yaml
parameters:
  training.lr:
    distribution: log_uniform_values
      min: 0.0001
      max: 0.0006
  training.rollout_steps:
    values: [250, 500]
```

## **Group Responsibilities**

Follow links to readme's for more info

- ### [model/](model/README.md)

    Defines policy/value model architecture.

- ### [task/](task/README.md)

    Defines environment/action/feature settings.

- ### [reward/](reward/README.md)

    Defines reward shaping and terminal reward behavior.

- ### [training/](training/README.md)

    Defines PPO and rollout hyperparameters

- ### TODO: [format/](format/README.md)

    Defines 2-player / 4-player rollout composition and environment counts.

- ### TODO: [curriculum/](curriculum/README.md)

    Defines curriculum stages.

- ### TODO: [opponents/](opponents/README.md)

    Defines opponent pool and self-play settings.

- ### TODO: [telemetry/](telemetry/README.md)

    Defines metric groups and W&B logging.

- ### TODO: [artifacts/](artifacts/README.md)

    Defines checkpoints, replay, Agent validation via Kaggle Docker image, and artifact pipeline behavior.

- ### [wandb_sweep/](wandb_sweep/README.md)

    Handles .yaml sweep composition, read the README.md for an in-depth guide.

## **Common Commands**

### **Smoke test**

```bash
uv run ow train training=smoke format=2p_16env curriculum=off opponents=noop_only telemetry=throughput_only artifacts=disabled print_resolved_config=true
```

### **Cheap shield training**

```bash
uv run ow train task=shield_cheap
```

### **Tiered shield debug**

```bash
uv run ow train task=shield_tiered telemetry=shield_debug
```

### **Override individual fields**

```bash
uv run ow train task=shield_cheap training.total_updates=10
```

```bash
uv run ow train task=shield_cheap task.feature_history_steps=15
```

```bash
uv run ow train task=shield_cheap telemetry.wandb.enabled=false
```
