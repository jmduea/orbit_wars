# Training Configs

PPO and rollout hyperparameter knobs

Typical fields:

```yaml
rollout_steps: 500
num_envs: 32
total_updates: 100
epochs: 2
minibatch_size: 512
gamma: 0.99
gae_lambda: 0.95
lr: 0.0003
ent_coef: 0.005
```

Examples

```bash
uv run ow train training=smoke
uv run ow train training=workstation
```

Use `training=smoke` for quick config and pipeline checks.
