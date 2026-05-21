# Config Migration Notes

Legacy flat YAML files and broad preset-style launches have been replaced by responsibility-based Hydra groups.

Use the group that owns the setting you want to change:

| Legacy intent | New responsibility path |
| --- | --- |
| Model architecture/capacity | `model=...`, `model.hidden_size=...` |
| Candidate count or task shape | `task.candidate_count=...`, `task.ship_bucket_count=...` |
| Reward shaping | `reward.reward_capture_planet=...`, `reward.reward_production_delta=...` |
| PPO budget and optimizer | `training.total_updates=...`, `training.lr=...` |
| 2p/4p rollout topology | `format=...` |
| Opponent source and snapshot pool | `opponents=...`, `opponents.snapshot.*` |
| Stage progression | `curriculum=...`, `curriculum.stages.*` |
| W&B metadata | `telemetry.wandb.group=...`, `telemetry.wandb.tags=...` |

Example:

```bash
uv run python -m src.train model=attention task.candidate_count=16 training.total_updates=500
```

The old generated `default_cfg.yaml` artifact has been removed. Use `print_resolved_config=true` to inspect the composed runtime config.
