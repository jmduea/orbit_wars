# Running Orbit Wars With Hydra

Orbit Wars uses Hydra-first training commands with responsibility-based config groups.

## Basic Launches

Default training:

```bash
uv run python -m src.train
```

Resolved config only:

```bash
uv run python -m src.train print_resolved_config=true
```

Override by responsibility:

```bash
uv run python -m src.train model=attention training.total_updates=1000
uv run python -m src.train task.candidate_count=16 reward.reward_production_delta=0.01
uv run python -m src.train format=mix_2p_4p_16env opponents=self_play_curriculum
```

## Responsibility Map

| Intent | Primary group or path | Nearby knobs to avoid |
| --- | --- | --- |
| Change model capacity | `model`, `model.hidden_size`, `model.attention_heads` | `training.*`, `task.*` |
| Change training budget | `training.total_updates`, `training.num_envs`, `training.rollout_steps`, `training.lr` | `model.*`, `reward.*` |
| Change task complexity | `task.candidate_count`, `task.ship_bucket_count`, `task.player_count`, `format` | `training.*`, `model.*` |
| Change reward shaping | `reward.*` | `task.*`, `training.*` |
| Change opponent behavior | `opponents`, `opponents.mix.*`, `opponents.snapshot.*` | `curriculum.stages.*` unless testing stage schedules |
| Change staged progression | `curriculum`, `curriculum.stages.*` | `opponents.self_play.*` unless changing the profile |
| Change logging | `telemetry.metric_groups.*`, `telemetry.wandb.*` | training and task knobs |
| Change artifacts | `artifacts.*` | training budget unless checkpoint cadence is the variable |

`opponents.snapshot` owns historical policy pool size, snapshot cadence, selection, and fallback. `curriculum` owns stage progression and stage-local opponent-family weights.

## Hydra Multirun

Use `-m` for Cartesian sweeps over coherent axes:

```bash
uv run python -m src.train -m \
  model=attention,entity_transformer_700k \
  training.total_updates=250,500 \
  task.candidate_count=8,16
```

Runs are stored under a campaign-oriented output root by default:

```text
outputs/campaigns/<campaign>/runs/<run_id>/
```

The default campaign is `scratch`. Set `output.campaign=<slug>` for named experimental questions or comparison frames, such as `capacity`, `baseline-stage2`, or `submission-candidate-eval`. Each run envelope contains Hydra's `.hydra/` snapshot, `manifest.json`, `logs/`, `checkpoints/`, queue state, and evaluation outputs. Model architecture remains manifest/W&B metadata rather than the physical directory root.

For larger campaigns, prefer setting the campaign rather than overriding Hydra's output directories directly:

```bash
uv run python -m src.train -m \
  model=attention,entity_transformer_700k \
  training.total_updates=250,500 \
  output.campaign=capacity
```

Generated W&B run files are routed into the run envelope. W&B artifact download and staging caches are routed under `outputs/cache/`, so top-level `wandb/` and `artifacts/` are legacy/local leftovers rather than canonical locations for new training runs.

## W&B Sweeps

Executable W&B sweep templates live in `conf/sweeps/wandb/`:

- `capacity.yaml`
- `baseline_stage1_comfort.yaml`
- `baseline_stage2_stability.yaml`
- `baseline_sentinels.yaml`
- `budget.yaml`
- `reward.yaml`
- `task_complexity.yaml`
- `curriculum.yaml`
- `throughput.yaml`

Each template sets `telemetry.wandb.group` and `telemetry.wandb.tags` so run tables carry campaign intent.

Use [Workstation-Friendly Baseline Sweep](baseline_sweep.md) when selecting a default comparison baseline that balances performance, throughput, stability, and active workstation comfort. The first promoted baseline is recorded in [Workstation-Friendly Baseline Sweep Results](baseline_sweep_results.md).

## Config Source

Only the responsibility-group paths above are public runtime config. Removed aliases such as old PPO, environment, and W&B roots are rejected instead of translated.

The generated `default_cfg.yaml` artifact has been removed; `conf/config.yaml` plus its selected groups are the source of truth.
