# Hydra Migration Guide

Orbit Wars has moved from broad preset launches to responsibility-based Hydra groups.

## Current Launch Style

Compose from `conf/config.yaml` and override the group that owns the thing you want to change:

```bash
uv run python -m src.train model=attention training.total_updates=1000
uv run python -m src.train task.candidate_count=16 reward.reward_production_delta=0.01
uv run python -m src.train format=mix_2p_4p_16env opponents=self_play_curriculum
```

## Old-to-New Examples

| Old intent | New command shape |
| --- | --- |
| Attention model run | `uv run python -m src.train model=attention` |
| Larger candidate set | `uv run python -m src.train model=attention task.candidate_count=16` |
| Mixed 2p/4p rollout | `uv run python -m src.train format=mix_2p_4p_16env` |
| Short budget scan | `uv run python -m src.train training.total_updates=250` |
| Reward shaping scan | `uv run python -m src.train reward.reward_production_delta=0.01` |

## Removed Override Aliases

The public config surface is the responsibility-group schema. Old nested roots are not compatibility aliases; update commands to `training.*`, `task.*`, `reward.*`, `telemetry.*`, `artifacts.*`, `format.*`, `opponents.*`, or `curriculum.*`.

## Source of Truth

The source of truth is `conf/config.yaml` plus config groups under `conf/`. The generated `default_cfg.yaml` artifact has been removed.
