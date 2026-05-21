# Deep Dive Spec: Config Consolidation

## Goal

Reorganize Orbit Wars configuration around clear responsibility groups so users can tell where to make a change, which knobs are sweep-safe together, and how to launch comparable Hydra or W&B sweeps.

## Decisions From Interview

- Remove `conf/experiment` as the primary abstraction for now rather than preserving broad experiment presets.
- Allow schema path changes when they produce cleaner ownership, provided they are migrated deliberately.
- Optimize both Hydra multirun workflows and W&B sweep YAMLs.

## Target Mental Model

Users should choose or override configs by intent:

- `model`: architecture and capacity.
- `task`: environment shape, player count, feature/action complexity, and other game distribution settings.
- `training`: PPO budget, optimizer, rollout, batching, and runtime training knobs.
- `format`: rollout/player-count mix and JAX rollout-group topology.
- `opponents`: opponent source policy, static opponent mixture, self-play enablement, and snapshot compatibility.
- `curriculum`: staged difficulty or staged opponent-family schedules.
- `reward`: reward shaping and terminal reward behavior.
- `telemetry`: metric groups, W&B enablement, W&B grouping/tags, and logging behavior.
- `artifacts`: checkpoint cadence, artifact pipeline, replay generation, validation backend, and retention.
- `campaign`: optional launch-time bundles for named sweep campaigns, if needed, but not broad inherited experiments.

## Old-to-New Ownership Map

| Current owner | Target owner | Initial package target | Compatibility behavior |
| --- | --- | --- | --- |
| `env.candidate_count`, `env.ship_bucket_count`, `env.max_fleets`, `env.player_count`, `env.max_ships`, `env.feature_history_steps`, `env.trajectory_shield_*` | `task.*` | `task` normalized to runtime `env` | Accept old `env.*` during migration; reject conflicts with `task.*`. |
| `env.reward_*`, `env.terminal_reward_mode`, `env.early_terminal_reward_shaping_*` | `reward.*` | `reward` normalized to runtime `env` reward fields | Accept old `env.reward_*` during migration; reject conflicts with `reward.*`. |
| `ppo.*`, reseed/plateau training controls | `training.*` | `training` normalized to runtime `ppo` and top-level training controls | Accept old `ppo.*` during migration; reject conflicts with `training.*`. |
| `training_format.*` | `format.*` | `format` normalized to runtime `training_format` | Accept old `training_format.*` during migration; reject conflicts with `format.*`. |
| `opponent`, `multi_opponent_mode`, `alternate_player_sides`, `self_play_enabled`, `self_play_update_interval`, `self_play_deterministic`, `opponent_mix.*` | `opponents.*` | `opponents` normalized to runtime top-level opponent fields and `opponent_mix` | Accept old fields during migration; reject conflicts with `opponents.*`. |
| `curriculum.enabled`, `curriculum.stages` | `curriculum.*` | `curriculum` | Keep here; stage progression remains curriculum-owned. |
| `curriculum.snapshot.*`, top-level `self_play_pool_size`, `self_play_snapshot_interval`, `self_play_latest_probability` | Choose one canonical owner before implementation: either `opponents.snapshot.*` or `curriculum.snapshot.*` | Normalized to the single runtime snapshot view | This is the highest-risk conflict; do not leave both canonical. |
| `telemetry.*`, `wandb.*` | `telemetry.metrics.*`, `telemetry.wandb.*` | `telemetry` normalized to runtime telemetry and W&B fields | Accept old `wandb.*` during migration; reject conflicts. |
| `artifact_pipeline.*`, `replay.*`, `checkpoint_retention.*`, `checkpoint_every`, `save_dir` | `artifacts.*` | `artifacts` normalized to runtime artifact fields | Accept old artifact paths during migration; reject conflicts. |

## Proposed Config Tree

```text
conf/
  config.yaml
  model/
  task/
  training/
  format/
  opponents/
  curriculum/
  reward/
  telemetry/
  artifacts/
  campaign/
  sweeps/
    hydra/
    wandb/
```

`campaign` should be optional and shallow. A campaign may select defaults across groups and set W&B/Hydra naming metadata, but it should not become a replacement for `experiment` inheritance.

## Schema Migration Principles

- Keep schema fields aligned with the target mental model, even if that means changing current paths.
- Move legacy top-level self-play fields under the owner that actually controls them.
- Prefer one owner per concept. For example, snapshot pool size and interval should not exist both as top-level `self_play_*` fields and under `curriculum.snapshot`.
- Keep compatibility shims only where they make migration safer; remove them once tests and docs are updated.
- Avoid `_global_` patches except for intentionally cross-cutting campaign files.
- Use a two-layer migration: first let Hydra compose the new responsibility surface, then normalize into the existing runtime `TrainConfig`; migrate runtime consumers only after compose and behavior tests are green.
- Add raw-config conflict checks for every renamed owner, following the existing `ppo.rollout_groups` vs `training_format.rollout_groups` validation style.

## Sweep Design

Hydra sweeps should primarily vary group choices or a small set of coherent knobs:

```bash
uv run python -m src.train -m \
  model=attention,entity_transformer_700k \
  training=budget_1k,budget_5k \
  task=two_player_default \
  reward=terminal_only
```

W&B sweep YAMLs should be split by campaign responsibility:

- `conf/sweeps/wandb/capacity.yaml`: model capacity/topology only, fixed budget/task/reward.
- `conf/sweeps/wandb/budget.yaml`: PPO budget/optimizer only, fixed model/task/reward.
- `conf/sweeps/wandb/reward.yaml`: reward shaping only, fixed model/budget/task.
- `conf/sweeps/wandb/task_complexity.yaml`: candidate count, ship buckets, player count, feature history, and format choices.
- `conf/sweeps/wandb/curriculum.yaml`: curriculum/opponent schedule only.
- `conf/sweeps/wandb/throughput.yaml`: runtime throughput knobs only, with task shape fixed unless the campaign explicitly studies shape scaling.

Each W&B sweep should set `wandb.group` and `wandb.tags` through config so comparison tables show campaign intent.

## Migration Plan

1. Add a compose matrix test for the current root default, every current `conf/experiment/*.yaml`, representative group combinations, W&B sweep parameter paths, and known conflict cases.
2. Inventory every `experiment=` reference in docs, scripts, sweep files, and tests; create an old-to-new launch mapping table before deleting any experiment files.
3. Decide the canonical default composition in `conf/config.yaml` without `experiment`, including whether `experiment: null` is removed immediately or quarantined for one migration phase.
4. Introduce new responsibility groups with explicit packages and minimal overlap, initially normalizing them into the existing runtime `TrainConfig` paths.
5. Resolve self-play ownership first. Pick the final owner for `enabled`, `pool_size`, `snapshot_interval`, `latest_probability`, `deterministic`, and update interval; update both Python opponent and JAX training paths through one normalized runtime view.
6. Move schema paths and runtime consumers in focused passes only after the new group surface composes cleanly.
7. Replace current W&B sweep files with responsibility-specific campaign files and fixed controls.
8. Update docs so the first thing users see is the responsibility map, old-to-new launch mapping, and sweep campaign map.
9. Remove or quarantine `conf/experiment/*.yaml` after equivalent group-based launch commands exist and stale `experiment=` references are gone outside migration docs.
10. Regenerate `default_cfg.yaml` and run focused config/default tests.

## Acceptance Criteria

- Docs include a responsibility map table, an old-to-new launch mapping, and at least one launch example per responsibility group.
- The root config composes without broad experiment presets.
- No concept has two active canonical owners.
- Hydra multirun examples vary responsibility groups or coherent axis sets.
- W&B sweep files are campaign-specific and include grouping/tag metadata.
- Every W&B sweep file can be dry-validated by translating one parameter set into a Hydra compose call.
- Config validation rejects ambiguous legacy combinations with clear messages during the migration period.
- Scripts and docs no longer hard-code `experiment=...` except in intentional migration notes.
- Python opponent code and JAX training code read self-play/curriculum settings through a single normalized runtime view.
- Focused config tests, curriculum tests, telemetry tests, and default config generation checks pass.

## Verification

Focused checks after implementation:

```bash
uv run python scripts/generate_default_cfg.py --check
uv run --group dev pytest tests/test_curriculum.py tests/test_telemetry.py
```

Add or update tests as needed for renamed schema paths, compatibility shims, compose matrix coverage, and W&B sweep dry-validation. If new config/default tests are added, include them in the focused verification command.
