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
| `curriculum.snapshot.*`, top-level `self_play_pool_size`, `self_play_snapshot_interval` | `opponents.snapshot.*` | Normalized to current runtime fields for one migration phase | `opponents.snapshot` is canonical. Old top-level snapshot fields and old `curriculum.snapshot.*` normalize into it temporarily; raw conflicts are rejected. |
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
- Move legacy top-level self-play fields under `opponents`, the owner that controls opponent source policy and self-play mechanics.
- Prefer one owner per concept. Snapshot pool size, interval, deterministic selection, and fallback belong under `opponents.snapshot`; static latest/historical weighting belongs under `opponents.mix`; curriculum owns staged progression and stage-local opponent-family weights only.
- Keep compatibility shims only where they make migration safer; remove them once tests and docs are updated.
- Avoid `_global_` patches except for intentionally cross-cutting campaign files.
- Use a two-layer migration: first let Hydra compose the new responsibility surface, then normalize into the existing runtime `TrainConfig`; migrate runtime consumers only after compose and behavior tests are green.
- Add raw-config conflict checks for every renamed owner, following the existing `ppo.rollout_groups` vs `training_format.rollout_groups` validation style.
- Treat `default_cfg.yaml` as a legacy generated runtime-schema artifact, not a source of truth. Prefer removing or quarantining it unless external consumers of `load_hydra_train_config(path)` are explicitly preserved.

## Opponent Field Ownership

`opponents` is the canonical public owner for self-play and opponent-source behavior:

| Target field | Current source fields | Meaning |
| --- | --- | --- |
| `opponents.self_play.enabled` | `self_play_enabled` | Enable self-play policy sources. |
| `opponents.self_play.update_interval` | `self_play_update_interval` | Current-policy update cadence. |
| `opponents.self_play.deterministic` | `self_play_deterministic` | Deterministic sampling for current self-play policies. |
| `opponents.mode` | `opponent`, `multi_opponent_mode`, `alternate_player_sides` | Opponent source mode and side assignment. |
| `opponents.mix.weights.*` | `opponent_mix.weights.*`, `self_play_latest_probability` | Static opponent-family mixture. Old `self_play_latest_probability` maps to latest/historical weights and should be removed as a canonical field. |
| `opponents.mix.temperature` | `opponent_mix.temperature` | Static mixture sampling temperature. |
| `opponents.snapshot.pool_size` | `self_play_pool_size`, `curriculum.snapshot.pool_size` | Historical policy pool size. |
| `opponents.snapshot.interval_updates` | `self_play_snapshot_interval`, `curriculum.snapshot.interval_updates` | Historical snapshot cadence. |
| `opponents.snapshot.deterministic` | `curriculum.snapshot.deterministic` | Deterministic sampling for historical snapshots. |
| `opponents.snapshot.selection` | `curriculum.snapshot.selection` | Historical snapshot selection strategy. |
| `opponents.snapshot.fallback` | `curriculum.snapshot.fallback` | Fallback when no historical snapshot is available. |

`curriculum` owns stage progression and stage-local opponent-family weights. During phase 1, the normalizer may write `opponents.snapshot.*` into the current runtime `curriculum.snapshot` fields for JAX compatibility, but `curriculum.snapshot` is not canonical in the public config surface.

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
5. Resolve self-play ownership first. Implement `opponents.self_play`, `opponents.mix`, and `opponents.snapshot` as the canonical public surface for enablement, update cadence, deterministic sampling, static latest/historical weights, pool size, and snapshot interval; update both Python opponent and JAX training paths through one normalized runtime view.
6. Move schema paths and runtime consumers in focused passes only after the new group surface composes cleanly.
7. Replace current W&B sweep files with responsibility-specific campaign files and fixed controls.
8. Replace script APIs that expose experiment names. `scripts/benchmark_jax_rl.py` and `scripts/compare_attention_candidates.py` should accept explicit Hydra override lists or first-class group flags, and should print group-based commands only.
9. Decide and execute `default_cfg.yaml` disposition. Recommended default: remove or quarantine `default_cfg.yaml`, `default_train_config_path`, and `scripts/generate_default_cfg.py` unless a real external consumer is identified; otherwise document it as a non-canonical runtime reference.
10. Update docs so the first thing users see is the responsibility map, old-to-new launch mapping, and sweep campaign map.
11. Remove or quarantine `conf/experiment/*.yaml` after equivalent group-based launch commands exist and stale `experiment=` references are gone outside migration docs.
12. Run focused config, curriculum, telemetry, sweep, and script verification.

## Acceptance Criteria

- Docs include a responsibility map table, an old-to-new launch mapping, and at least one launch example per responsibility group.
- The root config composes without broad experiment presets.
- No concept has two active canonical owners.
- Hydra multirun examples vary responsibility groups or coherent axis sets.
- W&B sweep files are campaign-specific and include grouping/tag metadata.
- Every W&B sweep file can be dry-validated by translating one parameter set into a Hydra compose call.
- Config validation rejects ambiguous legacy combinations with clear messages during the migration period.
- Scripts and docs no longer hard-code `experiment=...` except in intentional migration notes.
- Benchmark/comparison scripts accept group overrides or group flags instead of experiment names.
- `default_cfg.yaml` is either removed/quarantined, or explicitly documented and tested as a non-canonical runtime-schema reference.
- Python opponent code and JAX training code read self-play/curriculum settings through a single normalized runtime view.
- Focused config tests, curriculum tests, and telemetry tests pass.
- If `default_cfg.yaml` is retained, generator/check tests pass and docs mark it non-canonical; if it is removed/quarantined, generator/check requirements and references are removed from `AGENTS.md`, README/docs, and tests.

## Verification

Focused checks after implementation:

```bash
uv run --group dev pytest tests/test_curriculum.py tests/test_telemetry.py
```

Add `tests/test_config_consolidation.py` or equivalent coverage for root compose, every new group default, old-to-new launch mapping, raw conflict detection, representative multirun combinations, script defaults, and W&B sweep dry-validation. If `default_cfg.yaml` is retained, also keep a generator/check test; if it is removed, delete the generator/check requirement and update `AGENTS.md` plus docs that mention it.
