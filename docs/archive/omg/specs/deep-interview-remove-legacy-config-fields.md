# Remove Legacy Config Fields

## Goal

Make the Hydra public config surface and runtime `TrainConfig` shape match the responsibility-group model exactly.

Runtime code should use `cfg.task`, `cfg.reward`, `cfg.training`, `cfg.format`, `cfg.opponents`, `cfg.artifacts`, `cfg.telemetry`, and `cfg.model` rather than legacy `cfg.env`, `cfg.ppo`, `cfg.training_format`, `cfg.opponent_mix`, `cfg.wandb`, flat self-play/opponent fields, or flat artifact fields.

## Constraints

- Remove legacy Hydra field anchors from `conf/config.yaml`.
- Remove old CLI compatibility for legacy paths such as `env.*`, `ppo.*`, `training_format.*`, `opponent_mix.*`, `wandb.*`, `artifact_pipeline.*`, `replay.*`, `checkpoint_retention.*`, `save_dir`, `checkpoint_every`, and top-level `self_play_*`/opponent fields.
- Keep behavior stable by migrating runtime consumers to the new nested fields in the same pass.
- Keep config files glanceable and responsibility-owned.
- Preserve canonical group defaults and W&B sweep composition.
- Do not preserve backward compatibility for old Hydra override names.

## Non-Goals

- Do not reintroduce experiment presets.
- Do not keep duplicate public config owners.
- Do not keep compatibility tests that assert legacy override parsing.
- Do not rewrite unrelated environment, PPO, or telemetry behavior.

## Acceptance Criteria

- `conf/config.yaml` contains only root scalar controls and responsibility-group defaults, with no null legacy anchor block.
- `TrainConfig` no longer exposes legacy flat config fields or legacy section names for runtime use.
- Runtime code no longer references `cfg.env`, `cfg.ppo`, `cfg.training_format`, `cfg.opponent_mix`, `cfg.wandb`, `cfg.artifact_pipeline`, `cfg.replay`, `cfg.checkpoint_retention`, flat `cfg.self_play_*`, flat `cfg.opponent`, flat `cfg.multi_opponent_mode`, flat `cfg.alternate_player_sides`, flat `cfg.save_dir`, or flat `cfg.checkpoint_every`.
- Tests and docs describe only canonical responsibility paths.
- Legacy override attempts fail with normal Hydra struct errors or equivalent validation errors.
- Focused config/curriculum/telemetry tests pass after migration.

## Assumptions Resolved

- User prefers removing flat runtime fields too, not just public anchors.
- User prefers runtime schema to match public responsibility groups exactly.

## Interview Transcript

- Scope question: user selected `Remove flat runtime fields too`.
- Runtime shape question: user selected `Match public responsibility groups exactly`.
