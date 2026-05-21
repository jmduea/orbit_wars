# Deep Dive Trace: Config Consolidation

## Problem

Orbit Wars has moved to Hydra, but configuration ownership is still hard to read at a glance. The desired direction is to consolidate configs into responsibility-based groups and make both Hydra and W&B sweeps line up with those groups.

## Trace Lanes

### Lane 1: Experiment Presets Mix Responsibilities

Most likely explanation: `conf/experiment/*.yaml` acts as broad inherited recipe stacks even though the repo already has responsibility groups for model, training format, curriculum, opponent mix, replay, checkpoint retention, and telemetry.

Evidence for:

- `conf/config.yaml` already composes dedicated groups: `model`, `training_format`, `curriculum`, `opponent_mix`, `replay`, `checkpoint_retention`, and `telemetry`.
- `conf/experiment/full_training.yaml` mixes model selection, training format, run naming, artifact destination, checkpoint cadence, legacy opponent/self-play fields, environment shape, and PPO budget.
- `conf/experiment/attention_training.yaml`, `jax_training.yaml`, and `jax_mixed_2p_4p_training.yaml` inherit from other experiments, making unrelated domains propagate through preset names.
- The `full_training` inheritance chain currently carries `self_play_snapshot_interval: 50` into configs whose staged curriculum uses `curriculum.snapshot.interval_updates: 100`, which validation rejects when `curriculum.enabled` is true.

Evidence against:

- The docs already identify several canonical responsibility groups and warn against duplicate knobs, conflicting overrides, and too many moving parts per sweep.
- The validation layer fails loudly instead of silently accepting ambiguous legacy combinations.

Critical unknown resolved by interview:

- Experiment presets are not required as the primary user-facing abstraction. Removing them for now is acceptable.

### Lane 2: Hydra Groups vs Runtime Schema Paths

Most likely explanation: a conservative cleanup would preserve `TrainConfig` paths, but the desired design can permit path changes if they make responsibilities clearer.

Evidence for preserving paths:

- Runtime conversion merges Hydra output into `OmegaConf.structured(TrainConfig)`.
- Training code and tests consume stable paths such as `cfg.training_format.rollout_groups`, `cfg.curriculum`, `cfg.opponent_mix`, `cfg.ppo`, and `cfg.env`.
- Validation explicitly names current paths and rejects legacy conflicts.

Evidence for allowing changes:

- Several top-level self-play fields are already conceptual compatibility surfaces rather than clean ownership boundaries.
- `opponent_mix/self_play_curriculum.yaml` and `telemetry/default.yaml` use `_global_` patches because the current schema has cross-cutting ownership.
- If experiments are removed, this is a good moment to make the root config and schema match the mental model users should learn.

Critical unknown resolved by interview:

- Current user-facing override paths may change if the migration is deliberate, documented, and tested.

### Lane 3: Sweep Structure and Comparison Surfaces

Most likely explanation: docs are responsibility-aware, but executable sweep files are broad flat W&B parameter maps.

Evidence for:

- `conf/sweeps/wandb_attention_sweep.yaml` mixes env/action complexity, reward shaping, feature history, PPO budget/optimizer, and model capacity in one Bayesian sweep.
- `conf/sweeps/wandb_gnn_pointer_sweep.yaml` similarly mixes model topology, PPO settings, and feature history.
- `conf/sweeps/wandb_throughput_sweep.yaml` is throughput named but also sweeps environment/task shape knobs.
- W&B defaults leave `wandb.group` null and `wandb.tags` empty, so comparison surfaces rely mostly on flat scalar keys.

Evidence against:

- `docs/experiments.md` already has a goal-oriented playbook, sweep-safe matrix, and output hygiene recommendations.
- Hydra output metadata preserves composed config and override provenance per job.

Critical unknown resolved by interview:

- Optimize both Hydra multirun group selection and W&B sweep YAML campaigns.

## PDF Guidance Incorporated

The configuration-management section of `docs/production-ready-data-science.pdf` supports this direction:

- Separate configuration from code for maintainability.
- Use logical grouping of related configurations to manage complexity.
- Use Hydra command-line overrides and multirun execution for systematic experimentation.
- Use interpolation to reduce duplication.
- Use hierarchical YAML structure.
- Avoid secrets in config files.

## Convergence

The core issue is not Hydra itself. It is a mismatch between the desired responsibility model and the launch artifacts users interact with most: broad inherited experiment presets and broad flat sweep files.

## Recommended Direction

Remove `experiment` as the main abstraction for now. Make the root config compose responsibility groups directly, and make sweeps vary those group choices or tightly scoped fields. Permit schema cleanup where it removes cross-cutting legacy ownership, but require migration tests and documentation for every path change.
