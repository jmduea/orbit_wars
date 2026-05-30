# Telemetry Metrics Cleanup Spec

Generated: 2026-05-21
Workflow: deep-interview --standard
Final ambiguity: 16%

## Goal

Clean up the training metrics and telemetry system so metric collection is centralized, documented, configurable by explicit group booleans, and less noisy for WandB/local logs. The cleanup should prefer a clearer metric schema over preserving existing flat metric names.

## Context

- Telemetry is currently emitted through `TelemetryLogger` in `src/telemetry.py` and built largely from records assembled in `src/jax_train.py`, rollout metrics from `src/jax_ppo.py`, and curriculum telemetry from `src/curriculum.py`.
- Existing config only exposes coarse WandB toggles under `conf/telemetry/default.yaml` and `WandBConfig` in `src/conf_schema.py`.
- There are no per-metric or per-group toggles, no dedicated telemetry tests, and no central metric schema/registry.
- Metric names may break as part of this cleanup; legacy alias layers and dashboard migration helpers are explicitly out of scope.

## Requirements

- Introduce a central metric registry/schema that owns emitted metric names, groups, and concise descriptions.
- Add explicit boolean config options per metric group, rather than a single enum level or named preset system.
- Group metrics into discoverable categories such as core progress, losses, timing, curriculum, opponent composition, game state, action/decision behavior, trajectory shield/debug, or other categories that fit the current code.
- Log/write only enabled metric groups to WandB and local JSONL output.
- Avoid computing disabled expensive metrics where practical, while preserving training behavior.
- Keep the default config intentional and reasonably small/noise-aware; exact default group choices should be decided during implementation from the current metric inventory.
- Add focused tests for metric registry behavior, config parsing/defaults, and group filtering.

## Non-Goals

- Do not add legacy metric aliases or dashboard migration compatibility helpers.
- Do not preserve every existing metric name for backward compatibility.
- Do not introduce named presets or a single telemetry-level enum in the first pass.
- Do not refactor unrelated training, PPO, environment, or WandB artifact behavior beyond what is needed for telemetry cleanup.

## Acceptance Criteria

- Metric definitions are centralized enough that a developer can find metric names, groups, and meanings without chasing large record dictionaries.
- Hydra/config schema exposes explicit boolean toggles for metric groups.
- Disabling a metric group prevents that group's metrics from appearing in WandB/local JSONL records.
- Disabled metric groups avoid unnecessary expensive collection where the current architecture makes that practical.
- Existing tests pass, and new focused telemetry/config tests cover registry and filtering behavior.
- Generated default config is updated if schema/config defaults change.

## Assumptions Resolved

- The cleanup should combine noise reduction, registry/schema refactoring, and runtime/I/O overhead reduction.
- Breaking metric names is acceptable if the result is clearer and better organized.
- A real registry is preferred over a simple logger filter.
- The first implementation should include registry, explicit group config, docs/tests, and practical lazy collection, but no compatibility alias layer.

## Ontology

- Metric: A named scalar or structured value emitted during training.
- Metric group: A boolean-configurable category controlling collection and emission.
- Registry/schema: The central source of truth for metric names, groups, and descriptions.
- Telemetry sink: WandB and local JSONL output paths that receive filtered records.
- Expensive metric: A metric whose computation adds nontrivial runtime, JAX device transfer, aggregation, or I/O overhead.

## Interview Transcript

1. Primary outcome: combine reducing metric noise, adding a registry/schema, and reducing runtime/I/O overhead.
2. Compatibility: prefer breaking metric names for a clearer and better organized schema.
3. Success shape: combine config-toggled metric groups, documented/tested definitions, and avoiding disabled expensive collection.
4. Registry challenge: build the registry; long-term clarity matters.
5. Config surface: explicit boolean per group only.
6. Simplification: include the expected cleanup pieces, but no backcompat helpers or legacy alias layer.
