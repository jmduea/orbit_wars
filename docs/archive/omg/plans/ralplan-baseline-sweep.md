# Ralplan ADR: Workstation-Friendly Baseline Sweep

## Decision

Implement a config-and-docs-first baseline sweep workflow with three W&B templates and an evidence template. Defer trainer-integrated utilization telemetry until the manual/external guardrail workflow proves insufficient.

## Drivers

- The baseline must be useful for later hyperparameter comparisons.
- Workstation comfort is a hard constraint.
- Seed stability evidence is required before promotion.
- The implementation should stay aligned with existing Hydra and W&B sweep patterns.

## Alternatives Considered

- Helper script workflow: more repeatable, but too much scope before the first baseline run.
- Trainer-integrated guardrails: more observable, but adds dependency and platform risk in the hot training path.
- Full interaction sweep: strongest evidence, but too expensive for the initial baseline pass.

## Required Corrections From Review

- Control active load through the selected `format` profile because rollout groups override plain `training.num_envs` in mixed-format training.
- Make Stage 2 a fixed-config seed grid using root `seed.values`.
- Treat sentinel checks as bounded smoke checks with documented interpretation, not automated gates.
- Include a baseline evidence template so the workflow can record the selected overrides and W&B evidence after runs.

## Consequences

- The initial implementation is low-risk and easy to validate with config composition tests.
- Comfort status remains partly manual and should be recorded in W&B notes/tags.
- If this workflow is repeated often, the next step should be a helper script that promotes finalists and emits evidence summaries.
