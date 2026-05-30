# RALPLAN-DR Initial Plan: Telemetry Metrics Cleanup

Date: 2026-05-21
Spec: `.omg/specs/deep-interview-telemetry-metrics-cleanup.md`
Mode: SHORT
Status: Revised after critic rejection; awaiting second critic review

## Context

The approved spec asks for a telemetry cleanup that centralizes metric definitions, adds explicit boolean metric-group configuration, filters both WandB and JSONL outputs, avoids disabled expensive collection where practical, and adds focused tests. Backward-compatible alias layers, dashboard migration helpers, named presets, and a single telemetry-level enum are explicitly out of scope.

Current repo facts:

- Python 3.12 project managed with `uv`.
- WandB logging lives in `src/telemetry.py` through `TelemetryLogger`.
- Config schema lives in `src/conf_schema.py`; the current `WandBConfig` has sink toggles only.
- Hydra telemetry config lives in `conf/telemetry/default.yaml` and currently configures `wandb:` only.
- Main training metric records are assembled in `src/jax_train.py` as one large flat dictionary written to JSONL and WandB.
- Rollout diagnostics are produced in `src/jax_ppo.py`, including many action, game-state, opponent, and trajectory-shield metrics.
- Curriculum telemetry is produced in `src/curriculum.py` and merged into training records.

## Principles

1. Registry first: emitted metric names, groups, descriptions, and internal-vs-output intent should be discoverable from one central source.
2. Explicit configuration: use boolean group fields in config, not presets, aliases, or broad enum levels.
3. One filtering contract for all sinks: JSONL and WandB should receive the same enabled metric surface, with WandB flattening happening after filtering.
4. Preserve training behavior: filtering must not remove internal metrics required for curriculum, seed scheduling, plateau logic, or PPO updates.
5. Lazy where practical: avoid host transfer, record assembly, and diagnostic computation for disabled groups when the existing JAX/training structure makes that safe and low-risk.
6. Safety first: core progress, sweep objective, and checkpoint-retention metrics are protected by default so filtering does not silently degrade training operations.

## Top Decision Drivers

1. Maintainability: future contributors should add or find metrics through the registry instead of chasing large dictionaries across training code.
2. Training safety: JAX compilation, curriculum transitions, seed scheduling, and PPO updates must keep identical behavior unless a metric is only observational.
3. Noise and overhead reduction: disabled groups should disappear from both sinks and avoid unnecessary host transfers/computation where practical.

## Viable Implementation Options

### Option A: Central Registry Plus Staged Producer/Filter Integration (Recommended)

Create a metric registry module/API, add a `TelemetryConfig` with explicit boolean group fields, then integrate the registry at the main host-side assembly boundary in `src/jax_train.py` and at practical expensive diagnostic boundaries in `src/jax_ppo.py` and `src/curriculum.py`.

Pros:

- Satisfies the full spec: registry, group booleans, sink filtering, tests, and practical lazy collection.
- Keeps the biggest refactor at natural boundaries: record assembly, host transfer key selection, and rollout diagnostics.
- Allows a safe distinction between output metrics and internal-required metrics.
- Avoids building a legacy alias or compatibility layer.

Cons:

- More implementation work than sink-only filtering.
- Requires careful metric inventory and tests to avoid accidentally suppressing values needed by training control.
- Some JAX diagnostics may still be computed if removing them would create too much compilation or correctness risk in the first pass.

### Option B: Registry With Sink-Only Filtering

Create the registry and config booleans, but leave metric collection and record assembly mostly intact. Filter records immediately before JSONL append and WandB logging.

Pros:

- Lowest implementation risk and easiest to test quickly.
- Gives a central schema and removes output noise.
- Minimal interaction with JAX compilation and rollout internals.

Cons:

- Does not meet the practical lazy collection requirement well.
- Keeps large host transfers and large record assembly in place.
- Risks becoming a superficial cleanup rather than the requested telemetry architecture improvement.

### Option C: Fully Typed Metric Producers Per Subsystem

Build subsystem-specific producer objects or typed payload builders for rollout, PPO loss, curriculum, timing, opponent composition, and debug metrics, all registered centrally.

Pros:

- Strongest long-term architecture and testability.
- Makes ownership boundaries very explicit.
- Could substantially reduce accidental metric drift.

Cons:

- Too large for a first cleanup pass.
- Higher risk around JAX static shapes, compilation churn, and training-loop behavior.
- More likely to expand into an unrelated refactor, which the spec explicitly discourages.

## Recommended Option

Use Option A.

Implement a central registry and explicit boolean group config first, then route both sinks through the registry filter. Add practical lazy behavior in the highest-value places: host-transfer key selection in `src/jax_train.py`, optional assembly of expensive nested/event groups, and gated JAX rollout diagnostics where the flag can be captured in compiled rollout group config without destabilizing training behavior.

## Initial Decision Record

Decision: Build a central registry with group-aware filtering and staged lazy collection.

Drivers: maintainability, training safety, output noise/overhead reduction.

Alternatives considered: sink-only filtering and full producer-object redesign.

Why chosen: It is the smallest option that meets all approved requirements while keeping the refactor near current telemetry boundaries.

Consequences: Some metric names will intentionally change or be grouped more clearly; dashboards depending on old flat names may break; a few diagnostics may remain computed in the first pass if skipping them would risk training correctness or large JAX churn.

Follow-ups: After the first pass lands, inspect real WandB/JSONL output from a short run and tune default enabled groups if the default surface is still noisy.

## Protected Metric And Event Policy

The registry must distinguish three concepts:

- `enabled output`: metrics included because their group boolean is true.
- `protected output`: metrics always included in the filtered output record by default because repo operations depend on them.
- `internal required`: metrics always computed for training control, even if they are not emitted as ordinary output metrics.

Concrete protected/default metrics for the first implementation:

| Metric | Why protected | Enforcement |
| --- | --- | --- |
| `episode_reward_mean` | Default `checkpoint_retention.best_metric_name` and default plateau metric | Always computed; included in JSONL/WandB as a canonical registered metric |
| `overall_win_rate` | Common sweep objective and curriculum promotion metric | Enabled by default in core progress; kept in protected known-sweep set |
| `env_steps_per_sec` | Known throughput sweep objective | Enabled by default in timing/progress; kept in protected known-sweep set |
| `win_rate_2p` | Curriculum/sweep objective | Enabled by default in core progress |
| `first_place_rate_4p` | Curriculum/sweep objective | Enabled by default in core progress |
| `average_reward`, `average_episode_reward`, `survival_time`, `score_share`, `approx_kl` | Curriculum promotion allowlist in `src/curriculum.py` | Always computed for curriculum when needed; emitted according to group unless configured as retention/plateau/sweep objective |
| `update`, `total_env_steps`, `completed_episodes`, `samples` | Record identity/progress | Always emitted as core progress |

Validation and enforcement rules:

- Validate `checkpoint_retention.best_metric_name` and `plateau_metric` against canonical registry names during config validation/training startup. If metric names change, update the configs to canonical names rather than adding compatibility aliases.
- If a configured retention or plateau metric belongs to a disabled output group, force it into the filtered JSONL/WandB output and emit it with its registry name.
- Known sweep objectives from `conf/sweeps/*.yaml`, including `overall_win_rate` and `env_steps_per_sec`, should remain in default-enabled core/progress/timing groups. Add a test that parses all `conf/sweeps/*.yaml` metric names and fails if any are not registered and default-enabled/protected. External sweep objectives cannot be discovered at runtime, so users who introduce a new objective must ensure it is registered and enabled or configured as a protected objective if that support is added later.
- Do not add legacy metric aliases, replacement mappings, or dashboard compatibility helpers. Configured operational metric names must be canonical registry names.

## Emission Inventory To Cover

The implementation must inventory and classify every record shape before refactoring. At minimum:

- Main update records in `src/jax_train.py`: scalar training progress, PPO losses, timing, rollout diagnostics, curriculum telemetry, opponent mix, game-state analytics, historical pool state, embedded reseed events, and embedded curriculum phase events.
- Checkpoint result events in `src/jax_train.py`: `event=checkpoint_result`, status, final flag, reason, and error fields.
- Checkpoint queued/result events in `src/artifact_pipeline.py`: queue/job metadata and status records if they are written to the same JSONL/WandB surfaces.
- Historical snapshot events in `src/jax_train.py`: `event=historical_snapshot_added`, snapshot id, update, and parameter size.
- Operational artifact/replay logging through `TelemetryLogger.log_checkpoint`, `log_artifact`, and `log_replay`; these are sink operations, not scalar metrics, and should remain under existing WandB artifact toggles unless intentionally brought into event registry scope.

Event handling decision: add an explicit `events` metric group defaulting to true, and route sparse event records through the same registry-aware filter using `record_kind="event"`. Checkpoint-retention event identity fields should remain protected operational fields. Embedded event lists such as `reseed_events` and `curriculum_phase_events` should be grouped under `events` or the owning subsystem group, with the implementation documenting the chosen classification in the registry.

## Implementation Phases

### Phase 1: Metric Inventory and Registry Contract

Add a central metric registry in a dependency-light standalone module such as `src/metric_registry.py`, with metric definitions containing name, group, description, and whether the metric is output-only, internally required, or protected for sweep/checkpoint-retention behavior.

Acceptance criteria:

- Registry defines explicit groups such as `core_progress`, `losses`, `timing`, `curriculum`, `opponent_composition`, `game_state`, `action_decision`, and `trajectory_shield_debug`, adjusted to the actual inventory during implementation.
- Registry includes all currently emitted training-loop metrics that remain part of the cleaned telemetry surface.
- Registry can answer which metric names are enabled for a given config.
- Internal metrics required by curriculum, seed scheduling, plateau logic, checkpoint retention, or training updates are not dropped from computation just because their output group is disabled.
- Core progress, sweep objective, and checkpoint-retention metrics are protected by default while still allowing the cleaned metric schema to break old flat names.
- Registry includes record-kind awareness for update records and sparse event records.

### Phase 2: Config Schema and Hydra Defaults

Add a telemetry group config to `src/conf_schema.py`, probably separate from `WandBConfig`, so sink credentials remain under `wandb` while metric group controls live under `telemetry.metric_groups` or equivalent explicit boolean fields. Update `conf/telemetry/default.yaml` and regenerate/check `default_cfg.yaml` if schema defaults change.

Acceptance criteria:

- Config exposes explicit booleans per metric group.
- Defaults are intentionally small/noise-aware but still include essential progress, loss, timing, events, known sweep objectives, and checkpoint-retention metrics needed for routine training monitoring and operations.
- Hydra composition accepts overrides like `telemetry.metric_groups.trajectory_shield_debug=false` without named presets or enum levels.
- No legacy alias/backcompat helper fields are added.
- Config validation or training startup rejects/flags `checkpoint_retention.best_metric_name` and `plateau_metric` values that cannot be resolved to a registered metric.

### Phase 3: Unified Filtering for JSONL and WandB

Build the full internal record, derive a filtered output record once at the training record boundary, and pass the same filtered record to both `append_jsonl` and `TelemetryLogger.log`. Keep WandB flattening after filtering so nested enabled groups are flattened consistently only for WandB.

Acceptance criteria:

- Disabling a group removes that group's metrics from local JSONL records.
- The same disabled group is absent from WandB payloads.
- Required structural fields such as update/step identifiers remain available according to the registry's core group policy.
- Checkpoint-retention and sweep objective metrics remain emitted or otherwise available by default, so best-k pruning and sweep objectives do not silently degrade.
- Sparse event records and embedded event payloads are classified and filtered consistently, rather than bypassing the registry by accident.
- Filtering behavior is deterministic and covered by unit tests without requiring a live WandB run.

### Phase 4: Practical Lazy Collection and Host Transfer Reduction

Use the registry/config to narrow `src/jax_train.py` rollout scalar host transfers to enabled output metrics plus internal-required/protected metrics. Gate expensive or debug-heavy rollout diagnostics in `src/jax_ppo.py` where safe, using config captured in the jitted rollout group. Keep metrics needed by curriculum update, plateau detection, seed scheduling, checkpoint retention, and PPO losses computed regardless of output filtering. Disabling trajectory-shield telemetry must not bypass shield mask computation while `env.trajectory_shield_enabled` is true.

Acceptance criteria:

- Disabled observational groups avoid record assembly and host transfer where possible.
- Expensive trajectory-shield/debug or game-state aggregates are gated where doing so does not change environment behavior or PPO update semantics.
- Training behavior is preserved for curriculum transitions and seed scheduling even when output groups are disabled.
- Any diagnostic that remains eagerly computed has a clear reason in code structure or tests, not accidental omission.

### Phase 5: Focused Tests and Documentation Touch-Up

Add targeted tests for registry uniqueness/group filtering, config parsing/defaults, JSONL/WandB filtering, and one short training-loop smoke path with at least one disabled group. Update docs only where they help developers find and use the new metric groups.

Acceptance criteria:

- New tests cover registry behavior, explicit boolean config parsing, and filtering for both sinks.
- Existing relevant tests continue to pass, especially curriculum and JAX PPO coverage.
- `default_cfg.yaml` is regenerated or checked if schema/defaults change.
- Developer-facing documentation names the groups and points to the registry as the source of truth.

## Test Strategy

Unit tests:

- Add registry tests for unique metric names, valid groups, concise descriptions, enabled-name selection, and unknown/unregistered metric handling.
- Add config tests that instantiate `TrainConfig`, compose Hydra defaults, and override individual group booleans.
- Add filtering tests using a synthetic mixed record to prove disabled groups are removed and enabled/core groups remain.
- Add `TelemetryLogger` tests with a fake WandB object or direct filter call so no live WandB service is needed.
- Add protected-metric tests proving `checkpoint_retention.best_metric_name` and `plateau_metric` remain available after filtering even if their ordinary group is disabled.
- Add event filtering tests for `checkpoint_result`, `historical_snapshot_added`, and embedded `reseed_events`/`curriculum_phase_events` classification.

Integration/smoke tests:

- Add or extend a short `run_jax_training` test that disables a noisy group and verifies the resulting JSONL record omits that group while preserving core progress fields.
- Extend curriculum coverage enough to ensure curriculum state/event behavior still works when curriculum output metrics are disabled or enabled according to the new config.
- Add or extend checkpoint-retention coverage to prove best-k retention can still read its configured metric from filtered JSONL records.
- Add a sweep-objective/default coverage check that known repo sweep metric names remain registered and default-enabled.

Recommended verification commands:

- `uv run python scripts/generate_default_cfg.py --check`
- `uv run --group dev pytest tests/test_curriculum.py tests/test_jax_ppo.py`
- `uv run --group dev pytest tests/test_telemetry.py` once added
- `uv run --group dev pytest` for final validation if runtime permits

## Open Risks

- Metric inventory drift: large existing flat dictionaries make it easy to miss a metric unless the executor inventories all emitted keys before refactoring.
- Internal/output confusion: metrics like `overall_win_rate`, `approx_kl`, and `cfg.plateau_metric` may be needed for training control even when their output group is disabled.
- JAX recompilation and shape risk: gating diagnostics inside jitted rollout collection must use static config choices and avoid changing transition shapes consumed by PPO.
- Default group choice: the spec intentionally leaves exact defaults to implementation; executor should choose a small default surface and verify it remains useful for normal monitoring.
- Existing tests gap: this checkout does not currently contain dedicated telemetry tests, so new coverage must be introduced rather than only extended.
- External sweep objectives outside `conf/sweeps/` cannot be inferred automatically; the first implementation protects known repo sweep objectives and validates configured runtime objectives that are present in `TrainConfig`.

## Analyst Gap Check

No additional user preference questions are needed for the initial implementation plan. The approved spec resolves the key policy choices: central registry, explicit booleans, filtering for WandB/JSONL, practical lazy collection, tests, and no legacy alias layer. The only remaining implementation judgment is exact default enabled groups, which should be decided from the metric inventory during Phase 1/2 and validated by tests plus a short training smoke run.

## Success Criteria

- A developer can find metric names, groups, and meanings in the registry.
- Metric group booleans are available through schema and Hydra config.
- Disabled groups are absent from both JSONL and WandB payloads.
- Disabled groups avoid unnecessary expensive collection where practical without changing training behavior.
- Checkpoint retention still reads its configured metric from filtered JSONL records.
- Known repo sweep objective metrics are registered and default-enabled.
- Sparse event records are either registry-filtered or explicitly documented as operational events outside scalar metric filtering.
- New tests cover registry, config, filtering, and smoke behavior.
- No compatibility alias layer, dashboard migration helper, named preset system, or telemetry-level enum is introduced.
