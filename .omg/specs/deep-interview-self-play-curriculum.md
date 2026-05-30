# Deep Interview Spec: Self-Play Curriculum

Date: 2026-05-21
Status: Draft pending approval
Ambiguity: 17%

## Goal

Implement a trustworthy staged self-play curriculum for JAX mixed 2-player/4-player training. The curriculum must progressively mix latest-policy opponents, frozen previous policy snapshots, scripted exploiters, and simple baseline opponents while making the active opponent distribution observable and testable.

The implementation should replace the current split-brain scheduling model with one explicit curriculum schema, so progressive difficulty is controlled from a single authoritative surface rather than overlapping `opponent_mix.curriculum` and `training_format.phases` behavior.

## Brownfield Context

Current repo support is partial:

- `opponent_mix.curriculum` and `OpponentRegistry` support weighted opponent schedules in the JAX 4-player rollout path.
- `training_format.phases` and `CurriculumController` can mutate opponent weights separately, creating precedence and observability risk.
- Self-play pool fields such as `self_play_snapshot_interval` are validated but not wired into the active JAX trainer.
- `SelfPlayOpponentPool.add_snapshot` exists, but no active trainer caller populates it.
- The JAX rollout path exposes an `opponent_params_by_player` extension point, but the active training loop does not appear to pass frozen/historical params.
- Existing tests do not directly cover opponent registry behavior, snapshot rotation, latest versus historical swapping, staged scripted exploiters, or telemetry for opponent sampling.

## Decisions

- Target runtime: JAX mixed 2-player/4-player training.
- Schedule authority: create one explicit staged curriculum schema and migrate/replace overlapping current behavior.
- Historical opponents: default to frozen policy snapshots captured during the current training run.
- Evidence of correctness: tests plus dashboard-ready telemetry.
- Progression preference: metric-gated promotion, with openness to hybrid minimum-duration guards if needed for stability.
- First-slice deferral: public forum/shared external submissions are not required in the first implementation, but the design should leave a clear extension point.

## Constraints

- Preserve Hydra-first workflows and existing experiment/profile conventions.
- Keep mixed 2p/4p rollout behavior coherent; avoid a 4p-only solution unless explicitly documented as transitional.
- Make schedule precedence explicit. The trainer must not silently apply two competing curriculum systems.
- Frozen historical snapshots used as opponents should be deterministic when configured as deterministic.
- Opponent sampling, phase transitions, and snapshot pool state must be visible through structured metrics/logs suitable for W&B/telemetry.
- Existing experiments should either migrate cleanly or fail validation with actionable errors.

## Non-Goals

- Loading arbitrary public forum submissions in the first implementation.
- Building a new UI/dashboard.
- Optimizing opponent strength selection with Elo or league training unless added in a later spec.
- Rewriting unrelated environment, feature, or PPO logic outside the curriculum/opponent surfaces.

## Expected Opponent Families

First-class planned opponent identifiers should include:

- `latest`: current policy.
- `historical`: frozen previous snapshots from the current run.
- `random`: random baseline.
- `noop`: no-op baseline.
- `nearest_sniper`: scripted exploiter that favors focused attacks on nearby capturable targets.
- `turtle`: scripted exploiter that expands minimally and maintains high garrisons while keeping production above opponents.
- `opportunistic`: scripted exploiter that prioritizes immediately capturable enemy planets.

Additional exploiters may be added if they expose distinct failure modes and do not obscure the first implementation.

## Acceptance Criteria

1. A single authoritative curriculum config surface defines stages, promotion rules, and opponent weights for mixed JAX training.
2. Existing overlapping schedule behavior is removed, migrated, or guarded by validation so conflicting schedules cannot silently compose.
3. The active JAX trainer captures frozen policy snapshots at configured intervals and makes them available as historical deterministic opponents.
4. JAX mixed 2p/4p rollout collection can sample latest, historical, scripted, random, and noop opponents according to the active stage weights.
5. Metric-gated stage advancement is implemented, with anti-thrashing behavior if needed, and stage transitions are logged.
6. Telemetry reports active stage, configured weights, sampled opponent distribution, historical pool size, snapshot ages or ids, and stage promotion events.
7. Tests cover config validation/migration, stage selection, metric-gated advancement, opponent sampling distributions, snapshot insertion/selection, deterministic historical behavior, and at least smoke coverage through mixed rollout collection.
8. Documentation explains the new curriculum schema, how to configure staged opponent weights, what telemetry to monitor, and which features are intentionally deferred.

## Open Design Questions For Planning

- Should metric-gated advancement use win rate, episode return, shaped reward, capture metrics, or a configurable metric expression?
- Should stages have minimum update counts before promotion to reduce noisy advancement?
- Should historical snapshot selection be uniform, recent-biased, strength-biased, or stage-configurable?
- How should 2p rollout groups map multi-opponent curriculum weights when fewer opponent slots exist?
- Which existing configs should be migrated automatically versus rejected with validation errors?

## Ontology

- Curriculum stage: named progression unit with opponent weights and promotion rules.
- Promotion rule: metric-gated condition, optionally with minimum update/sample requirements.
- Opponent family: abstract type such as latest, historical, scripted, random, noop.
- Scripted exploiter: deterministic or stochastic hand-coded policy targeting a known weakness.
- Frozen snapshot: immutable copy of policy params captured during training for historical self-play.
- Historical pool: bounded set of frozen snapshots available for sampling.
- Telemetry event: structured record proving which stage/opponent/checkpoint behavior occurred.

Stability ratio: high. Core entities stabilized around stage, promotion rule, opponent family, frozen snapshot, and telemetry.

## Interview Transcript

1. Runtime scope: user selected JAX mixed 2p/4p training.
2. Schedule authority: user selected merging into one explicit curriculum schema.
3. Historical semantics: user is open to suggestions; frozen policy snapshots from training seem appropriate.
4. Success evidence: user wants tests plus dashboard-ready telemetry.
5. Contrarian challenge: user confirmed the schema merge should happen now.
6. Stage advancement: user is partial to metric-gated promotion, open to alternatives if a better case exists.
7. Simplifier pass: user selected deferring public forum submissions from the first implementation.

## Ambiguity Score

Final ambiguity: 17%

Dimension scores:

- Goal clarity: 88%
- Constraint clarity: 82%
- Success criteria: 84%
- Brownfield context clarity: 78%

Remaining ambiguity is concentrated in exact promotion metric choice, historical selection distribution, and migration details. These are planning-level design decisions rather than blockers to specifying the feature.
