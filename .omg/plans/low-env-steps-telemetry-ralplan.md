# RALPLAN: Low env_steps_per_sec Telemetry

## Decision

Implement lightweight, telemetry-only per-format rollout timing in the existing JAX training loop.

This is a staged hybrid approach: in-code metrics now, external profiler work later only if the new metrics show that coarse timing is insufficient.

## Decision Drivers

1. The observed long `gnn_pointer` run is rollout-dominated, so attribution must start around rollout collection.
2. The user wants better `env_steps_per_sec` and policy quality per wall-clock hour, but the first pass must be telemetry-only.
3. The implementation must not alter optimizer, policy, environment, opponent, or PPO semantics.

## Alternatives Considered

### Lightweight In-Code Timing

Pros: low risk, testable, W&B-ready, directly answers whether 2p or 4p rollout dominates.

Cons: does not identify kernel-level bottlenecks inside compiled JAX programs.

### External Profiler First

Pros: deeper device/JIT visibility.

Cons: harder to test, more manual, easier to lose sight of the immediate goal.

### Full Hybrid In One Pass

Pros: maximum observability.

Cons: too much scope for the first telemetry change.

## Chosen Approach

Use lightweight in-code timing now and preserve profiler integration as an explicit follow-up.

## Scope Boundaries

- No changes to `src/jax_ppo.py` in this stage unless a blocking dependency is discovered.
- No profiling hooks, stack traces, JAX profiler sessions, or kernel-level tracing in this stage.
- No optimizer, policy, model, environment, curriculum, opponent, or config default behavior changes.
- Existing aggregate metrics must remain unchanged.

## Metric Names

Use static flat metric names compatible with the current registry and W&B history:

| Metric | Group | Description |
| --- | --- | --- |
| `rollout_seconds_2p` | `timing` | Wall-clock seconds spent collecting 2-player rollout groups. |
| `rollout_seconds_4p` | `timing` | Wall-clock seconds spent collecting 4-player rollout groups. |
| `env_steps_per_sec_2p` | `timing` | 2-player env steps per full update second. |
| `env_steps_per_sec_4p` | `timing` | 4-player env steps per full update second. |
| `rollout_env_steps_per_sec_2p` | `timing` | 2-player env steps per 2-player rollout second. |
| `rollout_env_steps_per_sec_4p` | `timing` | 4-player env steps per 4-player rollout second. |
| `samples_per_sec_2p` | `timing` | 2-player learner decision samples per full update second. |
| `samples_per_sec_4p` | `timing` | 4-player learner decision samples per full update second. |
| `rollout_samples_per_sec_2p` | `timing` | 2-player learner decision samples per 2-player rollout second. |
| `rollout_samples_per_sec_4p` | `timing` | 4-player learner decision samples per 4-player rollout second. |
| `update_time_rollout_fraction` | `timing` | Fraction of update wall time spent collecting rollouts. |
| `update_time_ppo_fraction` | `timing` | Fraction of update wall time spent in PPO optimization. |

If a format is inactive in a run, emit `0.0` for that format's timing/rate metrics so dashboards and tests have stable keys.

## Implementation Plan

1. In `src/jax_train.py`, time each active `group.collect_fn(...)` call with `time.perf_counter()`.
2. Attribute each group by `group.cfg.task.player_count`, accumulating per-format rollout seconds, env steps, and learner samples from the group's rollout metrics.
3. Keep current aggregate `rollout_seconds`, `ppo_seconds`, `update_seconds`, `env_steps_per_sec`, `rollout_env_steps_per_sec`, `samples_per_sec`, and `ppo_samples_per_sec` unchanged.
4. Add per-format metrics to the update record after aggregate timing is computed.
5. Add update-time fractions from existing aggregate timings:
   - `update_time_rollout_fraction = rollout_seconds / update_seconds`
   - `update_time_ppo_fraction = ppo_seconds / update_seconds`
6. Add the twelve metrics listed above to `src/metric_registry.py` with `_metric(..., "timing", ...)` entries.
7. Add `tests/test_metric_registry.py::test_per_format_timing_metrics_are_registered_as_timing` covering all twelve metric names, their group, and default enablement.
8. Add a deterministic helper test for per-format metric construction. Prefer a small helper in `src/jax_train.py` that accepts already-measured seconds, env steps, samples, and update seconds so the test does not rely on wall-clock timing.
9. Add or extend one short training-loop smoke test to assert emitted keys exist, are numeric, and are non-negative. Do not assert exact elapsed values in this smoke test.
10. Verify metric names before implementation: names must be unique in `METRIC_DEFINITIONS`, flat strings without path separators, comfortably below W&B metric-name length limits, and absent from known protected/objective names unless deliberately protected.

## Test Strategy

- Registry test: all new metrics are registered in `timing` and enabled by default.
- Deterministic unit test: factor the per-format timing/rate record construction into a small helper or otherwise monkeypatch the clock so fixed inputs produce fixed metric values without depending on wall-clock timing.
- Training/log smoke test: a one-update tiny mixed-format JAX training run writes a JSONL update record containing the new keys.
- Wall-clock assertions should check presence, numeric type, and plausible non-negative ranges only. Exact timing assertions are allowed only in the fixed-clock helper test.
- Existing PPO tests do not need changes for this stage because PPO behavior is intentionally untouched.
- W&B compatibility is covered by registry-name checks plus the existing telemetry path: because `telemetry.log(record, step=update)` receives the same flat update record, registered flat keys should flow to W&B when W&B is enabled. A live W&B dashboard check is optional manual verification, not a required automated test.

## Acceptance Criteria

- New per-format throughput metrics appear in JSONL and W&B update records.
- Aggregate metrics remain present and semantically unchanged.
- Focused tests pass.
- No runtime behavior changes outside telemetry record construction.
- The next optimization decision can be made from the observed 2p/4p rollout timing split.

## Consequences

- The first implementation will not directly speed training up.
- The next pass can choose between `gnn_pointer` config sweeps, 4p rollout optimization, model-size variants, or profiler work using evidence rather than inference.
