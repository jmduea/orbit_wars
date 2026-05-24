# Deep Dive: Low env_steps_per_sec Optimization

## Goal

Improve Orbit Wars training throughput across configurations, with priority on:

1. Higher `env_steps_per_sec`.
2. Better policy quality per wall-clock hour.

The first implementation pass must be telemetry-only: no optimizer behavior changes, no policy semantics changes, and no config default changes until measurements identify the best optimization target.

## Trace Findings

See `.omg/specs/deep-dive-trace-low-env-steps.md` for the full trace artifact.

### Most Likely Root Cause

The low `env_steps_per_sec` is primarily real rollout/action-generation cost, not a PPO update bottleneck and not just a reporting artifact.

The stopped long `gnn_pointer` run reported approximately:

- `rollout_seconds`: 20.71
- `ppo_seconds`: 2.37
- `update_seconds`: 23.08
- `env_steps_per_sec`: 44.38
- `rollout_env_steps_per_sec`: 49.44

Rollout dominated the update budget. Because rollout-only throughput was also low, PPO tuning alone cannot fix the main issue.

### Root-Cause Drivers

- `gnn_pointer` is much heavier than the prior promoted attention baseline: hidden size 224, KNN graph construction, two message-passing layers, and an autoregressive pointer decoder with `max_moves_k=3`.
- Mixed 2p/4p training uses separate static rollout groups. Each active group is collected per update, then transitions are concatenated for PPO.
- The 4p rollout path performs substantially more work per env step: player-perspective encoding, shielding, opponent action generation, and multi-player stepping.
- The long run used settings inferred from fast MLP/attention Stage 1 results. At trace time, the active Stage 1 sweep had not produced measured `gnn_pointer` rows.
- `samples_per_sec` and `env_steps_per_sec` measure different things. `samples_per_sec` counts learner decision rows, while `env_steps_per_sec` counts environment steps. The high sample rate does not disprove low env simulation throughput.

### Important Non-Causes

- JSONL append and telemetry logging are not included in `update_seconds`, because `update_seconds` is computed before those calls.
- PPO update is not the dominant wall-clock component in the observed long run.
- Stage 1 short-run metrics are useful but include first-run JIT costs by design, so they should not be treated as pure steady-state measurements.

## First Implementation Scope

Telemetry-only instrumentation, focused on making rollout cost attributable.

### Required Metrics

Add per-format rollout timing and throughput metrics:

- `rollout_seconds_2p`
- `rollout_seconds_4p`
- `env_steps_per_sec_2p`
- `env_steps_per_sec_4p`
- `rollout_env_steps_per_sec_2p`
- `rollout_env_steps_per_sec_4p`
- `samples_per_sec_2p`
- `samples_per_sec_4p`
- `rollout_samples_per_sec_2p`
- `rollout_samples_per_sec_4p`
- `update_time_rollout_fraction`
- `update_time_ppo_fraction`

Metric names may be adjusted to match existing registry conventions, but the information must be visible in terminal logs, JSONL records, and W&B.

### Preferred Implementation Shape

- Time each active rollout group individually in `src/jax_train.py` around `group.collect_fn`.
- Attribute group metrics by `group.cfg.task.player_count` or `group.name`.
- Keep the existing aggregate metrics unchanged.
- Register new metrics in `src/metric_registry.py` under timing or throughput groups as appropriate.
- Add tests that verify mixed 2p/4p runs emit per-format timing/rate keys without changing PPO behavior.

## Follow-Up Optimization Options

These should wait until the telemetry-only pass gives evidence.

1. Create a `gnn_pointer`-specific throughput sweep with controlled axes: format, rollout steps, rollout microbatch envs, chunk rows, and maybe model-size variants.
2. Quantify 4p cost by comparing per-format rollout rates in mixed training, then add pure 2p/4p sentinels if needed.
3. Profile rollout internals if 4p or `gnn_pointer` is confirmed dominant: policy forward pass, trajectory shield, opponent generation, env step, reset, and feature encoding.
4. Test smaller `gnn_pointer` variants if policy quality per hour suffers: hidden size, `max_moves_k`, message-passing layers, and neighbor count.
5. Optimize scripted/noop/random opponent action generation in 4p if profiling shows unnecessary policy/shield work for opponent paths.

## Acceptance Criteria For First Pass

- A short mixed 2p/4p smoke run logs aggregate metrics plus per-format timing/throughput metrics.
- Existing PPO behavior is unchanged.
- Existing per-format loss diagnostics continue to work.
- Focused tests pass for metric registration and a short JAX training or PPO path.
- The resulting metrics are sufficient to decide whether the next bottleneck is 2p rollout, 4p rollout, `gnn_pointer` model cost, or update/PPO cost.
