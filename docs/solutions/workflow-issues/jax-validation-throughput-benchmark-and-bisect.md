---
title: Use validation 30-update benchmark for JAX throughput bisect, not smoke trains
date: 2026-06-05
category: workflow-issues
module: jax-training
problem_type: workflow_issue
component: tooling
severity: high
applies_when:
  - "HEAD env_steps_per_sec looks orders of magnitude slower than an older baseline SHA"
  - "Investigating whether stagger perf or multitask-smoke fixes explain production throughput"
  - "Comparing commits during a git bisect on rollout or env hot-path changes"
tags:
  - throughput
  - benchmark
  - validation-preset
  - bisect
  - env-steps-per-sec
  - legacy-preflight-smoke
  - stagger
related_components:
  - ow benchmark training
  - conf/training/2p4p_32_split.yaml
  - src/jax/env.py
  - src/jax/rollout/collect.py
---

# Use validation 30-update benchmark for JAX throughput bisect, not smoke trains

## Context

A production-path throughput investigation compared baseline SHA `dcafdc8`
(~3.8–4k `env_steps_per_sec` on the workstation validation profile) against
HEAD (~299 `env_steps_per_sec` on the same apples-to-apples validation
benchmark). That gap is **not** explained by rollout stagger performance work
(`638cff5`), which fixes episode-spread cost on a **legacy preflight smoke profile** (since removed from `conf/training/`) but
targets a different profile and metric window.

Parallel smokes (`multitask_smoke`, `stagger_perf_smoke`, tier-2 launch-hygiene
e2e) measure different override bundles, update counts, or rollout budgets.
Treating their JSONL `env_steps_per_sec` as interchangeable with the validation
benchmark produces false regressions or false confidence.

## Guidance

### Canonical production validation benchmark

Use the committed script and documented preset — not ad-hoc `ow train` smokes:

```bash
uv run ow benchmark training \
  --preset validation \
  --updates 30 \
  --warmup 2 \
  --label <descriptive-label> \
  --out /tmp/<label>.json
```

The `--preset validation` bundle is `WORKSTATION_VALIDATION_OVERRIDES` in
`ow benchmark training` (`src/benchmark/training.py` presets):

- `training=2p4p_32_split` (32 envs, 2p/4p split groups)
- `training.rollout_steps=128`, `epochs=2`, `update_chunk_rows=2048`
- `opponents=self_play_only`, `curriculum=off`, `seed=42`
- W&B and artifact pipeline disabled

Read `env_steps_per_sec` from the JSON payload. The script computes it as
`(measured_updates × rollout_steps × num_envs) / seconds_total` after warmup.

Also record `rollout_seconds_mean`, `update_seconds_mean`, and
`compile_seconds_to_update_3` for attribution — do not optimize from
`env_steps_per_sec` alone.

### Preset comparability across SHAs

Older baseline artifacts (e.g. `docs/benchmarks/validation-500u.json` at
`dcafdc8`) used `format=2p_4p_16env` + `training=workstation`. The current
validation preset uses `training=2p4p_32_split` instead of the `format=`
override. When bisecting:

1. Run **the same** `--preset validation` command at each SHA (let Hydra resolve
   the current bundle).
2. Do **not** compare a `format=2p_4p_16env` artifact directly to a
   `training=2p4p_32_split` artifact without noting the override drift.
3. Store `commit_sha`, full `overrides` array, and raw JSON for every trial.

### Bisect findings (2026-06-05 arc)

Git bisect on the validation benchmark localized a large slowdown between:

| SHA | Approx. `env_steps_per_sec` | Notes |
| --- | ---: | --- |
| `71c3e91` | ~8.5–9.2k | Last known-fast in bisect window |
| `b11b9b0` | ~380 | First known-slow; comet env stepping still on train path |
| `dcafdc8` | ~3.8–4k | Documented validation baseline (older override bundle) |
| HEAD (post-stagger) | ~299 | Stagger perf merged; regression persists |

Leading suspects in the slow window: comet env integration (`33b56e2` area) and
train-loop / rollout-group refactors — not stagger encode deferral. Confirm with
`rollout_seconds_mean` splits before editing stagger or sampler code.

### What not to conflate

| Profile | Purpose | Throughput authority? |
| --- | --- | --- |
| `--preset validation` + 30 updates | Production-path apples-to-apples bisect | **Yes** for this investigation |
| Legacy preflight smoke (10 updates, profile removed) | Episode spread + rollout budget experiments | No — stagger target only |
| `scripts/ce_optimize/multitask_smoke_measure.py` | Encode/decode attribution | No — different overrides |
| `make test-launch-hygiene-e2e-throughput` | Tier-2 vs pre-hygiene baseline JSON | Related but different gate and baseline SHA |

Stagger perf (`638cff5`) is **verified** against that legacy preflight rollout
seconds and `episode_done` cadence. A passing stagger smoke does **not** clear
a failing validation benchmark at HEAD.

## Why This Matters

Microbench and profile-specific smokes can pass while the canonical validation
path regresses by 10–30×. Without a single committed benchmark recipe, bisect
results are not reproducible and fixes chase the wrong subsystem (stagger vs env
step vs collect).

The validation script runs **outside pytest** with isolated worker env setup
(see Kaggle notebook worker env wiring in `src/orchestration/kaggle_runner.py`), avoiding the pytest JAX timing skew
documented in
[launch-hygiene-incremental-carry-throughput.md](../performance-issues/launch-hygiene-incremental-carry-throughput.md).

## When to Apply

- Before claiming a throughput regression or fix on `main` / a feature branch.
- Before merging env hot-path changes (`src/jax/env.py`, rollout collect,
  train-loop group wiring).
- When stagger, multitask-smoke, or tier-2 e2e numbers disagree — re-run
  validation preset at both SHAs first.
- When planning cherry-pick / nuclear rollback workflows — see
  `docs/solutions/workflow-issues/nuclear-cherry-pick-manifest-baseline-integration.md`.

## Examples

**Good bisect trial loop:**

```bash
git checkout <sha>
uv run ow benchmark training \
  --preset validation --updates 30 --warmup 2 \
  --label bisect-<short-sha> \
  --out docs/benchmarks/bisect-<short-sha>.json
# Record env_steps_per_sec, rollout_seconds_mean, overrides[], commit_sha
```

**Bad comparison:**

```text
stagger_perf_smoke rollout_seconds improved → declare production throughput fixed
```

**Override drift trap:**

```text
dcafdc8 validation-500u.json (format=2p_4p_16env) vs HEAD --preset validation
(training=2p4p_32_split) → treat as same preset without checking overrides[]
```

## Related

- Production-path timing split (rollout vs PPO):
  [production-training-throughput-profiling.md](../developer-experience/production-training-throughput-profiling.md)
- Tier-2 launch-hygiene gate and learner ablation tiebreaker:
  [launch-hygiene-learner-ablation-gate.md](../tooling-decisions/launch-hygiene-learner-ablation-gate.md)
- Comet / env-parity throughput plan: `docs/solutions/workflow-issues/jax-validation-throughput-benchmark-and-bisect.md`
- Stagger perf plan (orthogonal fix): `docs/solutions/workflow-issues/jax-validation-throughput-benchmark-and-bisect.md`
- Validation benchmark index: `docs/benchmarks/issues-jax-validation-500u.md`
- Open issues: [#204](https://github.com/jmduea/orbit_wars/issues/204) (SSOT wall clock), [#188](https://github.com/jmduea/orbit_wars/issues/188) area (comet env)
