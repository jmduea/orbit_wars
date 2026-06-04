---
title: Profile production training throughput before optimizing JAX hot paths
date: 2026-06-01
category: developer-experience
module: jax-training
problem_type: developer_experience
component: tooling
severity: high
applies_when:
  - "Training throughput regresses but sampler microbenchmarks still pass"
  - "GPU utilization snapshots look low during JAX training"
  - "A performance fix would touch rollout, PPO, or launch-hygiene semantics"
tags:
  - throughput
  - profiling
  - benchmark
  - rollout
  - ppo
  - launch-hygiene
  - jax
related_components:
  - src/jax/training_benchmark.py
  - src/cli/benchmark/training.py
  - src/jax/action_sampling.py
  - src/jax/rollout/collect.py
  - src/jax/ppo_update.py
---

# Profile production training throughput before optimizing JAX hot paths

## Context

Launch hygiene correctly prevented degenerate multi-launch sequences, but the
full training loop slowed from the pre-hygiene baseline of about 9,776
env steps/sec and 1.64 sec/update to about 950-1,100 env steps/sec and 14-17
sec/update. The isolated factorized sampler gate had already passed, so further
sampler microbench work did not explain why training an agent became
impractical.

Several early signals were misleading. A single `nvidia-smi` sample showed low
GPU utilization, but JAX reported `cuda:0` and a longer sampling run showed the
GPU reaching high utilization. Prior session history also reinforced a broader
rule: long training work needs measured training-path throughput before spending
large GPU budgets, but those older sessions did not contain the specific
rollout-vs-PPO finding (session history).

## Guidance

When end-to-end training throughput regresses, instrument the production
benchmark path before optimizing individual JAX helpers. The useful split is:

- rollout collection time
- PPO update time
- residual host overhead

For Orbit Wars, this became an opt-in benchmark mode:

```bash
uv run ow benchmark training \
  --preset primary \
  --label timing_post_hygiene \
  --updates 2 \
  --warmup 3 \
  --detailed-timing \
  --out /tmp/ow_timing_split.json
```

The JSON output includes:

```json
{
  "rollout_collect_seconds_per_update_mean": 13.68,
  "ppo_seconds_per_update_mean": 0.70,
  "host_overhead_seconds_per_update_mean": 0.005,
  "default_backend": "gpu",
  "devices": ["cuda:0"]
}
```

That split reversed the working hypothesis. PPO replay was not the dominant
cost; rollout collection was. The next design target moved to
`_sample_shielded_factored_sequence_with_params` via
`collect_rollout_jax`, not `ppo_update_jax`.

Use `--profile-dir` only when a trace is needed:

```bash
uv run ow benchmark training \
  --preset primary \
  --label trace_probe \
  --updates 3 \
  --warmup 3 \
  --detailed-timing \
  --profile-dir /tmp/ow-launch-hygiene-profile \
  --out /tmp/ow_trace_probe.json
```

JAX tracing can add significant overhead or stall while writing trace artifacts,
so start with timing buckets first. A stalled trace is less useful than a small,
completed timing split.

## Why This Matters

Microbenchmarks can be locally true and globally misleading. The sampler
microbenchmark proved the isolated factorized sampler was within its gate, but
the full train loop still missed the calibrated e2e gate by roughly 8-10x. The
production split showed that rollout collection consumed almost the whole update
wall time while PPO consumed less than a second.

The same measurement prevented several expensive dead ends:

- An all-env inactive K-step fast path measured worse.
- Replacing rollout's dense forbidden grid with compact `ForbiddenCarry`
  measured worse.
- Stop-first rollout sampling measured neutral.
- `task=shield_off` stayed slow, so trajectory-shield mode alone was not the
  explanation.

Without the timing buckets, each of those could have looked plausible from code
inspection alone.

## When to Apply

- Before optimizing rollout, PPO, action sampling, or launch-hygiene code for
  throughput.
- Before treating low `nvidia-smi` snapshots as evidence of CPU fallback.
- Before changing PPO replay semantics to fix a training-loop throughput gap.
- When a benchmark subprocess or JAX trace looks hung during first compile.

## Examples

**Bad loop:**

```text
sampler microbench slow -> optimize sampler -> sampler passes -> e2e still slow
```

**Better loop:**

```text
e2e slow -> split production update timing -> identify rollout/PPO/host bucket
        -> optimize the dominant bucket -> re-run the same e2e gate
```

For this regression, the split showed:

| Bucket | Mean seconds/update | Interpretation |
| --- | ---: | --- |
| rollout collection | 13.68 | Dominant bottleneck |
| PPO update | 0.70 | Not the first optimization target |
| host overhead | 0.005 | Not a Python orchestration issue |

The follow-up design was captured in
`docs/plans/2026-06-01-launch-hygiene-rollout-throughput-design.md`: stop
micro-optimizing forbidden carry representation and design selected-action
hygiene validation with rollout/PPO log-prob parity handled deliberately.

## Related

- Benchmark CLI package (`ow benchmark training` runner): `docs/solutions/architecture-patterns/benchmark-cli-package-split-agent-native-parity.md`
- `docs/architecture/jax-policy-encoder.md` — shared encoder + separate policy/value
  heads; PPO replay encodes once before the factorized decoder scan (relevant when
  judging whether to duplicate the trunk).
- `docs/solutions/performance-issues/launch-hygiene-incremental-carry-throughput.md`
  documents the earlier isolated sampler microbenchmark fix. This learning is
  adjacent but distinct: it covers production-path attribution after that gate
  passed.
- `docs/solutions/developer-experience/benchmark-subprocess-training-observability.md`
  covers subprocess visibility during long benchmark training. This learning
  covers attribution once the benchmark is observable.
- `docs/plans/2026-06-01-launch-hygiene-rollout-throughput-design.md`
  records the rollout-throughput evidence and failed experiments.
