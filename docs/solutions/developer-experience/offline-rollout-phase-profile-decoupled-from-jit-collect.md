---
title: Offline rollout phase profiling — keep host-timed collect out of production train
date: 2026-06-06
category: developer-experience
module: jax-rollout
problem_type: developer_experience
component: tooling
severity: high
applies_when:
  - "You need itemized rollout_s breakdown (policy vs opponent vs env vs reset)"
  - "Admission or map-pool work needs phase shares before optimizing a hot path"
  - "telemetry=rollout_phase_timing on ow train stalls 20+ minutes on 32×256"
symptoms:
  - "Gate or admission train run hangs 20–30+ min on first update with rollout phase timing enabled"
  - "rollout_seconds is known but policy/opponent/env/reset shares are not"
root_cause: wrong_api
resolution_type: tooling_addition
tags:
  - rollout-phase-timing
  - profiling
  - throughput
  - collect
  - benchmark
  - map-pool
  - admission
related_components:
  - src/jax/rollout/collect_timed.py
  - src/jax/rollout_phase_profile.py
  - src/jax/train/rollout_groups.py
  - src/jax/rollout/phase_timing_report.py
  - src/cli/benchmark/rollout_phase_profile.py
  - conf/telemetry/rollout_phase_timing.yaml
---

# Offline rollout phase profiling — keep host-timed collect out of production train

## Context

Operators needed an **itemized breakdown of `rollout_s`** inside rollout collect: learner policy+shield, opponent sampling+encoding, env step, episode reset (map-pool gather), and post-step bookkeeping.

The first attempt wired `telemetry=rollout_phase_timing` into the **production `ow train` path** by routing collect through `collect_rollout_jax_timed` and emitting phase metrics into `*_jax.jsonl`. On admission geometry (32 envs × 256 steps), the run **stalled 20+ minutes** on the first update and never reached a useful breakdown (session history).

Root cause: host-timed collect calls `jax.block_until_ready` **once per rollout step per phase**. That is correct for diagnostic wall-clock shares but incompatible with the JIT-compiled, gate-spine train loop at full geometry.

**Verified rework:** phase timing is **offline-only**. Production train always uses `init_rollout_groups()` → `timed_collect=False` → `collect_rollout_jax`. Profiling uses `init_profile_rollout_groups()` → `timed_collect=True` → `collect_rollout_jax_timed`, with **no outer `jax.jit` on `collect_fn`** so host timers can fire.

`conf/telemetry/rollout_phase_timing.yaml` is **deprecated for `ow train`**; both `orbit_wars` and `orbit_wars-integration` point operators at the benchmark CLI instead.

## Guidance

### Architecture

| Layer | Production train | Offline profile |
|-------|------------------|-----------------|
| Init | `init_rollout_groups()` | `init_profile_rollout_groups()` |
| Collect impl | `collect_rollout_jax` | `collect_rollout_jax_timed` |
| `collect_fn` | `jax.jit` wrapped | **not** JIT-wrapped |
| Metrics sink | Normal rollout metrics → JSONL (no phase keys by default) | In-process only; optional `--out` JSON |
| Entry | `src/jax/train/loop.py` | `src/jax/rollout_phase_profile.py` |

### Phase definitions (`collect_timed.py`)

Host-instrumented buckets (each step syncs before accumulating):

- **policy** — learner `_sample_shielded_sequence_with_params` + shield + `build_action_from_factored_batch`
- **opponent** — opponent action sampling **and** `encode_turn` for opponent feature cache refresh
- **env_step** — `batched_step` / `batched_step_multi_player`
- **reset** — episode-done branch only: `batched_reset_with_pool` (map pool) or `batched_reset`, plus `assign_learner_players`
- **post_step** — decoder carry bookkeeping and other post-step work outside the above

Emitted keys: `rollout_phase_{policy,opponent,env_step,reset,post_step}_{seconds,fraction}` plus `rollout_phase_measured_total_seconds`.

### Geometry modes

**Quick (default)** — admission preset with `training=smoke` → **2 envs × 8 steps**. Phase **fractions** are interactive; absolute seconds are **not** throughput-comparable to admission.

**Full geometry (`--full-geometry`)** — operator-locked admission overrides: `training=2p4p_32_split`, `training.rollout_steps=256`, etc. First update may take **30+ minutes**; stderr warns before start.

Progress streams on **stderr** per update. Do not pipe profile output to `tail` (see `docs/solutions/developer-experience/ow-long-cli-stderr-progress-no-tail-pipe.md`).

### CLI surface

**Integration worktree (implementation home):**

```bash
cd /home/jmduea/projects/orbit_wars-integration

uv run ow benchmark rollout-phase-profile \
  --preset admission \
  --train-overrides task=map_pool \
  --updates 3 \
  --warmup 2 \
  --out /tmp/rollout-phase-profile.json

uv run ow benchmark rollout-phase-breakdown /tmp/rollout-phase-profile.json
```

**Main repo (delegates to integration):**

```bash
cd /home/jmduea/projects/orbit_wars

uv run ow benchmark rollout-phase-profile \
  --repo-root /home/jmduea/projects/orbit_wars-integration \
  --train-overrides task=map_pool \
  --updates 3
```

### Do not

- Enable `telemetry=rollout_phase_timing` on `ow train` / gate spine for phase shares.
- Treat quick-mode fractions as admission throughput proof — use `ow benchmark training` / tier-2 e2e for that.
- Expect `rollout_phase_measured_total_seconds` ≈ `rollout_seconds`; host sync overhead and unmeasured gaps are normal.

## Why This Matters

Post-hygiene throughput work already showed **rollout collect dominates** update time. Without phase shares inside collect, optimization guesses wrong targets (policy decoder vs opponent `encode_turn` vs env step vs map-pool reset).

Wiring host timers into production train **poisoned the very path being measured**: 32×256 × per-step sync × compile = multi-hour first updates, zero actionable data. Offline profiling answers “where inside collect?” in minutes on quick geometry, then optionally confirms on full geometry when worth the wait.

**Verified quick `task=map_pool` profile (3 measured updates after warmup):**

| Phase | Share |
|-------|-------|
| opponent | ~68% |
| policy | ~17% |
| env_step | ~10% |
| reset | ~0% |

At smoke geometry with map pool, **opponent sampling+encoding** is the largest collect slice; reset/gather is negligible at this episode length — not a license to skip map-pool reset profiling at full geometry or longer episodes.

## When to Apply

- Before optimizing rollout collect, launch hygiene, or map-pool reset — run quick profile with the **same `task=` and opponent recipe** you care about.
- When admission debugging needs phase fractions but gate train must stay on the fast JIT path.
- When comparing `task=map_pool` vs other task overrides via `--train-overrides`.

Skip when:

- You only need coarse rollout vs PPO split → `ow benchmark training --detailed-timing` (see `docs/solutions/developer-experience/production-training-throughput-profiling.md`).
- You need submit-valid or learning proof → gate primitives, not phase profile.

## Examples

### Quick map_pool phase shares

```bash
cd /home/jmduea/projects/orbit_wars-integration
uv run ow benchmark rollout-phase-profile \
  --preset admission \
  --train-overrides task=map_pool \
  --updates 3 \
  --warmup 2
```

### Full admission geometry (expect long first compile)

```bash
uv run ow benchmark rollout-phase-profile \
  --preset admission \
  --full-geometry \
  --train-overrides task=map_pool \
  --updates 3 \
  --warmup 2 \
  --out /tmp/full-geometry-profile.json
```

### Key implementation files (`orbit_wars-integration`)

- `src/jax/rollout/collect_timed.py` — `collect_rollout_jax_timed`, per-step host timers
- `src/jax/rollout_phase_profile.py` — `run_rollout_phase_profile`, admission/quick override bundles
- `src/jax/train/rollout_groups.py` — `init_rollout_groups` vs `init_profile_rollout_groups`
- `src/jax/rollout/phase_timing_report.py` — breakdown extraction and formatting
- `tests/test_rollout_phase_profile.py` — preset must not include `telemetry=rollout_phase_timing`

## Related

- [`production-training-throughput-profiling.md`](production-training-throughput-profiling.md) — coarse rollout vs PPO split (`--detailed-timing`); use this doc for the next level down
- [`jax-validation-throughput-benchmark-and-bisect.md`](../workflow-issues/jax-validation-throughput-benchmark-and-bisect.md) — validation bisect uses coarse `rollout_seconds_mean`; phase profile is diagnostic-only
- [`benchmark-cli-package-split-agent-native-parity.md`](../architecture-patterns/benchmark-cli-package-split-agent-native-parity.md) — `ow benchmark` package layout (refresh for `rollout-phase-*` subcommands)
- GitHub [#189](https://github.com/jmduea/orbit_wars/issues/189), [#200](https://github.com/jmduea/orbit_wars/issues/200) — rollout bottleneck epics
