---
title: "feat: Env parity A/B throughput (pre vs post Kaggle parity)"
type: feat
status: active
date: 2026-06-04
origin: docs/plans/2026-06-03-008-feat-jax-comet-subsystem-plan.md
related_issue: 189
---

# Plan: JAX env parity A/B throughput

## Summary

Measure whether post–#188 Kaggle env parity work (comet subsystem + reference planet generation) explains training throughput regression. Ship an agent-native **`ow benchmark env-parity-ab`** that compares three `task.env_parity_mode` arms on the same JIT path, plus optional short `ow benchmark training` arms.

## Problem Frame

Post-hygiene e2e throughput dropped vs pre-hygiene baseline (`docs/benchmarks/launch-hygiene-e2e-baseline.json`). Rollout/sampler optimizations are largely exhausted. #188 added comet state, spawn/movement hooks, and (briefly) `pure_callback` on the default train path; `b11b9b0` restored JAX-native `_reset_train` for `env_parity_mode=train` but **comet stepping** (`_advance_comet_positions`, `_expire_comets_pre_launch`, `is_comet` masks) still runs every env step.

Planet layout for `train` matches pre-#188 `_reset_train` (variable groups/orbits). The open question is **comet machinery cost** vs **Kaggle reference generators** (`env_parity_mode=kaggle`).

## Requirements

| ID | Requirement |
|----|-------------|
| R1 | Add `task.env_parity_mode: legacy` — pre-#188 comet-free hot path (no comet pre-launch, advance, or collision branches) |
| R2 | Keep `train` (default) and `kaggle` (reference `generate_planets` + comet spawn via `pure_callback`) |
| R3 | `ow benchmark env-parity-ab` — microbench JIT `vmap(reset)+vmap(step)` per arm; JSON stdout/`--out` |
| R4 | Report `env_steps_per_sec`, `reset_seconds`, `step_seconds`, arm labels, commit SHA, JAX device |
| R5 | Optional `--training` flag runs short `ow benchmark training` per arm (same overrides except parity mode) |
| R6 | Hydra groups: `task=env_legacy`, `task=kaggle_parity` (existing), default `train` |
| R7 | Fast tests: legacy mode reset+step smoke; benchmark payload shape test (CPU, no GPU required) |
| R8 | Document interpretation in benchmark JSON `notes` and `docs/benchmarks/README.md` index line |

## Key Technical Decisions

**KTD1 — `legacy` = comet-free, not git checkout.** Pre-#188 planet reset is already `_reset_train`; legacy only bypasses comet subgraph in `step` / `_move_and_resolve`. Rationale: isolates comet cost without worktree drift.

**KTD2 — `kaggle` arm is diagnostic only.** `pure_callback` in jitted collect is not production; include in A/B to quantify reference-generator + spawn cost, not to gate merges.

**KTD3 — Median of N≥2 repeats.** Match ce-optimize / launch-hygiene protocol; default `--repeats 3`.

**KTD4 — Microbench first, training optional.** Microbench isolates env; `--training` uses `multitask_smoke` or `--preset validation` with `training.total_updates=20`.

## Implementation Units

### U1. `legacy` env parity mode

**Files:** `src/game/shield_config.py` (or `src/jax/env_parity.py`), `src/jax/env.py`, `src/config/schema.py`, `conf/task/env_legacy.yaml`, `conf/task/base.yaml` comment

**Verification:** `pytest tests/test_jax_env_legacy_mode.py`

### U2. Env microbench core

**Files:** `src/jax/env_parity_benchmark.py`

**Verification:** unit test on payload aggregation

### U3. CLI

**Files:** `src/cli/benchmark/env_parity_ab.py`, `src/cli/benchmark/parser.py`, `src/cli/benchmark/__init__.py`, `src/cli/__init__.py` dispatch

**Verification:** `uv run ow benchmark env-parity-ab --help`

### U4. Docs

**Files:** `docs/benchmarks/README.md` (one paragraph + example command)

## Test Scenarios

- Legacy reset+step 10 noop steps: finite reward, no comet groups active.
- `env_parity_mode=kaggle` still composes under Hydra.
- Benchmark returns three arms with `env_steps_per_sec > 0`.

## Risks

- Legacy skips may diverge from true pre-#188 if other #188 edits touched non-comet paths — document scope.
- GPU contention: run one benchmark session at a time.

## Deferred

- Pure JAX `src/jax/planet_generation.py` / `comet_generation.py` (trace-hygiene plan) as fourth arm
- Automatic regression gate on A/B delta
