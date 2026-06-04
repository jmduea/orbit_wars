---
title: "feat: SSOT pipeline foundation slice (LFG #211)"
type: feat
status: active
date: 2026-06-04
origin: docs/plans/2026-06-03-013-feat-ssot-training-pipeline-plan.md
parent_issue: 211
---

# feat: SSOT pipeline foundation slice (LFG #211)

## Summary

First implementation PR for [#211](https://github.com/jmduea/orbit_wars/issues/211): foundational units **U2**, **U1 skeleton**, **U3 partial**, **U4 config stub**, and targeted tests. Parent plan [`2026-06-03-013-feat-ssot-training-pipeline-plan.md`](2026-06-03-013-feat-ssot-training-pipeline-plan.md) remains authoritative for U5–U8.

**Runtime spine (unchanged):** config → `make test-fast` → W&B sweep → packaging (winner) → long train → qualifiers → bracket → submission.

## Scope (this PR)

| Unit | Deliverable |
|------|-------------|
| U2 | `training_seed_set` / `eval_seed_set`; disjoint validation; `SeedScheduler` uses training pool only |
| U1 | `conf/wandb_sweep/ssot_preflight.yaml` + fixed + metric (Gates 2–3 floors from calibration JSON) |
| U3 | `ow eval package` flags: `--packaging-seed`, `--packaging-player-count`; compose test |
| U4 | `conf/artifacts/ssot_pipeline.yaml` stub (W&B on, legacy funnels off) |

**Deferred:** U5 JAX qualifiers, U6 calibration campaign, U7 bracket/submission, U8 teardown, W&B `--wandb-run` resolver, long-train curriculum hooks, automated winner promotion.

## Key Technical Decisions

**KTD-L1 — Backward compat for `heldout_eval_seed_set`.** When `training_seed_set` is empty and `heldout_eval_seed_set` is non-empty, runtime raises with migration message (R29). New configs must set explicit partitions.

**KTD-L2 — Default eval seeds.** `eval_seed_set` defaults to `[43, 44, 45, 46]` (historical validation seeds in `docs/benchmarks/validation-seed-*.json`). `training_seed_set` defaults empty → scheduler uses `random_jump` / incremental from `training.seed`.

**KTD-L3 — SSOT sweep metric.** W&B objective minimizes negative composite: eligible runs score `win_rate_delta_10` when KL/entropy pass calibrated floors; ineligible → sentinel (planet-flow pattern).

## Implementation Units

### U2. Train / eval seed partition

**Files:** `src/config/schema.py`, `conf/config.yaml`, `src/config/runtime.py`, `src/training/seed_scheduler.py`, `src/jax/train/loop.py`, `tests/test_eval_seed_contamination.py`, `tests/test_seed_scheduler.py`

**Verification:** `make test-fast`; AE6 contamination test fails on overlap.

### U1. W&B sweep preflight skeleton

**Files:** `conf/wandb_sweep/ssot_preflight.yaml`, `conf/wandb_sweep/fixed/ssot_preflight.yaml`, `conf/wandb_sweep/metric/ssot_preflight_learning_signal.yaml`, `tests/test_ssot_wandb_sweep_compose.py`

**Verification:** `uv run ow make wandb_sweep=ssot_preflight`; compose smoke in CI.

### U3. Packaging validation CLI partial

**Files:** `src/cli/eval.py`, `src/artifacts/kaggle_submission.py`, `tests/test_ssot_packaging_cli.py`

**Verification:** CLI help documents SSOT defaults; unit test asserts seed 0 / 4p path when flags set.

### U4. `ssot_pipeline` Hydra stub

**Files:** `conf/artifacts/ssot_pipeline.yaml`, `tests/test_ssot_pipeline_config.py`

**Verification:** `uv run ow train print_resolved_config=true artifacts=ssot_pipeline` composes; wandb enabled; hybrid/bracket funnels disabled.

## Requirements traceability

| ID | This PR |
|----|---------|
| R14, R25, AE6 | U2 |
| R5–R6, R9–R11 (skeleton) | U1 |
| R7–R8 (CLI shape) | U3 partial |
| R12–R13 (profile stub) | U4 |

## Sources

- Parent: `docs/plans/2026-06-03-013-feat-ssot-training-pipeline-plan.md`
- Requirements: `docs/brainstorms/2026-06-03-training-pipeline-ssot-requirements.md`
- Learning: `docs/solutions/architecture-patterns/ssot-training-pipeline-config-to-kaggle-submission.md`
