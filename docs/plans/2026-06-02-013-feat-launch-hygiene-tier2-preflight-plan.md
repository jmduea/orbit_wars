---
title: "feat: Launch hygiene tier-2 gate, Phase B recovery, preflight refresh"
type: feat
status: completed
date: 2026-06-02
origin: docs/ROADMAP.md (Later section)
related_plan: docs/plans/2026-06-01-launch-hygiene-e2e-throughput-plan.md
---

# Plan: Launch hygiene tier-2 gate closure and preflight refresh

## Summary

Execute ROADMAP **Later** items on baseline GPU. Tier-2 e2e gate was run and **failed** (~4× slower than pre-hygiene band). Hot-path recovery options are **exhausted** per `docs/plans/2026-06-01-launch-hygiene-rollout-throughput-design.md`. **Pivot:** learner ablation (A pre-hygiene SHA `79162a…` vs B launch-hygiene `main`); winner = better preflight learn-proof / gate trends, not throughput. Document in benchmark JSON, ROADMAP, and PR.

## Problem Frame

Operator runbook PR #181 landed tier-1/tier-2 docs and baseline artifact, but tier-2 gate has not passed on current `main`. Baseline `gap_assessment` documents ~90% regression vs pre-hygiene capture. Profiling (`docs/plans/2026-06-01-launch-hygiene-rollout-throughput-design.md`) implicates rollout collection (~13.7s/update) not PPO (~0.7s).

## Requirements

- R1. Run tier-2 gate on RTX 5080 with JAX cache env per runbook; record pass/fail metrics honestly.
- R2. If fail: Phase B per U7 — behavior-preserving rollout hot-path fix; re-gate until pass or document blocker.
- R3. If tier-2 pass: run `make preflight-learn-proof` through `beat_random` vs `preflight-calibration.json`.
- R4. Update `docs/ROADMAP.md` Later/Done; update baseline `gap_assessment` with current SHA/metrics.
- R5. `make test-fast` after code changes.

## Key Technical Decisions

**KTD1 — Gate authority:** `--assert-within-pct 10` vs `docs/benchmarks/launch-hygiene-e2e-baseline.json`; no invented thresholds.

**KTD2 — Phase B target:** Rollout sampling in `src/jax/action_sampling.py` per rollout-throughput design (selected-action validation or equivalent), preserving launch-hygiene semantics and tier-1 microbench.

**KTD3 — Single cohesive PR:** fixes + gap_assessment + ROADMAP triage + preflight evidence in one branch when possible.

## Implementation Units

### U1. Tier-2 e2e gate measurement

**Goal:** Establish current pass/fail on baseline GPU at `main` baseline.

**Files:** `docs/benchmarks/launch-hygiene-e2e-baseline.json` (gap_assessment only)

**Verification:** `make test-launch-hygiene-e2e-throughput` exit code and `/tmp/launch_hygiene_e2e_gate.json` metrics.

### U2. Learner ablation (A vs B)

**Goal:** Compare pre-hygiene baseline SHA vs launch-hygiene `main` on learn-proof through `beat_random`.

**Dependencies:** U1 fail + hot-path options exhausted

**Files:** `docs/benchmarks/launch-hygiene-ablation.json`

**Verification:** Both arms run `ow benchmark learn-proof --model transformer_factorized_small --through beat_random`; document verdicts and key metrics vs `preflight-calibration.json` thresholds.

### U3. Preflight learn-proof refresh (arm B)

**Goal:** Fresh learn-proof on current `main` if not already captured at HEAD.

**Dependencies:** U2

**Verification:** `make preflight-learn-proof` or equivalent CLI; VERIFIED through `beat_random` preferred.

### U4. ROADMAP and operator closure

**Goal:** Move completed Later items to Done; update gap_assessment artifact.

**Files:** `docs/ROADMAP.md`, `docs/benchmarks/launch-hygiene-e2e-baseline.json`, `docs/operator-runbook.md` if status notes needed

**Verification:** ROADMAP reflects measured outcomes.
