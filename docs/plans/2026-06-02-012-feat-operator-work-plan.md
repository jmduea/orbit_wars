---
title: "feat: Operator work — launch hygiene runbook, ROADMAP Later, preflight polish"
type: feat
status: active
date: 2026-06-02
origin: LFG operator work (ROADMAP empty; tier-2 gate wired but undocumented for operators)
---

# Plan: Operator work (verification / infra)

## Summary

Close operator-facing gaps deferred while ROADMAP Now/Next/Later are empty: consolidate launch-hygiene tier-1/tier-2 and preflight gate runbooks, refresh ROADMAP **Later** with acceptance criteria, polish Makefile/`ow benchmark` discovery, and add a fast-tier regression test for the committed e2e baseline artifact. No invented throughput thresholds; GPU e2e gate remains operator-run (`make test-launch-hygiene-e2e-throughput`).

## Problem

Launch hygiene tier-2 (`make test-launch-hygiene-e2e-throughput`, baseline JSON, CLI assert) landed on `main`, but operators lack a single runbook, ROADMAP **Later** is empty, `make help` omits hygiene targets, and pytest does not validate the committed baseline artifact schema.

## Requirements

- R1. ROADMAP **Later** lists deferred operator items with clear acceptance (tier-2 e2e gate pass, preflight learn-proof, Phase B recovery).
- R2. Operator runbook doc with tier-1 vs tier-2, baseline capture, gate command, preflight primitive sequence.
- R3. `make help` documents launch-hygiene and preflight Makefile targets.
- R4. Fast-tier test: `docs/benchmarks/launch-hygiene-e2e-baseline.json` passes `validate_e2e_baseline_artifact`.
- R5. Cross-links from `docs/benchmarks/preflight-calibration.md` and launch-hygiene solution doc.

## Out of scope

- Phase B hot-path recovery (U7 in e2e throughput plan) — ROADMAP Later only.
- GPU e2e gate execution in CI (no GPU workflow).
- Preflight profile registry (separate plan `2026-06-02-005`).

## Implementation units

### U1. ROADMAP Later operator backlog

**Files:** `docs/ROADMAP.md`

**Acceptance:** Three Later rows: launch-hygiene tier-2 gate pass on baseline machine; preflight learn-proof after calibration refresh; launch-hygiene Phase B if gap remains out-of-band.

### U2. Operator runbook

**Files:** `docs/operator-runbook.md` (new)

**Content:** Tier-1/tier-2 table; baseline SHA and capture recipe from AGENTS.md; gate command; preflight sanity → gate dry-run → learn-proof; terminal/GPU hygiene note.

### U3. Makefile help polish

**Files:** `Makefile`

**Acceptance:** `make help` lists `test-launch-hygiene-throughput`, `test-launch-hygiene-e2e-throughput`, and expanded preflight comment on `preflight-learn-proof`.

### U4. Committed baseline artifact test

**Files:** `tests/test_training_benchmark_gate.py`

**Acceptance:** Test loads `docs/benchmarks/launch-hygiene-e2e-baseline.json`, `validate_e2e_baseline_artifact` returns `[]`.

### U5. Cross-links

**Files:** `docs/benchmarks/preflight-calibration.md`, `docs/solutions/performance-issues/launch-hygiene-incremental-carry-throughput.md`

## Verification

- `make test-fast` (includes U4)
- Manual: `make help` shows hygiene targets
- GPU (operator): `make test-launch-hygiene-e2e-throughput` on baseline machine — not required for merge

## Sources

- `docs/plans/2026-06-01-launch-hygiene-e2e-throughput-plan.md`
- `AGENTS.md` launch hygiene section
- `docs/benchmarks/launch-hygiene-e2e-baseline.json`
