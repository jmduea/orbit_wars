---
title: "fix: Bootstrap unified incumbent to scripted nearest_sniper"
type: fix
status: completed
date: 2026-06-03
origin: docs/brainstorms/2026-06-03-gate5-unified-tournament-requirements.md
---

# fix: Bootstrap unified incumbent to scripted nearest_sniper

## Summary

Stage 2 unified tournament proof failed with `incumbent_not_defeated` because calibration bootstrapped `incumbent_checkpoint_path` to the same checkpoint as the challenger (self-incumbent). Product intent: the **scripted `nearest_sniper` opponent** is the initial incumbent until a checkpoint beats it at 100% per-seed combined over 30 seeds; promoted campaign manifests override bootstrap after R9 swap.

## Problem Frame

| Symptom | Root cause |
|---------|------------|
| `incumbent_not_defeated` on post-hygiene ckpt | `incumbent_checkpoint_path` in `preflight-calibration.json` pointed at the evaluated checkpoint |
| Self-incumbent 2p head-to-head | `resolve_incumbent` loaded checkpoint agent with `agent_id="incumbent"` from calibration path |

## Scope

| In scope | Out of scope |
|----------|--------------|
| Bootstrap incumbent = scripted `nearest_sniper` | Recalibrating Stage-1 noop/random floors |
| Remove `incumbent_checkpoint_path` from calibration + spec | Changing R9 per-seed 100% threshold |
| Promoted manifest still overrides bootstrap | Re-running full tournament-proof (optional operator step) |
| Tests: incumbent ≠ challenger on bootstrap path | Weakening enforcement assertions |

## Requirements

- R1. Default/bootstrap incumbent resolves to scripted `nearest_sniper` (runtime `sniper`) when no campaign promoted manifest exists.
- R2. Promoted incumbent checkpoint from campaign manifest takes precedence over bootstrap.
- R3. Remove `incumbent_checkpoint_path` from `UnifiedTournamentSpec`, Hydra config, and calibration JSON; replace with `incumbent_bootstrap_opponent: "nearest_sniper"`.
- R4. Stage 2 scheduling uses bootstrap scripted agent with `agent_id="incumbent"` for scoring compatibility.
- R5. Evaluating a checkpoint without promoted manifest must not load that checkpoint as incumbent.

## Key Technical Decisions

**KTD1 — Scripted bootstrap via `agent_from_baseline`.** Add `agent_from_baseline(name)` in `src/artifacts/tournament/resolve.py` using existing `build_baseline_agent` + `normalize_baseline_name`. Sentinel `checkpoint_path` of `scripted:<name>` for AgentEntry compatibility.

**KTD2 — Spec field rename.** `incumbent_checkpoint_path` → `incumbent_bootstrap_opponent: str = "nearest_sniper"`. Enforcement no longer blocks on missing checkpoint path; bootstrap is always available unless explicitly null.

**KTD3 — Calibration JSON update.** Update `docs/benchmarks/preflight-calibration.json` and `docs/benchmarks/unified-tournament-calibration.json`: drop `incumbent_checkpoint_path`, add `incumbent_bootstrap_opponent: "nearest_sniper"`.

## Implementation Units

### U1. Spec + incumbent resolution

**Files:** `src/artifacts/tournament/unified/spec.py`, `src/artifacts/tournament/unified/incumbent.py`, `src/artifacts/tournament/resolve.py`, `src/config/schema.py`, `conf/artifacts/base.yaml`

**Test scenarios:**

| ID | Scenario | Expected |
|----|----------|----------|
| T1 | No promoted manifest, default spec | `resolve_incumbent` returns scripted nearest_sniper with `agent_id=incumbent` |
| T2 | Promoted manifest exists | Checkpoint incumbent from manifest |
| T3 | Challenger ckpt path ≠ incumbent path on bootstrap | Incumbent is scripted, not challenger ckpt |

### U2. Calibration + CLI wiring

**Files:** `docs/benchmarks/preflight-calibration.json`, `docs/benchmarks/unified-tournament-calibration.json`, `src/jax/unified_tournament_calibration.py`, `src/cli/benchmark.py`

**Test scenarios:**

| ID | Scenario | Expected |
|----|----------|----------|
| T4 | Load committed preflight calibration | Spec has `incumbent_bootstrap_opponent=nearest_sniper`, no checkpoint path |
| T5 | `default_unified_tournament_stub` | Emits bootstrap opponent, not checkpoint path |

### U3. Test updates

**Files:** `tests/test_unified_tournament_incumbent.py`, `tests/test_unified_tournament_spec.py`, `tests/test_unified_tournament_ladder.py`, `tests/test_unified_tournament_calibration.py`

**Test scenarios:**

| ID | Scenario | Expected |
|----|----------|----------|
| T6 | Bootstrap incumbent differs from challenger | Agent IDs and checkpoint paths differ |
| T7 | `enforcement=true` without promoted manifest | Stage 2 runs (not `no_incumbent`) |

## Verification

- `make test-fast` (required before push)
- Optional operator: re-run `ow benchmark tournament-proof` on post-hygiene ckpt after merge

## Dependencies

- Existing baseline mapping: `nearest_sniper` → `sniper` in `src/artifacts/tournament/runner.py`
