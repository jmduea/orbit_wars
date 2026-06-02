---
title: "feat: Planet Flow queue residuals (#166–#170)"
date: 2026-06-02
status: completed
type: feat
origin: "docs/ROADMAP.md Now slice; GitHub #166–#170"
---

# feat: Planet Flow Queue Residuals (#166–#170)

## Summary

Close the Planet Flow **Now** roadmap slice by verifying the shipped shortlist CLI (#166), centralizing Planet Flow metric descriptors (#169), strengthening compiler-control guardrail tests (#170), and extracting a shared PPO epoch driver (#168). Split decoder replay batch contracts (#167) is deferred — it is a cross-cutting rollout/PPO refactor that deserves its own pass.

---

## Problem Frame

Prior LFG on Planet Flow (`0984f54`, proof pipeline plan) landed sweep shortlist, noop smoke, and sweep_score v3. Residual review findings filed GitHub issues #166–#170. #166 is already implemented on `main`; #167–#170 remain open maintainability and verification gaps.

---

## Requirements

- **R1 (#166):** Confirm `ow benchmark shortlist-planet-flow-sweep` and noop smoke are tested and documented; close issue in PR.
- **R2 (#169):** Define Planet Flow metric family once; derive registry, contract tuples, and delta finalization suffix lists from descriptors.
- **R3 (#170):** Compiler-control tests assert finite seeded diagnostics, positive demanded mass on active fixtures, same-seed stability, and numeric finalized control/delta keys.
- **R4 (#168):** Extract shared PPO ratio/clip/metrics/scan finalization used by factorized and Planet Flow update paths.
- **R5:** Update `docs/ROADMAP.md` Done/Now on merge.

---

## Key Technical Decisions

- **#166 verification-only** — no new CLI surface; existing `src/jax/planet_flow_shortlist.py` and benchmark subcommands satisfy R8/R9 from proof pipeline plan.
- **Descriptor module colocated with contract** — extend `src/jax/rollout/metric_contract.py` with structured Planet Flow descriptors; telemetry registry imports generated definitions.
- **Shared PPO helpers in `ppo_update.py`** — extract ratio/KL/format-metrics/scan-finalize without changing numerical behavior.
- **Defer #167** — nested replay union touches `JaxTransitionBatch`, rollout collect, concat, and both PPO paths; out of scope for this PR.

---

## Scope Boundaries

### Non-goals

- Split decoder replay batch contracts (#167) — follow-up issue/PR.
- New sweep recipes, learn-proof threshold changes, or reachability mask work.
- Re-doing submit-valid operator closure (#176) or CLI hardening (#160/#161).

### Deferred to Follow-Up Work

- **#167:** `PlanetFlowReplayFields` / `FactorizedReplayFields` union on `JaxTransitionBatch`.

---

## Implementation Units

### U1. Verify sweep shortlist CLI (#166)

**Goal:** Confirm operator entry point is complete and tested.

**Files:** `src/jax/planet_flow_shortlist.py`, `src/cli/benchmark.py`, `tests/test_planet_flow_shortlist.py`, `docs/benchmarks/preflight-calibration.md`

**Verification:** Existing tests pass; PR references Closes #166.

### U2. Centralize Planet Flow metric descriptors (#169)

**Goal:** Single descriptor list drives contract keys, registry entries, and delta suffix enumeration.

**Files:**
- `src/jax/rollout/metric_contract.py`
- `src/telemetry/planet_flow_registry.py` (new)
- `src/telemetry/metric_registry.py`
- `src/jax/train/metrics.py`
- `tests/test_planet_flow_metrics.py`

**Test scenarios:**
- All `PLANET_FLOW_*` contract tuples match descriptor-derived names.
- Registry contains every descriptor name with `action_decision` group.

### U3. Strengthen compiler-control tests (#170)

**Goal:** Behavioral guardrails on seeded control path and rollout finalization.

**Files:**
- `tests/test_planet_flow_compiler.py`
- `tests/test_planet_flow_metrics.py` (numeric delta assertions if needed)

**Test scenarios:**
- Seeded control: finite diagnostics, positive `demanded_mass` when active planets exist, same-seed equality.
- Finalized control/delta keys are finite floats after `finalize_cross_chunk_rate_metrics`.

### U4. Extract shared PPO epoch driver (#168)

**Goal:** Deduplicate ratio/clip/loss metrics and scan finalization between factorized and Planet Flow PPO.

**Files:**
- `src/jax/ppo_update.py`
- `tests/test_planet_flow_action_contract.py` (regression)

**Verification:** `make test-fast` green; PPO metric names unchanged.

---

## Verification

- `make test-fast`
- Targeted: `uv run pytest tests/test_planet_flow_shortlist.py tests/test_planet_flow_metrics.py tests/test_planet_flow_compiler.py tests/test_planet_flow_action_contract.py tests/test_benchmark_cli.py -q`
