---
title: "feat: Observability debug metrics bundle"
date: 2026-06-02
status: active
type: feat
origin: "docs/ROADMAP.md Now + Next telemetry gating"
---

# feat: Observability Debug Metrics Bundle

## Summary

Add ROADMAP **Now** debug metric `mean_ships_per_launch` (average ships per fleet launch) and complete **Next** telemetry gating: per-format PPO `_2p`/`_4p` loss diagnostics and `update_time_*_fraction` metrics default off behind `telemetry.metric_groups.debug`.

---

## Problem Frame

Operators need launch sizing signal (`mean_ships_per_launch`) without bloating default JSONL. Per-format PPO loss splits and update-time fraction metrics are diagnostic-only but still register under `losses`/`timing` groups, so they leak into default training logs.

---

## Requirements

- **R1:** Emit `mean_ships_per_launch` from rollout (factorized + Planet Flow paths) when `metric_groups.debug` is enabled.
- **R2:** Register launch sizing metrics in the `debug` telemetry group; use sum/count + cross-chunk finalize pattern.
- **R3:** Move PPO `*_2p`/`*_4p` loss diagnostics and `loss_sample_count_*` from `losses` to `debug` group.
- **R4:** Move `update_time_rollout_fraction` and `update_time_ppo_fraction` from `timing` to `debug` group.
- **R5:** Update metric registry tests and add rollout unit coverage for launch sizing.
- **R6:** Triage `docs/ROADMAP.md` (Now → Done; Next telemetry item → Done or deferred with note).

---

## Key Technical Decisions

- **Sum/count finalize** — `launch_ship_count_sum` / `active_launch_count` merge across chunks; `mean_ships_per_launch` derived in `finalize_cross_chunk_rate_metrics` (matches win-rate pattern).
- **Planet Flow reuse** — derive launch ship sums from existing `planet_flow_emitted_ship_mass_sum` and `planet_flow_emitted_launch_count` when Planet Flow data is present.
- **Group moves only** — PPO still computes format-split metrics; `filter_update_record` drops them unless debug is enabled (no PPO logic fork).

---

## Scope Boundaries

### Deferred to Follow-Up Work

- Cursor session-start hook (ROADMAP Next #2) — separate PR.
- Additional launch hygiene metrics (`duplicate_launch_rate`, `friendly_ping_pong_rate`) from ideation doc.

---

## Implementation Units

### U1. Metric contract and registry

**Goal:** Register debug launch sizing metrics; re-home gated telemetry metrics to `debug` group.

**Files:** `src/jax/rollout/metric_contract.py`, `src/telemetry/metric_registry.py`

**Test scenarios:** Registry group assignments; enabled names with debug off omit `_2p`/`_p4` losses and update-time fractions.

### U2. Rollout launch sizing computation

**Goal:** Accumulate launch ship sums in factorized and Planet Flow rollout metric paths when debug keys are active.

**Files:** `src/jax/rollout/metrics.py`, `src/jax/train/metrics.py`

**Test scenarios:** Synthetic factorized data yields expected mean; Planet Flow sums map to mean after finalize.

### U3. Tests and ROADMAP

**Goal:** Update registry/timing tests; add focused launch metric test; triage ROADMAP.

**Files:** `tests/test_metric_registry.py`, `tests/test_launch_debug_metrics.py`, `tests/test_jax_train_timing.py`, `docs/ROADMAP.md`

**Verification:** `make test-fast` green.
