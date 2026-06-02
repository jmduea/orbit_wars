---
title: "feat: Planet Flow sweep v2 with replay and proof score"
type: feat
status: active
date: 2026-06-02
origin: docs/plans/2026-06-01-004-feat-planet-flow-ppo-sweep-plan.md
---

# feat: Planet Flow Sweep v2 (random_only + proof score + async replay)

## Summary

Replace the gameable `overall_win_rate` / `noop_only` sweep with a v2 recipe that optimizes a collapse-resistant `planet_flow_sweep_score`, tunes on `random_only` (200 updates), and enables async local HTML replays so sweep → calibration → learn-proof runs are visually inspectable.

## Problem Frame

Sweep v1 (`3zeu25xq`) produced a degenerate optimum: zero launches, entropy collapse, `overall_win_rate=1.0` vs noop, while Planet Flow learned metrics went to zero. W&B Bayes treated this as a winner. Replay was blocked by Planet Flow P0 runtime guards (`artifacts=disabled`).

## Requirements

- Log `win_rate_delta_10` and `planet_flow_sweep_score` each training update when action-decision metrics are enabled.
- W&B sweep optimizes `planet_flow_sweep_score` (maximize), not raw `overall_win_rate`.
- Sweep fixed profile: `opponents=random_only`, `training.total_updates=200`, `artifacts=planet_flow_proof`.
- `planet_flow_proof` artifacts: pipeline + async local replay only; no Docker/tournament/promotion.
- Relax Planet Flow runtime guards to allow replay-only artifact paths.
- Replay worker loads Planet Flow checkpoints (`allow_planet_flow=True`).
- Gate Planet Flow rate metrics when demand mass is below a floor.
- Update sweep tests and follow-up shortlist guardrails.
- Calibration and learn-proof Planet Flow runs use `artifacts=planet_flow_proof` for visual replay parity.

## Key Technical Decisions

- **Objective:** `planet_flow_sweep_score = win_rate_delta_10` when activity/stability floors pass, else `-1.0`. Aligns Bayes with preflight trend gates and rejects launch collapse.
- **Opponent:** `random_only` for sweep tuning; learn-proof ladder still validates `beat_noop` → `beat_random`.
- **Replay:** `replay_backend=local`, `replay_async=true`, `checkpoint_every=50`, HTML under `evaluations/replay_u*/`.

---

## Implementation Units

### U1. Proof score telemetry

**Goal:** Emit rolling trend and composite sweep score on each update.

**Files:** `src/jax/train/sweep_score.py`, `src/jax/train/loop.py`, `src/telemetry/metric_registry.py`, `tests/test_planet_flow_sweep_score.py`

**Test scenarios:** Score is `-1` when launches=0; positive when win trend and activity floors met; delta matches first/last 10-window mean.

### U2. Planet Flow rate metric floors

**Goal:** Avoid misleading `held_demand_rate=1.0` at near-zero demand.

**Files:** `src/jax/train/metrics.py`, `tests/test_planet_flow_metrics.py`

### U3. Replay-only artifacts profile

**Goal:** `artifacts=planet_flow_proof` enables async local replays without promotion/Docker.

**Files:** `conf/artifacts/planet_flow_proof.yaml`, `src/config/runtime.py`, `tests/test_config_consolidation.py`

### U4. Replay runtime for Planet Flow checkpoints

**Goal:** HTML replays render Planet Flow policies.

**Files:** `src/artifacts/tournament/runner.py`

### U5. Sweep v2 config recipe

**Goal:** Update W&B sweep to v2 axes and metric.

**Files:** `conf/wandb_sweep/metric/planet_flow_sweep_score.yaml`, `conf/wandb_sweep/planet_flow_ppo_signal.yaml`, `conf/wandb_sweep/fixed/planet_flow_ppo_signal.yaml`, `tests/test_config_consolidation.py`

### U6. Proof path artifacts parity

**Goal:** Calibration and preflight Planet Flow trains use `planet_flow_proof`.

**Files:** `src/jax/preflight_calibration.py`, `src/jax/preflight.py`, `outputs/_meta/sweeps/planet_flow_sweep_followup.py`

### U7. Verification

**Goal:** Targeted tests pass; regenerate sweep YAML.

**Verification:** `make test-domain-config`, planet flow metric tests, `uv run ow make wandb_sweep=planet_flow_ppo_signal`.

---

## Scope Boundaries

### In Scope

- Sweep v2 config, proof score telemetry, replay-only artifacts, guardrail updates.

### Out of Scope

- Running W&B agents in CI.
- Planet Flow Docker validation or tournament replay paths.
- Committing calibrated thresholds to `docs/benchmarks/preflight-calibration.json`.

### Deferred to Follow-Up Work

- `#166` sweep shortlist CLI in `ow`.
- Composite score weighting compiler-control deltas beyond activity floors.

---

## Operational Notes

1. `uv run ow make wandb_sweep=planet_flow_ppo_signal`
2. Register **new** W&B sweep (do not resume `3zeu25xq`).
3. `uv run wandb agent …`
4. HTML replays: `outputs/campaigns/<campaign>/runs/<run_id>/evaluations/replay_u*/replay/*.html`
5. Shortlist → `planet_flow_sweep_followup.py` → calibrate → learn-proof
