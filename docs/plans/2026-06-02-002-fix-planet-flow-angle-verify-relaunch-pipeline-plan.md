---
title: "fix: Verify Planet Flow launch angles and relaunch sweep pipeline"
type: fix
status: completed
date: 2026-06-02
origin: docs/solutions/logic-errors/planet-flow-sweep-gameable-objective.md
related:
  - docs/solutions/logic-errors/planet-flow-catalog-reachability-mismatch.md
---

# Plan: Planet Flow Angle Verification + Sweep Pipeline Relaunch

**Target repo:** `.worktrees/feat/planet-flow-policy` on branch `feat/planet-flow-policy`

## Summary

Confirm whether orbiting home-planet motion causes Planet Flow launch-angle bugs (replay fleets missing intended targets), then relaunch the Planet Flow PPO W&B sweep on `training=2p4p_16_split` and wire calibration/learn-proof to the same training profile.

## Debug Summary (catalog reachability — primary root cause)

Full write-up: [`planet-flow-catalog-reachability-mismatch.md`](../solutions/logic-errors/planet-flow-catalog-reachability-mismatch.md).

| Finding | Detail |
|---------|--------|
| Visual symptom | u150 replay seed 192: spam launches 16→24, never 16→19; -1 reward |
| Structural cause | Enemy 19 outside top-K catalog from sole owned source 16 for entire game |
| Metrics | `unreachable_demand_rate ≈ 0.65`, `held_demand_rate ≈ 0.84`, entropy ≈ 3.8×10⁻⁵ |
| Not a bug | Replay path matches training; angles use live coords and match catalog target |
| Design follow-up | Brainstorm A–D: reachability-masked head, compiler rewrite, catalog bias, or training gates |

Orbiting-home hypothesis is **secondary** (angle drift ~0.03/step from moving target, not wrong planet selection).

## Problem Frame

Replay at u150 shows repeated launches from home planet 16 toward neutral 24, never enemy 19. Prior debug traced catalog reachability (~65% unreachable demand) and policy collapse. User hypothesis: non-static home planet breaks angle calculation so fleets never hit intended targets even when catalog target is correct.

## Requirements

| ID | Requirement |
|----|-------------|
| R1 | Prove or disprove orbiting-source angle bug with a targeted compiler test |
| R2 | Sweep fixed axis uses `training=2p4p_16_split` (not `2p_16`) |
| R3 | Planet Flow calibration overrides match sweep training profile |
| R4 | Regenerate W&B sweep YAML and start a new agent (do not resume `40il23b3`) |
| R5 | Document/run calibration → learn-proof commands after sweep config lands |

## Key Technical Decisions

**KTD1 — Angle uses live source coordinates.** `compile_planet_flow_action` computes `arctan2` from `game_row.planets.x/y` (current positions), matching factorized `_launch_angle_for_edge`. Orbiting source is not a separate bug class; snapshot angles can still miss moving intercept targets (game physics).

**KTD2 — Sweep profile `2p4p_16_split`.** Matches throughput proof and exercises both 2p/4p rollout groups; addresses single-format training gap from prior debug.

**KTD3 — New W&B sweep ID.** Old sweep `40il23b3` pinned `training=2p_16`; register fresh sweep after YAML regen.

## Implementation Units

### U1. Orbiting-source angle regression test

**Goal:** Lock in that compiler angles use current source/target positions, not static initial coords.

**Requirements:** R1

**Files:** `tests/test_planet_flow_compiler.py`

**Approach:** Build batched game where owned source row has `x,y` shifted from `initial_planets`; set target pressure on a reachable edge; assert emitted angle equals `arctan2` from current source to current target.

**Test scenarios:**
- Source orbit offset: initial vs current coords differ; angle must use current
- Target orbit offset: angle updates when only target position changes

**Verification:** `uv run pytest tests/test_planet_flow_compiler.py -q`

### U2. Sweep config + generator test

**Goal:** Planet Flow sweep uses `2p4p_16_split`.

**Requirements:** R2

**Files:** `conf/wandb_sweep/fixed/planet_flow_ppo_signal.yaml`, `tests/test_config_consolidation.py`

**Verification:** `test_planet_flow_ppo_signal_sweep_generates_expected_guardrails`

### U3. Calibration profile alignment

**Goal:** Planet Flow preflight calibration trains on `2p4p_16_split` when `model=planet_flow_target_heatmap`.

**Requirements:** R3

**Files:** `src/jax/preflight_calibration.py`, `tests/test_preflight_calibration.py` (if assertion exists)

**Verification:** targeted preflight config test

### U4. Relaunch sweep → calibration → learn-proof

**Goal:** Regenerate sweep, register new W&B sweep, start agent; run calibration and learn-proof entrypoints.

**Requirements:** R4, R5

**Commands:**
```bash
uv run ow make wandb_sweep=planet_flow_ppo_signal
uv run wandb sweep outputs/_meta/sweeps/planet_flow_ppo_signal.yaml
uv run wandb agent <entity>/planet-flow-policy/<new_id>
make preflight-calibrate  # after sweep shortlist or with default overrides
uv run ow benchmark learn-proof --through beat_random ...
```

**Verification:** sweep YAML contains `2p4p_16_split`; agent log shows new training value

## Scope Boundaries

### In scope
- Angle verification test + sweep/calibration profile alignment + sweep relaunch

### Deferred to Follow-Up Work
- Compiler reachability-mask redesign (brainstorm outcome)
- Full sweep completion and winner shortlist automation

## Risks & Dependencies

- GPU contention: run one heavy job at a time
- Learn-proof requires calibration thresholds from `preflight-calibration.json`; do not relax gates
