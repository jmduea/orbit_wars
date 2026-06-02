---
title: Planet Flow replay "wrong target" from catalog reachability, not angle bugs
date: 2026-06-02
category: logic-errors
module: planet-flow-compiler
problem_type: logic_error
component: planet_flow_compiler
symptoms:
  - "u150 2p_random replay shows repeated launches from home 16 toward neutral 24, never enemy 19"
  - "planet_flow_unreachable_demand_rate ~0.65 and held_demand_rate ~0.84 at u150"
  - "Entropy collapsed (~3.8e-5); planet_flow_sweep_score = -1.0; learner never expands territory"
root_cause: design_mismatch
resolution_type: diagnosis
severity: high
tags:
  - planet-flow
  - compiler
  - reachability
  - top-k-edges
  - replay
  - ppo-signal
related_components:
  - src/jax/planet_flow.py
  - src/features/catalog/edge.py
  - src/jax/train/sweep_score.py
  - docs/brainstorms/2026-06-01-planet-flow-policy-requirements.md
---

# Planet Flow replay "wrong target" from catalog reachability, not angle bugs

## Problem

Replay `replay_u000150_2p_random.html` (run `20260602T030312Z-s42-232b5bed`, checkpoint u150, seed 192) shows the learner spam-launching from home planet **16** toward neutral **24**, never toward enemy home **19**. The agent loses (-1 reward). Visuals suggest broken targeting.

## Root cause chain

1. **P0 compiler contract:** `compile_planet_flow_action` turns per-planet demand into launches via per-owned-source **argmax over top-K edge slots** (`candidate_count=6` → K=5 legal targets per source).

2. **Catalog gap on seed 192:** The learner only ever owns planet 16. From row 16, the legal edge catalog contains **neutrals only**: 24, 20, 8, 12, 4. Enemy home **19 never appears** in any owned source's catalog for steps 2–150.

3. **Unreachable heatmap mass:** The policy can assign high demand to planet 19 globally, but that demand is **not reachable** through the catalog. Training logs at ~u150: `planet_flow_unreachable_demand_rate ≈ 0.65`, `held_demand_rate ≈ 0.84`.

4. **Compiler still fires:** Each turn the compiler launches at the **best legal edge** — highest pressure among catalog targets. That is neutral 24 (angle ~1.47 rad, geometrically correct for snapshot geometry). Angles drift ~0.03/step because planet 24 orbits and the compiler recomputes `arctan2` each turn.

5. **Policy collapse compounds the trap:** Entropy ≈ 3.8×10⁻⁵; `planet_flow_sweep_score = -1.0`. The policy never expands territory, so enemy 19 never enters any source's edge list. Bad learning, but the visual "wrong target" effect would still occur whenever demand sits on planets outside local top-K catalogs.

## Not the cause

- **Replay/inference bug:** `build_checkpoint_agent` → `compile_planet_flow_action` matches training. Angles match selected catalog targets.
- **Orbiting home angle bug:** Launch angles use **current** `game_row.planets.x/y`, not `initial_planets` (same as factorized `_launch_angle_for_edge`). Orbiting explains small angle drift, not attacking the wrong planet class.

## Secondary robustness fix

`_edge_target_pressure` previously used `jnp.take(target_pressure, planet_id)`, assuming **row index == planet id**. Fixed to gather demand by matching `game_row.planets.id` to each edge target id (same idea as launch-angle lookup). Kaggle/JAX env today keeps id == row, but permuted or duplicate-id layouts misread demand without the fix.

## Design tension (brainstorm input)

| Option | Idea |
|--------|------|
| A. Reachability-masked head | Policy only outputs demand on planets reachable from at least one owned source this turn |
| B. Compiler rewrite | Flow-based allocation / multi-hop intent, not independent per-source argmax |
| C. Catalog change | Ensure enemy homes enter catalogs (distance rank, threat bias) |
| D. Training-only | Keep P0 compiler; add reachability mask in loss + sweep gates on `unreachable_demand_rate` |

**Prevention:** Align PPO target space with compiler reachability so unreachable heatmap mass does not convert into misleading neutral spam.

**Brainstorm outcome:** `docs/brainstorms/2026-06-02-planet-flow-reachability-contract-requirements.md` — adopt reachability-masked PPO (A) + hard gates (D); defer compiler rewrite (B); optional catalog threat slot (C-lite) if A+D insufficient.

## Tests added

- `test_edge_target_pressure_uses_target_row_not_planet_id_index` — id ≠ row pressure gather
- `test_compile_planet_flow_action_fires_best_catalog_edge_when_enemy_unreachable` — documents P0 hold/spill to best catalog neutral
- `test_compile_planet_flow_action_uses_current_orbiting_positions_for_angle` — orbiting is not an angle-calculation bug

## Verification commands

```bash
uv run pytest tests/test_planet_flow_compiler.py -q
```

## Related

- Prior sweep objective issue: `docs/solutions/logic-errors/planet-flow-sweep-gameable-objective.md`
- Pipeline plan: `docs/plans/2026-06-02-002-fix-planet-flow-angle-verify-relaunch-pipeline-plan.md`
