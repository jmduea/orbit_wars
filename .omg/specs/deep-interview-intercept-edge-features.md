# M4: Bucket-aware edge intercept features

**Workflow:** deep-interview → ralplan → omg-autopilot
**Status:** approved (post deep-interview, post ralplan iter-1 descoping)
**Slug:** `intercept-edge-features`
**Origin:** User-initiated `/ralplan M4` from `gnn_pointer` improvement brief. Original brief name was "orbital-time-conditional-features"; scope evolved during interview to include shield thinning, then descoped during ralplan iter-1 to features-only after architect+critic synthesis recommendation.

## Scope Note (post-ralplan iter-1)

Ralplan iteration 1 (planner+architect+critic) surfaced critical issues with combining edge-intercept features and shield-thinning in one milestone. Per architect+critic consensus, scope decoupled:

- **This milestone (M4):** edge intercept features ONLY. Dynamic shield retained unchanged.
- **Follow-up milestone (`thin-trajectory-shield`):** shield thinning work deferred to a separate spec/plan. Registered with `status=deferred` in the workflow manifest.

This descoping eliminates Critic C1/C2/C3 (feature/shield disagreement, static-path guard bypass, legality-net duplication) and Architect A1/A2 (compile-time branching, parity test gating) from M4's surface.

## Goal

Replace current `TurnBatch` edge geometry with **bucket-aware intercept geometry at two anchor fleet speeds**. Expected outcome: policy learns intercept-aware aiming directly from features. Dynamic trajectory shield remains the legality oracle at sample time (unchanged from current behavior).

## Constraints

1. **Two anchor fleet speeds**: `s ∈ {1.0, 6.0}` (slow=1-ship, fast=full-bucket via `fleet_speed` formula).
2. **Edges REPLACE current geometry**: snapshot `delta_x`, `delta_y`, `distance`, `turns` are removed in favor of intercept-aware versions at both anchors.
3. **`crosses` field SPLIT** into two distinct features (raised in critic review C1):
   - `crosses_now` (current aim line, sun-crossing) — matches what the existing shield uses for legality. Retained.
   - `sun_cross_at_intercept` per anchor — predictive feature for what the policy can attend to.
4. **Trajectory shield unchanged** — dynamic shield remains the legality oracle. This milestone does NOT touch `src/game/trajectory_shield.py`.
5. **Submission validity** — unchanged; the existing dynamic shield continues to enforce it.
6. **Schema version bump** in `src/artifacts/checkpoint_compat.py`; new metadata field `intercept_anchors`.
7. **Backward-incompatible** with current checkpoints; existing reward sweep results under v2 do not transfer.
8. **ADRs touched**: ADR-002 (edge content semantically replaced; top-K shape preserved). ADR-001/003/004 untouched.

## Non-Goals

- Per-planet position lookahead features at fixed τ — **deferred to a separate milestone**.
- **Thin trajectory shield — deferred to follow-up milestone `thin-trajectory-shield`** (registered separately with `status=deferred`).
- Iterative / fixed-point intercept math — first-order at two anchors is sufficient.
- Submission legality net additions — not needed; dynamic shield is unchanged.
- Continuous ship counts (still bucketed).
- Action space changes — that's M1 territory.
- Encoder architecture changes — that's M2 territory.
- MCTS / planning — that's M3 territory.

## Acceptance Criteria

### Primary win gate
- `episode_reward_mean` improvement **≥2% over current `gnn_pointer` baseline** at 3 seeds × 500 updates, mixed 2p+4p, matched compute budget.

### Hard gates (must pass; not part of "win")
- Submission validator passes for ≥100 sampled games per format (2p, 4p), zero illegal-action rejections. (Should be free since shield is unchanged.)
- Rollout throughput (`env_steps_per_sec`) NOT a regression vs baseline. NOTE: this milestone does not target throughput improvement; that lives in the deferred `thin-trajectory-shield` milestone.
- No NaN/inf in losses; no policy collapse.

### Diagnostic metrics (informational, not gating)
- `trajectory_shield_legal_non_noop_rate` (unchanged metric; shield untouched).
- Fleet hit rate (fraction of fleets reaching intended target alive).
- `approx_kl`, value loss, entropy curves vs baseline.
- Per-format throughput breakdown (2p vs 4p).

## Assumptions Exposed & Resolved

| # | Assumption | Resolution |
|---|---|---|
| 1 | Original brief said per-planet τ ∈ {5,10,20} position features. | Deferred. Edges-only this milestone. |
| 2 | Existing `turns` edge feature uses constant `MAX_FLEET_SPEED=6.0`. | Identified as an inaccuracy; `fleet_speed` is ship-dependent (1.0→6.0 via log-scaled formula). Two-anchor encoding addresses this. |
| 3 | Shield is required at full fidelity for legality during training. | UNCHANGED — dynamic shield remains M4's legality oracle. Shield thinning deferred to follow-up milestone. |
| 4 | Per-bucket intercept geometry must be encoded at full fidelity. | Two anchor speeds give a sufficient interpolation prior. |
| 5 | Submission validity requires full shield at inference. | UNCHANGED — dynamic shield enforces it. |
| 6 | (formerly: Eliminating dynamic shield is safe for policy quality.) | OUT OF SCOPE for M4 after descoping. Reactivates in `thin-trajectory-shield` follow-up spec. |

## Ontology (Key Entities)

- **`TurnBatch.edge_features`** — replaced field. New dimension TBD by exact list in ralplan.
- **Anchor speeds** — `s ∈ {1.0, 6.0}`. For each: `τ = distance_now / s` (first-order); `target_future_pos = orbit_forward(τ, angular_velocity, rotating_flag)`; `intercept_delta_xy = target_future_pos - source_now`; `sun_crosses_at_intercept` via static check on the new aim line.
- **Thin shield** — subset of current `apply_trajectory_shield_to_turn_batch_v2` keeping static sun, bounds, horizon, target/source/bucket validity. Drops `evaluate_flat_edge`'s dynamic per-bucket trajectory loop.
- **Schema version** — bumps from current; `intercept_anchors` added to checkpoint metadata for forward compatibility.
- **`fleet_speed(ships)`** — game formula in `src/jax/env.py` and `trajectory_shield.py`; speed=1 at ships=1, asymptotes to 6 at ships≥1000.

## Interview Transcript

| Round | Question (weakest dimension) | Answer | Ambiguity after |
|---|---|---|---|
| 1 | Scope of lookahead awareness (Constraint) | Replace edges with intercept geometry (option D) + fleet-speed correction | 55% |
| 2 | Bucket dependence strategy (Constraint) | Two anchor speeds (1.0, 6.0) | 45% |
| 3 | Per-planet lookahead in scope? (Goal) | Defer to separate milestone | 35% |
| 4 | Success criterion specifics (Success) | `episode_reward_mean` ≥2% at 3 seeds × 500 updates | 25% |
| 5 (Contrarian) | Why duplicate shield in features? | Hold two-anchor design **and** thin the shield (scope expansion) | 50% |
| 6 | Shield replacement strategy (Constraint) | Thin shield: drop dynamic, keep static | 18% |

## Open Questions for Ralplan (post-descoping)

These are implementation-level decisions deferred to consensus planning:

1. **Exact replacement edge feature list** — locked in ralplan iter-2: per anchor (5 fields × 2 = 10): `intercept_delta_x`, `intercept_delta_y`, `intercept_distance`, `intercept_turns`, `sun_cross_at_intercept`. Plus retained snapshot fields (8 dims): `crosses_now` (legality-aligned), `tgt_ships`, `owner_slot` (4 dims), `incoming_friendly`, `incoming_enemy`, `ordered_valid`, `tgt_active`. **Total E=18** (vs current E=12).
2. **`tgt_ships` projection** — keep as snapshot. Confirmed deferred; do not project to intercept time.
3. **Top-K sort order under ADR-002** — keep snapshot-distance sort (golden-test stability). Add explicit TODO in code referencing follow-up milestone for intercept-aware re-ranking.
4. **Throughput measurement methodology** — `scripts/benchmark_jax_rl.py --warmup 2 --updates 20 × 3 reps`, pinned baseline and M4 commits, median `env_steps_per_sec` per format. Non-regression gate only.
5. **Non-rotating planet handling** — `is_rotating_xy` predicate already exists; intercept math returns current position when `rotating_flag=False`. Confirm via unit test.

## Next Step

Spec is **approved**. Ralplan iteration 2 produces revised plan with descoped scope. After critic approval, hand to **omg-autopilot** for execution.
