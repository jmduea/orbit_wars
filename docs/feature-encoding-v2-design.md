# Feature Encoding v2 — Design Summary

See ralplan: `.omg/plans/ralplan-feature-encoding-v2.md`

## Direction

- **Greenfield v2** side-by-side with v1
- **Planet tensor** `(MAX_PLANETS, P)` + **edge features** for owned→active pairs
- **Global** `(G,)` with optional global-only history stack when H>1
- **Joint pointer** over `(source, target)` + ship bucket
- **JAX-only** at cutover (training + submission)

## Interface Options (Explored)

| Option | Summary |
|--------|---------|
| A Structured flat | Minimal refactor; rejected for v2 greenfield |
| B Planet tensor | **Selected** as canonical encoder output |
| C Token sequence | Transformer consumption layer on B |
| D Hybrid flat | Fallback if B+C slips |

**Decision:** B encoder + C consumption (GNN on graph; transformer on planet tokens).

## History

- Planet **delta fields** each step
- **Global-only** frame stack when `feature_history_steps > 1`
- Avoid v1 full ×H blow-up on all groups

## Edge Representation (Ralplan)

**Recommended:** top-K edges per owned source (K ≈ `candidate_count - 1`), not dense `(P,P,E)`.

## Policy v2 v1

**GNN pointer first**; transformer adapter deferred.

## Pointer

Joint `(source, target)` over valid edges + NO_OP. See `docs/feature-encoding-v2-pointer.md` (ADR-001 finalized Phase 0).

**Schema (locked):** P=13, E=12, G=46 — see `docs/feature-encoding-v2-phase0-results.md`.

## Ship feature scale (ADR-003)

v2 renames v1's `task.max_ships` → **`task.ship_feature_scale`**: encoder normalization only, default `1000.0`. Not the fleet-speed cap (that uses hardcoded `log(1000)` in `fleet_speed()` and `MAX_FLEET_SPEED = 6.0`). See `docs/feature-encoding-v2.md` ADR-003.

## Symmetry & canonicalization (ADR-004)

Learner frame: **`θ_ref`** = angle from sun to **unweighted centroid of owned planets** (same 2p/4p). Planets use sun-polar `r, θ`; edges carry learner-frame geometry; decode adds `θ_ref` back for absolute launch angles. Details: `docs/feature-encoding-v2-symmetry.md`.
