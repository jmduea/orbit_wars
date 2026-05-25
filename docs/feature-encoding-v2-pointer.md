# Feature Encoding v2 — Pointer Action Space

## v1 vs v2

| | v1 | v2 |
|---|----|----|
| Target selection | Candidate slot index (0=no-op) | Joint `(source, target)` edge index |
| Visibility | Top C-1 ranked targets | Top-K edges per source (ralplan default) |
| Shield | Slot-based `candidate_ids` / `target_angles` | Edge-based src/tgt + angle lookup |

## Joint Pointer (Locked)

- Action selects **(source_planet, target_planet)** from legal owned×active edges
- **NO_OP** as extra logit or dedicated stop edge
- Ship bucket conditioned on chosen pair

## Legality Mask (Action Time)

1. Source owned by learner, active
2. Target active, target ≠ source
3. Sun-crossing (policy decision: mask at action — align with JAX v1)
4. Trajectory shield at sample time (bucket-dependent)

## Phase 0 ADR Required

Before coding: flat edge index layout, NO_OP encoding, K-step multi-launch semantics, submission API planet-id confirmation.

## Side-by-Side Caveat

v1 vs v2 ablation compares **win rate / reward / throughput**, not action-space equivalence.
