# Feature Encoding v2 — Pointer Action Space

**Status:** Phase 0 finalized (ADR-001)

## v1 vs v2

| | v1 | v2 |
|---|----|----|
| Target selection | Candidate slot index (0=no-op) | Joint `(source, target)` edge index |
| Visibility | Top C-1 ranked targets | Top-K edges per source (K = C−1) |
| Shield | Slot-based `candidate_ids` / `target_angles` | Edge-based src/tgt + angle lookup |
| Logits | C-way per owned source | Flat `MAX_PLANETS×K + 1` (+ NO_OP) |

## Joint Pointer (Locked)

- Action selects **(source_planet, target_planet)** from legal owned×active edges
- **NO_OP** at flat index `MAX_PLANETS * K`
- Ship bucket conditioned on chosen pair
- `max_moves_k` sub-steps reuse pointer within turn; mask exhausted sources

## Flat index layout

```
K = max(0, candidate_count - 1)
flat_idx = src_row * K + slot_k     # src_row ∈ [0, MAX_PLANETS)
NO_OP_idx = MAX_PLANETS * K
```

Edge list order: row-major over `(source planet row, K slot)`.

## Legality Mask (Action Time)

1. Source owned by learner, active
2. Target active, target ≠ source
3. Sun-crossing masked (JAX v1 training alignment)
4. Trajectory shield at sample time (bucket-dependent)
5. NO_OP always legal

## Submission decode

```
(src, tgt) = edge_lookup(flat_idx)
angle_abs = canonical_launch_angle + θ_ref   # ADR-004
ships = bucket_to_count(garrison[src], bucket)
API move = [src_planet_id, angle_abs, ships]
```

Game API accepts planet-id source + angle; target is geometry-implied.

## Side-by-Side Caveat

v1 vs v2 ablation compares **win rate / reward / throughput**, not action-space equivalence.

## References

- Full ADR: `docs/feature-encoding-v2.md` ADR-001
- Phase 0 evidence: `docs/feature-encoding-v2-phase0-results.md`
