# Feature Encoding v2 — Pointer Action Space

**Status:** ADR-001 (joint flat, production default) + **ADR-005 (factorized top-K, planned M1)**

## v1 vs v2 (joint flat — current default)

| | v1 | v2 joint flat |
|---|----|----|
| Target selection | Candidate slot index (0=no-op) | Joint `(source, target)` edge index |
| Visibility | Top C-1 ranked targets | Top-K edges per source (K = C−1) |
| Shield | Slot-based `candidate_ids` / `target_angles` | Edge-based src/tgt + angle lookup |
| Logits | C-way per owned source | Flat `MAX_PLANETS×K + 1` (+ NO_OP) |

## Joint Pointer (ADR-001 — current default)

- Action selects **(source_planet, target_planet)** from legal owned×active edges via **flat index**
- **NO_OP** at flat index `MAX_PLANETS * K`
- Ship bucket conditioned on chosen pair
- `max_moves_k` sub-steps reuse pointer within turn; mask exhausted sources

## Factored Top-K Pointer (ADR-005 — M1 target)

Per launch step within fixed `max_moves_k` loop:

1. **Stop head** — terminate launch sequence early (replaces NO_OP-in-joint-softmax semantics)
2. **Source head** — pointer over owned planets with ships
3. **Target-slot head** — pointer over `K` slots for chosen source (same top-K as ADR-002)
4. **Bucket head** — ship bucket conditioned on `(source, slot)`

```
log π = log π_stop + active × (log π_src + log π_tgt_slot + log π_bucket)
```

Rollout stores `source_index`, `target_slot`, `stop_flag`, `step_mask` (not flat `target_index`).

Shield reuses `evaluate_edge_pair(src_row, slot)` — same O(P×K) cost class as joint flat.

## Flat index layout (ADR-001 joint flat only)

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
4. Trajectory shield at sample time (bucket-dependent) via `evaluate_edge_pair`
5. Joint flat: NO_OP always legal at flat index `P×K`

## Submission decode

Joint flat:

```
(src, tgt) = edge_lookup(flat_idx)
angle_abs = canonical_launch_angle + θ_ref   # ADR-004
ships = bucket_to_count(garrison[src], bucket)
API move = [src_planet_id, angle_abs, ships]
```

Factored (planned): resolve `(source_index, target_slot)` → same angle/ships mapping.

Game API accepts planet-id source + angle; target is geometry-implied.

## References

- Full ADR: `docs/feature-encoding-v2.md` ADR-001, ADR-005
- Phase 0 evidence: `docs/feature-encoding-v2-phase0-results.md`
- M1 plan: `.omg/plans/ralplan-factored-pointer-decoder.md`
