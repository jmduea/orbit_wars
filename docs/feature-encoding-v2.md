# Feature Encoding v2

**Status:** Phase 0 contract draft  
**Plan:** `.omg/plans/ralplan-feature-encoding-v2.md`  
**ADR:** ADR-001 (action space), ADR-002 (edge layout) below

## Overview

v2 replaces v1's self/candidate/global flat groups with:

- **Planet tensor** `(MAX_PLANETS, P)` — per-planet state
- **Edge list** — top-K `(source → target)` pairs per owned source with features `(E,)`
- **Global vector** `(G,)` — game-level aggregates + optional global history stack

**Action space:** joint pointer over legal edges `(source_planet, target_planet)` + ship bucket + NO_OP.

**Runtime:** JAX encoder canonical at cutover; v1 side-by-side until ablation passes gates.

---

## ADR-001: Action Space

### Decision

Joint pointer selects an **edge index** from a flat list of legal `(source, target)` pairs for the current decision step. Ship bucket is sampled conditioned on the chosen edge.

### Index space

```
edges[e] = (src_planet_id, tgt_planet_id)   for e in 0..E_flat-1
logits[e] for e in 0..E_flat-1
logits[NO_OP]  — dedicated stop logit (index E_flat)
```

**K-step launches:** For `max_moves_k > 1`, repeat pointer+ship within turn; decrement available ships per source; mask edges whose source has no ships remaining.

### Legality mask (action sampling)

| Rule | Mask |
|------|------|
| Source active and owned by learner | Required |
| Target active | Required |
| `src != tgt` | Required |
| Sun-crossing shot | **Masked** (align with JAX v1 training) |
| Trajectory shield | Applied at sample time per bucket |
| NO_OP | Always legal |

### Game API mapping (submission)

External format unchanged: `[source_planet_id, angle, ships]`.

Internal decode:

```
(src, tgt) = edges[e]
angle = atan2(tgt.y - src.y, tgt.x - src.x)
ships = bucket_to_count(source_ships, bucket)
```

Target planet is **angle-implied** at game API — must match training shield geometry.

### File touch list

See ralplan appendix. Critical: `trajectory_shield.py`, `opponents/jax_actions/builders.py`, three inference decode loops (packager, runtime, replay).

---

## ADR-002: Edge Layout (Option A — Approved)

### Decision

**Top-K edges per owned source**, not dense `(P, P, E)`.

- `K = max(0, candidate_count - 1)` — continuity with v1 slot budget
- Sort key: distance ascending, sun-blocked deprioritized (same spirit as v1 lexsort)
- Flat edge list order: row-major over owned sources × K slots

### Rationale

Dense `(60, 60, E)` ≈ 3600×E floats vs v1 ~171/decision row — rollout memory and JIT compile risk (architect/critic review).

### Shapes (static JIT)

```
planet_features:  (MAX_PLANETS, P)
planet_mask:      (MAX_PLANETS,)
edge_features:    (MAX_PLANETS, K, E)     # padded; invalid slots zeroed
edge_mask:        (MAX_PLANETS, K)        # valid target for (src, slot)
edge_src_ids:     (MAX_PLANETS,)          # planet id per row (owned sources padded)
edge_tgt_ids:     (MAX_PLANETS, K)
global_features:  (G,) or (H * G,) if history
```

Flat pointer logits: reshape `(MAX_PLANETS * K + 1,)` with NO_OP at end, masked.

---

## Schema v2 (Draft — dims TBD Phase 0 lock)

### Planet features (P — target ~12–14)

| Field | Dim | Normalization |
|-------|-----|---------------|
| active | 1 | 0/1 |
| x, y | 2 | / BOARD_SIZE |
| radius | 1 | / 5.0 |
| ships | 1 | / max_ships |
| production | 1 | / MAX_PRODUCTION |
| rotating_flag | 1 | 0/1 |
| owner_slot | 4 | relative one-hot (4p) |
| incoming_friendly_pressure | 1 | / max_ships |
| incoming_enemy_pressure | 1 | / max_ships |
| outgoing_friendly_ships | 1 | / max_ships |
| ship_delta | 1 | / max_ships |

### Edge features (E — target ~10–12)

| Field | Dim | Normalization |
|-------|-----|---------------|
| delta_x, delta_y | 2 | / BOARD_SIZE |
| distance | 1 | / BOARD_SIZE |
| sun_crossing | 1 | 0/1 |
| target_ships | 1 | / max_ships |
| target_production | 1 | / MAX_PRODUCTION |
| target_owner_slot | 4 | relative one-hot |
| turns_to_arrival | 1 | / MAX_STEPS |
| target_incoming_friendly | 1 | / max_ships |
| target_incoming_enemy | 1 | / max_ships |
| target_ship_delta | 1 | / max_ships |
| owner_changed | 1 | 0/1 |

### Global features (G — target ~45, evolved from v1 global-only)

Retain v1 global group semantics without duplicating into planet rows:

- step_fraction, planet/ship/fleet fractions
- owner-relative counts/ships/fleets/production (4 slots)
- active_owner_mask, player_count
- delta slots (ships, planets, fleets, production)

### Dim budget (default H=1, K=3, P≈13, E≈12, G≈45)

Approx per decision: `P*60 + K*60*E + G ≈ 780 + 2160 + 45` — **encoder payload larger than v1 row** but structured; pointer softmax over `60*K+1 ≈ 181` vs v1 `C=4` per source. **Phase 0 spike required** to validate JIT/memory; may reduce P/E or owned-row sparsity.

---

## History

- **Planet:** `ship_delta`, `owner_changed` from prior step (single frame)
- **Global:** stack H frames when `feature_history_steps > 1`
- **Edges:** recompute each step; no frame stack (deltas capture target dynamics)

---

## Checkpoint metadata v2

```yaml
feature_metadata:
  schema_version: 2
  encoding_version: v2
  planet_feature_dim: <P>
  edge_feature_dim: <E>
  global_feature_dim: <G>
  edge_layout: top_k_per_source
  edge_k: <K>
  feature_history_steps: <H>
```

v1 checkpoints rejected when `encoding_version=v2` training loads v1 weights.

---

## Config (sketch)

```yaml
# conf/task/default.yaml (future)
encoding_version: v1   # v1 | v2

# conf/model/gnn_pointer_v2.yaml (future)
architecture: gnn_pointer_v2
```

---

## Ablation cutover gates (defaults)

| Metric | Gate |
|--------|------|
| Win rate | v2 ≥ v1 − 2% |
| 4p rollout throughput | v2 ≥ 85% v1 |
| Shield legal non-noop rate | ± 5pp |
| Seeds | ≥ 3 |
| Formats | 2p + 4p |

---

## Submission audit (Phase 0)

| Finding | Detail |
|---------|--------|
| Current encoding | Python `encode_turn` in packager |
| Policy | JAX |
| Game API | `[source_planet_id, angle, ships]` |
| v2 requirement | JAX `encode_turn_v2`; decode edge → angle |
| Packager | `scripts/validate_kaggle_docker_submission.py` |

---

## v1 → v2 mapping (summary)

| v1 | v2 |
|----|-----|
| self_features per source | planet row (owned) |
| candidate_features per slot | edge_features[src, k] |
| global_features (broadcast) | global_features (once) |
| candidate slot index | flat edge index |
| NO_OP slot 0 | NO_OP logit |

---

## Phase 0 exit checklist

- [x] ADR-001 action space drafted
- [x] ADR-002 edge layout (Option A top-K)
- [x] Schema draft tables
- [x] Submission audit
- [x] Cutover numeric gates documented
- [ ] **P, E, G dims locked** (after float budget spike)
- [ ] **v1 baseline metrics captured**
- [ ] **jax-ppo-split dependency cleared**
