# Feature Encoding v2

**Status:** Phase 0 contract draft  
**Plan:** `.omg/plans/ralplan-feature-encoding-v2.md`  
**ADR:** ADR-001 (action space), ADR-002 (edge layout), ADR-003 (ship feature scale), ADR-004 (symmetry frame) below  
**Symmetry exploration:** `docs/feature-encoding-v2-symmetry.md`

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

## ADR-003: Ship Feature Scale (`task.ship_feature_scale`)

### Decision

Rename v1's misleading `task.max_ships` to **`task.ship_feature_scale`** in v2 config and documentation.

This knob is an **encoder normalization denominator only**. It does not cap planet ships, fleet launches, or simulation behavior.

### What it does

Express ship-related raw counts as fractions for the policy network:

| Feature kind | Formula | Typical range |
|--------------|---------|---------------|
| Garrison / target ships | `min(ships, ship_feature_scale) / ship_feature_scale` | `[0, 1]` |
| Fleet pressure, outgoing ships | `raw_ship_count / ship_feature_scale` | often `[0, 1]`, can exceed 1 if counts are large |
| Ship deltas | `(ships_now - ships_prev) / ship_feature_scale` | roughly `[-1, 1]`, can exceed if swings are large |
| Global ship/fleet aggregates | `total_ships / (MAX_PLANETS * ship_feature_scale)` | `[0, ~1]` |

Config path: **`task.ship_feature_scale`** (Hydra), read by the v2 JAX encoder as `env_cfg.ship_feature_scale`.

**Default:** `1000.0` — chosen to align *conceptually* with the game's fleet-speed reference point (see below), not because the sim reads this config.

### What it is NOT (do not confuse)

| Concept | Where it lives | Role |
|---------|----------------|------|
| **`ship_feature_scale`** | `task.*` config | Feature encoding + telemetry denormalization |
| **Fleet speed reference `1000`** | Hardcoded in `fleet_speed()` | `log(ships) / log(1000)` — at 1000 ships the speed curve saturates toward `MAX_FLEET_SPEED` |
| **`MAX_FLEET_SPEED = 6.0`** | `src/game/constants.py` | Max fleet movement speed in game units per step |
| **Planet ship cap** | *(none via this config)* | Planets can hold more ships than `ship_feature_scale`; garrison features clip at 1.0 |

Fleet speed is computed in `src/jax/env.py` and `src/game/trajectory_shield.py` using a **literal `1000.0`**, not `task.max_ships`. Changing `ship_feature_scale` must **not** change gameplay.

### v1 migration

| v1 | v2 |
|----|-----|
| `task.max_ships` | `task.ship_feature_scale` |

While v1 and v2 run side-by-side:

- v1 encoder keeps reading `task.max_ships` (unchanged until v1 removal).
- v2 encoder reads `task.ship_feature_scale`.
- Hydra may expose both during transition, or a single alias that maps to both fields for matched ablations.

When documenting feature tables below, **`/ ship_feature_scale`** means divide by this config value using the formulas in the table footnotes.

---

## ADR-004: Symmetry — Learner-Centric Frame

### Decision

Canonicalize spatial features in a **sun-centered frame rotated by the learner reference angle** `θ_ref`:

```
θ_ref = atan2(cy - 50, cx - 50)
(cx, cy) = unweighted mean (x, y) of active planets owned by the learner
```

Same rule for 2p and 4p. See `docs/feature-encoding-v2-symmetry.md` for edge cases and decode.

### Encoding

| Where | Fields |
|-------|--------|
| **Planet** | `r` (sun distance / `BOARD_SIZE`), `θ` (canonical angle) — **no absolute x,y** |
| **Edge** | `delta_x`, `delta_y`, `distance` in learner frame (`/ BOARD_SIZE`) |
| **Global** | add `angular_velocity` (normalized) |
| **Decode** | `angle_abs = angle_canonical + θ_ref` for submission API |

Layers A (owner-relative) + B (frame) + C (edge-primary geometry) are **in scope** for v2 v1.

---

## Schema v2 (Draft — dims TBD Phase 0 lock)

### Planet features (P — target ~12–14)

Scale: `S = task.ship_feature_scale`. Frame: ADR-004 (`r`, `θ` learner-canonical).

| Field | Dim | Encoding |
|-------|-----|----------|
| active | 1 | 0/1 |
| orbit_radius | 1 | `hypot(x-50, y-50) / BOARD_SIZE` (sun-centered) |
| orbit_angle | 1 | `wrap(atan2(y-50, x-50) - θ_ref) / π` |
| radius | 1 | `/ 5.0` |
| ships | 1 | `min(ships, S) / S` |
| production | 1 | `/ MAX_PRODUCTION` |
| rotating_flag | 1 | 0/1 |
| owner_slot | 4 | relative one-hot (4p) |
| incoming_friendly_pressure | 1 | `pressure / S` |
| incoming_enemy_pressure | 1 | `pressure / S` |
| outgoing_friendly_ships | 1 | `outgoing / S` |
| ship_delta | 1 | `(ships - ships_prev) / S` |

### Edge features (E — target ~10–12)

Distances/deltas in **learner frame** (ADR-004), then scaled.

| Field | Dim | Encoding |
|-------|-----|----------|
| delta_x, delta_y | 2 | learner-frame `/ BOARD_SIZE` |
| distance | 1 | learner-frame `/ BOARD_SIZE` |
| sun_crossing | 1 | 0/1 |
| target_ships | 1 | `min(ships, S) / S` |
| target_production | 1 | `/ MAX_PRODUCTION` |
| target_owner_slot | 4 | relative one-hot |
| turns_to_arrival | 1 | `distance / MAX_FLEET_SPEED / MAX_STEPS` |
| target_incoming_friendly | 1 | `pressure / S` |
| target_incoming_enemy | 1 | `pressure / S` |
| target_ship_delta | 1 | `(ships - ships_prev) / S` |
| owner_changed | 1 | 0/1 |

### Global features (G — target ~45, evolved from v1 global-only)

Retain v1 global group semantics without duplicating into planet rows. Ship/fleet totals use `MAX_PLANETS * S` as denominator where v1 used `MAX_PLANETS * max_ships`.

- step_fraction, planet/ship/fleet fractions
- owner-relative counts/ships/fleets/production (4 slots)
- active_owner_mask, player_count
- **angular_velocity** (normalized; Kaggle obs field, useful for rotating planets)
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
  ship_feature_scale: <S>          # recorded for checkpoint/debug; must match task config
  edge_layout: top_k_per_source
  edge_k: <K>
  feature_history_steps: <H>
```

v1 checkpoints rejected when `encoding_version=v2` training loads v1 weights.

---

## Config (sketch)

```yaml
# conf/task/default.yaml (future v2 fields)
encoding_version: v1              # v1 | v2
ship_feature_scale: 1000.0        # v2 encoder normalization only (ADR-003)
# max_ships: 1000.0               # v1 only — alias during side-by-side or remove after v1 deprecation

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
| `task.max_ships` | `task.ship_feature_scale` (ADR-003; encoding only) |
| self_features per source | planet row (owned) |
| candidate_features per slot | edge_features[src, k] |
| global_features (broadcast) | global_features (once) |
| candidate slot index | flat edge index |
| NO_OP slot 0 | NO_OP logit |

---

## Phase 0 exit checklist

- [x] ADR-001 action space drafted
- [x] ADR-003 ship feature scale (`ship_feature_scale` vs fleet speed)
- [x] ADR-004 symmetry frame (`θ_ref` = sun → owned-planet centroid)
- [x] Schema draft tables
- [x] Submission audit
- [x] Cutover numeric gates documented
- [ ] **P, E, G dims locked** (after float budget spike)
- [ ] **v1 baseline metrics captured**
- [ ] **jax-ppo-split dependency cleared**
