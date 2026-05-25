# Feature Encoding v2

**Status:** Phase 5 cutover complete — production default `encoding_version=v2`; v1 path retained for rollback/tests  
**Plan:** `.omg/plans/ralplan-feature-encoding-v2.md`  
**ADR:** ADR-001 (action space), ADR-002 (edge layout), ADR-003 (ship feature scale), ADR-004 (symmetry frame), **ADR-005 (factored pointer — planned)** below  
**Symmetry exploration:** `docs/feature-encoding-v2-symmetry.md`

## Overview

v2 replaces v1's self/candidate/global flat groups with:

- **Planet tensor** `(MAX_PLANETS, P)` — per-planet state
- **Edge list** — top-K `(source → target)` pairs per owned source with features `(E,)`
- **Global vector** `(G,)` — game-level aggregates + optional global history stack

**Action space:** joint pointer over legal edges `(source_planet, target_planet)` + ship bucket + NO_OP.

**Runtime:** JAX encoder canonical at cutover; v1 side-by-side until ablation passes gates.

---

## ADR-001: Action Space (Final)

### Decision

Joint pointer selects an **edge index** from a flat list of legal `(source, target)` pairs for the current decision step. Ship bucket is sampled **conditioned on the chosen edge**.

### Index space (static JIT layout)

```
K = max(0, candidate_count - 1)
E_flat = MAX_PLANETS * K

For src_row in 0 .. MAX_PLANETS-1, slot k in 0 .. K-1:
  flat_idx = src_row * K + k
  edge = (edge_src_ids[src_row], edge_tgt_ids[src_row, k])   when edge_mask[src_row, k]

logits: float32[E_flat + 1]
  logits[0 .. E_flat-1]  — edge candidates (masked illegal)
  logits[E_flat]           — NO_OP (always legal)
```

**Padding contract:** Non-owned source rows keep `edge_mask=False`; edge features zeroed. Policy softmax uses the legality mask before sampling.

**K-step launches (`max_moves_k > 1`):** Repeat pointer → ship bucket within the same env step. After each launch, decrement available ships on the chosen source; mask edges whose source has zero ships remaining. NO_OP remains legal each sub-step.

### Legality mask (action sampling)

| Rule | Mask |
|------|------|
| Source active and owned by learner | Required |
| Target active | Required |
| `src != tgt` | Required |
| Sun-crossing shot | **Masked** (align with JAX v1 training) |
| Trajectory shield | Applied at sample time per bucket |
| NO_OP | Always legal (`logits[E_flat]`) |

### Game API mapping (submission)

External format unchanged: `[source_planet_id, angle, ships]`.

Internal decode:

```
(src, tgt) = edges[flat_idx]
angle_abs = atan2(tgt.y - src.y, tgt.x - src.x)    # or canonical + θ_ref (ADR-004)
ships = bucket_to_count(source_ships, bucket)
```

Target planet is **angle-implied** at game API — must match training shield geometry.

### File touch list (Phase 3 scope)

| Area | Files |
|------|-------|
| Shield | `src/game/trajectory_shield.py` — edge-batch variant |
| Action builders | `src/opponents/jax_actions/builders.py` — `build_action_from_edge_batch` |
| Sampling | `src/opponents/jax_actions/sampling.py` |
| Rollout | `src/jax/rollout/types.py`, `collect.py`, `metrics.py` |
| PPO | `src/jax/ppo_update.py` — v2 transition flatten |
| Policy | `src/jax/policy.py` — joint edge decoder |
| Env | `src/jax/env.py` — encode dispatch |
| Submission | `scripts/validate_kaggle_docker_submission.py`, packager decode loops |
| Inference | `src/opponents/runtime.py` (if Python path retained for debug) |

---

## ADR-005: Factored Top-K Pointer + Stop Head (Planned — M1)

### Decision

Replace ADR-001 **joint flat** pointer with a **factorized** decoder per launch step:

1. **Source** — softmax over `MAX_PLANETS` (owned + ships mask)
2. **Target slot** — softmax over `K` slots **conditioned on chosen source row** (ADR-002 top-K preserved)
3. **Ship bucket** — softmax over buckets conditioned on `(source, slot)`
4. **Stop** — Bernoulli/logit per step; padding within fixed `max_moves_k` loop via `step_active_mask`

**Preserves ADR-002:** target candidates remain top-K per source; shield evaluates `(src_row, slot)` via `evaluate_edge_pair`.

**Amends ADR-001:** no flat `P×K+1` joint softmax; NO_OP slot removed from target head (stop head replaces trailing NO_OP semantics).

### Log-probability factorization

```
log π_step = log π_stop + active × (log π_src + log π_tgt_slot + log π_bucket)
active = 1 when stop=0 and step is before padding cutoff
```

### Checkpoint plane (no schema_version bump)

| Field | Values |
|-------|--------|
| `pointer_decoder` | `joint_flat` \| `factorized_topk` |
| `action_layout_version` | `1` = joint flat (ADR-001), `2` = factorized top-K |

Decoder weights are incompatible across values; load-time rejection mirrors `encoder_backbone`.

### Status

Phase 0 (contract + shield spike) in progress. Default runtime remains `joint_flat` until M1 Phase 4 ablation.

---

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

## Schema v2 (Locked — Phase 0)

**Dims:** P=13, E=18, G=46. **Evidence:** `docs/feature-encoding-v2-phase0-results.md`, M4 intercept-edge milestone (`intercept-edge-features`).

Edge geometry uses **two anchor fleet speeds** (`task.intercept_anchors`, default `[1.0, 6.0]`) encoding intercept-time target positions. Schema floor is **`schema_version=4`**.

### Planet features (P = 13)

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
| ship_delta | 1 | `(ships - ships_prev) / S` |

Note: P=13 omits planet-row `incoming_enemy_pressure` and `outgoing_friendly_ships` (edge tensor carries target pressures; outgoing is source-local and omitted at this dim budget).

### Edge features (E = 18)

Distances/deltas in **learner frame** (ADR-004), then scaled. Per-anchor intercept block (slow `s=1.0`, fast `s=6.0`) replaces snapshot `delta_*`, `distance`, and `turns_to_arrival`. `crosses_now` is the legality-aligned snapshot sun-crossing field; per-anchor `sun_cross_at_intercept_*` is predictive.

| Field | Dim | Encoding |
|-------|-----|----------|
| intercept_delta_coords_s1 | 2 | learner-frame intercept delta at slow anchor `/ BOARD_SIZE` |
| intercept_distance_s1 | 1 | magnitude at slow anchor |
| intercept_turns_s1 | 1 | `raw_distance / 1.0 / MAX_STEPS`, clipped `[0,1]` |
| sun_cross_at_intercept_s1 | 1 | future-aim sun crossing at slow anchor |
| intercept_delta_coords_s6 | 2 | same at fast anchor (`s=6.0`) |
| intercept_distance_s6 | 1 | magnitude at fast anchor |
| intercept_turns_s6 | 1 | `raw_distance / 6.0 / MAX_STEPS`, clipped `[0,1]` |
| sun_cross_at_intercept_s6 | 1 | future-aim sun crossing at fast anchor |
| crosses_now | 1 | snapshot-line sun crossing (legality-aligned) |
| target_ships | 1 | `min(ships, S) / S` |
| target_owner_slot | 4 | relative one-hot |
| target_incoming_friendly | 1 | `pressure / S` |
| target_incoming_enemy | 1 | `pressure / S` |

### Global features (G = 46)

Retain v1 global group semantics (45 dims) plus **`angular_velocity`** (normalized; Kaggle obs field). Ship/fleet totals use `MAX_PLANETS * S` as denominator where v1 used `MAX_PLANETS * max_ships`.

- step_fraction, planet/ship/fleet fractions
- owner-relative counts/ships/fleets/production (4 slots)
- active_owner_mask, player_count
- **angular_velocity** (1)
- delta slots (ships, planets, fleets, production) — 16 dims

### Dim budget (default H=1, K=3)

| Component | Floats |
|-----------|-------:|
| Planets (60×13) | 780 |
| Edges (60×3×18) | 3240 |
| Global | 46 |
| **Total** | **2986** |
| Pointer softmax | 181 |

---

## History

- **Planet:** `ship_delta`, `owner_changed` from prior step (single frame)
- **Global:** stack H frames when `feature_history_steps > 1`
- **Edges:** recompute each step; no frame stack (deltas capture target dynamics)

---

## Checkpoint metadata v2

```yaml
feature_metadata:
  schema_version: 4
  encoding_version: v2
  planet_feature_dim: <P>
  edge_feature_dim: <E>
  global_feature_dim: <G>
  ship_feature_scale: <S>          # recorded for checkpoint/debug; must match task config
  intercept_anchors: [1.0, 6.0]  # anchor fleet speeds for intercept edge block
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

- [x] ADR-001 action space finalized
- [x] ADR-002 edge layout (top-K) finalized
- [x] ADR-003 ship feature scale (`ship_feature_scale` vs fleet speed)
- [x] ADR-004 symmetry frame (`θ_ref` = sun → owned-planet centroid)
- [x] Schema draft tables → **P=13, E=18, G=46 locked** (schema v4 intercept edges)
- [x] Submission audit
- [x] Cutover numeric gates documented
- [x] Float budget spike (`scripts/spike_feature_encoding_v2_phase0.py`)
- [x] Equivariance spike (known transforms)
- [x] v1 baseline metrics captured (`docs/feature-encoding-v2-phase0-results.md`)
- [x] jax-ppo-split dependency cleared
- [x] Config sketch (`conf/model/gnn_pointer_v2.yaml`)
- [ ] Layer D planet sort — deferred

**Phase 0: COMPLETE** — see `docs/feature-encoding-v2-phase0-results.md`.
