# Feature Encoding v2 — Symmetry & Canonicalization

**Status:** Phase 0 contract (symmetry frame locked)  
**Parent:** `docs/feature-encoding-v2.md`, `docs/feature-encoding-v2-design.md`

## Motivation

Orbit Wars has **designed geometric symmetry** at reset (4-fold planet groups, diagonal/quad homes, symmetric comet paths on Kaggle), but v1 features are mostly in a **fixed absolute board frame** (`x/100`, `y/100`). Only **player identity** is canonicalized via `relative_owner_slot`.

v2 is a good time to canonicalize observations so the policy sees **learner-centric, symmetry-aware** inputs while still emitting **real planet IDs and absolute angles** at the action boundary.

---

## What is actually symmetric?

### Exact invariants (reset / rules)

| Invariant | Detail |
|-----------|--------|
| **Sun anchor** | Fixed at `BOARD_CENTER (50, 50)`, `SUN_RADIUS = 10` |
| **4-fold planet groups** | Each group: 4 planets related by board symmetries; shared prod/ships/radius |
| **2p homes** | Diagonal opposites (quadrants 0 and 3) in one group |
| **4p homes** | One home per quadrant in one group |
| **Shared rotation** | One `angular_velocity` per episode for all rotating planets |
| **Owner-relative slots** | `(owner - player) % player_count` — already in v1/v2 draft |
| **Kaggle comets** | 4 symmetric path copies (90° rotations) per spawn |

Tests: `test_reset_generates_fourfold_symmetric_planet_groups_for_two_players`, `test_four_player_reset_home_planets_are_rotationally_symmetric` in `tests/test_jax_env_parity.py`.

### Not symmetric (or breaks quickly)

| Element | Why |
|---------|-----|
| **Mid-game state** | Captures, fleets, production asymmetry break full-board symmetry |
| **Between-group RNG** | Each planet group has independent angle/orbit/stats |
| **Planet IDs** | Fixed generation order — used in action API |
| **Comet path params** | Random per spawn (symmetric copies, not identical board) |
| **JAX training env** | No comets; fixed 10 groups vs Kaggle variable groups |
| **CCW rotation only** | Reflection would flip rotation sense |

**Takeaway:** Exploit symmetry as **frame choice + inductive bias**, not as “the board is always symmetric mid-game.”

---

## What v1 already canonicalizes

| Mechanism | Preserves |
|-----------|-----------|
| `relative_owner_slot` / `relative_owner_slots` | Player label permutation |
| `target_ownership_flags` | Neutral / self / other (loses 4p opponent identity) |
| `owner_relative_*` aggregates | Self vs opponents in fixed 4 slots |
| Training `alternate_player_sides` | Which player index is the learner |

## What v1 breaks (spatial symmetry)

| Feature | Issue |
|---------|--------|
| `source_coords`, `target_coords` | Absolute board position |
| `delta_coords` | Translation-invariant from source, **not** rotation-invariant |
| `shot_crosses_sun`, `rotating_planet_flag` | Tied to fixed sun/center |
| Planet row order | Fixed `MAX_PLANETS` / ID order |
| GNN k-NN | Built from absolute candidate coords |
| `angular_velocity` | In Kaggle obs but **not** in v1 feature schema |

v2 draft (`feature-encoding-v2.md`) still uses absolute `x, y` on planets — same issue.

---

## Action / submission constraint (non-negotiable)

Kaggle API:

- **Observation:** shared absolute board (`planets`, `fleets`, `player`, …)
- **Action:** `[from_planet_id, direction_angle, num_ships]`

Canonicalization is **encoder-internal**. At act time:

1. Policy chooses in canonical space (edge index, bucket, …)
2. Decoder maps to **real `planet_id`** and **absolute angle**
3. Inverse transform if using learner-frame rotation

Submission and training must share the **same forward canonicalization** and **same inverse decode**.

---

## Canonicalization layers (composable)

These stack cleanly on the v2 planet + edge + joint-pointer design.

### Layer A — Player-relative ownership (keep)

Already planned: `owner_slot` on planets/edges, global owner-relative blocks.

**No change needed** beyond v2 draft; drop redundant 3-way ownership flags in 4p.

### Layer B — Learner-centric board frame (**locked**)

Transform coordinates into a **sun-centered frame rotated by a learner reference angle** `θ_ref`.

#### Reference angle (decision locked)

**`θ_ref` = angle from sun to the centroid of learner-owned planets** (unweighted mean of `(x, y)` over active planets with `owner == player`).

```
(cx, cy) = mean(x, y) over owned active planets
θ_ref = atan2(cy - BOARD_CENTER_y, cx - BOARD_CENTER_x)
```

Same rule for **2p and 4p** (no format-specific frame).

| Case | Behavior |
|------|----------|
| **≥1 owned planet** | Centroid as above; single planet reduces to “home-only” geometry |
| **0 owned planets** | No decision rows (encoder returns empty / masked batch); `θ_ref = 0` if global features still needed for terminal states |
| **Centroid at sun** | Degenerate (extremely unlikely); fallback `θ_ref = 0` |

**Why centroid over home planet:** home may be lost mid-game; centroid tracks where the learner’s **mass** sits on the board and stays stable under symmetric expansions better than a single fixed ID.

**Planet encoding (replaces absolute x,y):**

```
r = hypot(x - cx_sun, y - cy_sun) / BOARD_SIZE     # sun-centered radius
θ = wrap(atan2(y - cy_sun, x - cx_sun) - θ_ref)    # learner-canonical angle
```

**Edges:** `Δx, Δy`, distance, and launch angles computed in the same rotated frame (or equivalently from canonical polar deltas).

**Decode:** add `θ_ref` back to canonical angles before emitting Kaggle `[source_id, absolute_angle, ships]`.

**Sun crossing:** compute in absolute frame before or after rotate — boolean is invariant.

**Fleet / edge angles:** subtract `θ_ref` in canonical frame; add back at decode.

**Pros:** Robust mid-game; exploits board symmetry without per-format rules.  
**Cons:** Centroid shifts as captures accrue (intentional); comet paths on Kaggle need the same transform.

### Layer C — Edge-primary spatial signal (recommended)

Minimize absolute position on planet rows; put geometry on **edges**:

- `delta_r`, `delta_θ` or rotated `(Δx, Δy)` in learner frame
- `distance`, `sun_crossing`, `turns_to_arrival`

Planet rows focus on **local state**: ships, production, pressures, owner_slot, rotating flag.

**Pros:** Aligns with v2 edge tensor; translation-invariant; pairs naturally with Layer B.  
**Cons:** Rotation-invariant only together with Layer B.

### Layer D — Invariant planet ordering (optional, medium cost)

Sort planet tensor rows by canonical keys, e.g.:

```
(owner_slot, -ships, r, θ, planet_id)   # planet_id tie-break only
```

Keep **`planet_id` side channel** (or lookup table) for pointer → action decode.

**Pros:** Removes ID-order bias; helps GNN/pointer.  
**Cons:** JIT sort + remapping edges; must remap joint pointer indices every step.

### Layer E — Training augmentation (complement)

On early-game states (or synthetic resets), apply random **90° board rotation** + consistent owner relabel + angle shift; train policy to be equivariant.

**Pros:** No submission change if decoder is correct.  
**Cons:** Hard mid-game; best as add-on to B+C, not alone.

---

## Recommended v2 package

**Default proposal:** **A + B + C**, evaluate **D** in Phase 0 spike.

```
Global:  step, owner aggregates, angular_velocity, optional history stack
Planet:  r, θ (learner frame), ships, production, owner_slot, rotating,
         pressures, ship_delta  — no absolute x,y
Edge:    Δx, Δy (learner frame), distance, sun_crossing, target state,
         arrival, pressures, deltas
Side:    planet_id map for decode only (not policy input, or separate non-gradient channel)
```

**New global field:** `angular_velocity` (normalized) — in Kaggle obs, needed for orbit prediction.

---

## Interaction with joint pointer

| Topic | Implication |
|-------|-------------|
| **Joint (src, tgt) pointer** | Edge list should be built **after** canonical frame + optional sort |
| **Top-K edges** | Sort targets by **canonical distance**, not raw board x |
| **NO_OP** | Unchanged |
| **Shield** | Must use **absolute** geometry internally, or canonical↔absolute consistently |

---

## Symmetry strategies ranked (feasibility for v2)

| Rank | Strategy | Feasibility | Impact |
|------|----------|-------------|--------|
| 1 | Learner-frame sun polar (Layer B) | High | High |
| 2 | Edge-primary geometry (Layer C) | High | High |
| 3 | Invariant planet sort (Layer D) | Medium | Medium |
| 4 | Train-time rotation aug (Layer E) | Medium | Medium (train only) |
| 5 | Full D4 equivariant network | Low | High (research) |

---

## Phase 0 additions (proposed)

- [x] **ADR-004 Symmetry:** learner frame with **`θ_ref` = sun → owned-planet centroid** (2p/4p same rule)
- [ ] Planet row fields: sun-polar `r, θ` + edge-primary geometry (Layer C)
- [ ] Spike: encode symmetric reset; verify 90°/180° equivariance under known transforms
- [ ] Document decode: canonical angle + `θ_ref` → absolute submission angle
- [ ] Add `angular_velocity` to global schema

---

## Open questions

1. ~~**Reference angle:** home planet vs owned-centroid?~~ **Locked: owned-planet centroid (unweighted).**
2. **Planet sort (D):** worth JIT cost vs fixed ID order with polar coords only?
3. ~~**2p vs 4p:** same canonical frame rule?~~ **Locked: same rule.**
4. **Comets (Kaggle):** include in v2 encoder from day one, or JAX-only path first?
5. **Equivariance target:** canonical frame only vs + training augmentation vs architectural (future)?

---

## v1 → v2 mapping (symmetry-related)

| v1 | v2 canonical |
|----|----------------|
| `x, y` absolute | `r, θ` in learner frame |
| `delta_x, delta_y` board | `Δx, Δy` in learner frame (edges) |
| `relative_owner_slots` | `owner_slot` (keep) |
| Fixed planet row index | Optional sort (D) + `planet_id` lookup |
| (missing) | `angular_velocity` global |
