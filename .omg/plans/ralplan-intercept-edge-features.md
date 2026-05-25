# Ralplan iter-3 (final): M4 — Bucket-aware Intercept Edge Features (descoped)

**Source spec:** `.omg/specs/deep-interview-intercept-edge-features.md`
**Slug:** `intercept-edge-features`
**Workflow:** deep-interview → ralplan → omg-autopilot
**Status:** approved (iter-3 mechanical fixup pass applied; no further consensus loop)
**Supersedes:** iter-2 of this plan (architect+critic items applied)

## Iter-3 Change Log

Architect approved-with-changes; critic approved-with-changes; both flagged the same set of mechanical issues. Applied in this revision:

1. Per-edge τ approach explicitly resolved (Phase 2 design note) — shape-polymorphic primitive in `feature_primitives.py`, per-edge gather of orbital params, no vmap.
2. Schema floor in `validate_checkpoint_feature_compatibility` (line 313) bumps from `< 3` to `< 4` (Phase 1).
3. Schema inferrer in `infer_feature_metadata_from_state_dict` (line 260) bumps from `schema_version: 3` to `4` (Phase 1).
4. Validator cutoff decision documented: under v4, schema_version < 4 means legacy v1/v2/v3, all rejected with "v3 → v4 migration required" (Phase 1).
5. `crosses_now` is an outright rename (no co-existence with `crosses`); schema bump already invalidates checkpoints.
6. Edge-mask gating note: `edge_mask = ordered_valid & (~crosses_now) & owned_source[:, None]` — legality-aligned field drives mask, not `sun_cross_at_intercept`.
7. Phase 3 fixture sweep extended: `tests/test_checkpoint_compat.py:26,49-55,90`; `tests/test_kaggle_submission_packager.py:182,198`; `scripts/spike_feature_encoding_v2_phase0.py:36`.
8. Phase 4 gains a baseline-variance characterization sub-step before the full A/B run.
9. ADR Alternatives now records the rejected `task.intercept_features_enabled` antithesis flag.

## Iter-2 RALPLAN-DR Summary

Iter-1 combined feature-encoding changes with thin-trajectory-shield changes in one milestone. Both architect and critic independently recommended decoupling, and the user accepted the synthesis path. **M4 in iter-2 is features-only.** `src/game/trajectory_shield.py` is read-only this milestone; the dynamic shield continues to enforce legality. Shield thinning is registered as the deferred `thin-trajectory-shield` follow-up milestone.

This collapses the original five-phase plan to four phases, drops the submission legality net, drops the throughput uplift gate (now non-regression only), and surfaces a clean set of refactors needed to land the encoder change correctly.

### Mode

**SHORT.** With shield work removed the only meaningful design risk is whether intercept geometry actually moves `episode_reward_mean`. Implementation is mechanical: two anchor passes through the existing `_edge_features` pipeline, schema bump, golden tests, ablation runbook.

## Principles

1. **Replace edge geometry; preserve the top-K shape contract.** ADR-002's top-K-per-source shape is untouched. Only the per-edge feature content changes; edge_feature_dim grows from 12 to 18.
2. **Reuse, do not re-implement, the orbit lookahead.** `_planet_positions_at_step_jax` already does closed-form orbital position at a fractional step in `src/game/trajectory_shield.py`. Re-export it via `src/jax/feature_primitives.py` rather than writing a parallel formula.
3. **Hydra coercion is a real cost surface.** `intercept_anchors` will arrive as `ListConfig`, not `tuple`. Normalize explicitly at every read site (`feature_metadata`, `_edge_features`, encoder) via `tuple(env_cfg.intercept_anchors)`. Don't trust dataclass annotations to coerce.
4. **Shield is the legality oracle. Features are the prior.** The encoder learns intercept-aware aiming from features; the unchanged dynamic shield still rejects illegal launches at sample time. They are independent surfaces.
5. **One source of truth for edge_feature_dim.** `EDGE_FEATURE_CATALOG.base_dim` is canonical. Every consumer (`feature_metadata`, `PlanetEdgeBackboneEncoder.edge_feature_dim`, tests, docs) derives from it. The `12 → 18` migration is a grep-and-update sweep.

## Decision Drivers

1. **The current `turns_to_arrival` is fleet-speed-blind.** It divides by `MAX_FLEET_SPEED=6.0` regardless of bucket, so the policy has no signal for slow-launch arrival into rotating targets. Two-anchor encoding fixes this without action-space or encoder-architecture changes.
2. **Coupling shield thinning to feature changes inflated risk dramatically.** Iter-1's critic identified feature/shield disagreement, static-path guard bypass, and legality-net duplication as critical failure modes. Decoupling moves those concerns entirely out of M4.
3. **Throughput uplift is no longer this milestone's job.** Without shield changes, M4 is expected to be roughly throughput-neutral. Any improvement is the deferred milestone's deliverable. Gate accordingly: non-regression only, ±5% tolerance for tracing overhead on the widened `_edge_features` signature.

## Viable Options per Open Question

### Q1. Exact replacement edge feature list

| Option | Per-edge block | Total E | Pros | Cons |
|---|---|---|---|---|
| **A — Recommended** | 5 fields × 2 anchors + 1 `crosses_now` + 7 retained snapshot fields | **18** | Symmetric per-anchor block; `crosses_now` keeps legality-aligned snapshot signal separate from predictive `sun_cross_at_intercept` | E grows by 50%; encoder param count rises modestly |
| **B** | Same as A but with shared snapshot `delta_coords/distance` instead of `crosses_now` | 19 | Even more redundancy for static planets | Static-planet anchors already collapse to snapshot; adding raw delta_coords is pure duplication |
| **C** | One anchor only (`s=6.0`) + bucket-fraction scalar | 13 | Tiny dim increase | Rejected in interview Round 2: forces policy to reconstruct slow-anchor geometry from scalar |

**Recommendation: A.** Locked field list:

| Group | Field | Size | Replaces / source |
|---|---|---:|---|
| Anchor `s=1.0` | `intercept_delta_coords_s1` | 2 | replaces `delta_coords` (snapshot) |
| Anchor `s=1.0` | `intercept_distance_s1` | 1 | replaces `distance` (snapshot) |
| Anchor `s=1.0` | `intercept_turns_s1` | 1 | replaces `turns_to_arrival` (snapshot, constant speed) |
| Anchor `s=1.0` | `sun_cross_at_intercept_s1` | 1 | new (predictive, per anchor) |
| Anchor `s=6.0` | `intercept_delta_coords_s6` | 2 | (per-anchor copy) |
| Anchor `s=6.0` | `intercept_distance_s6` | 1 | (per-anchor copy) |
| Anchor `s=6.0` | `intercept_turns_s6` | 1 | (per-anchor copy) |
| Anchor `s=6.0` | `sun_cross_at_intercept_s6` | 1 | (per-anchor copy) |
| Retained | `crosses_now` | 1 | renamed from existing `sun_crossing`; legality-aligned snapshot line, matches what the dynamic shield uses |
| Retained | `target_ships` | 1 | unchanged |
| Retained | `target_owner_slot` | 4 | unchanged |
| Retained | `target_incoming_friendly` | 1 | unchanged |
| Retained | `target_incoming_enemy` | 1 | unchanged |

**Total E = 10 + 8 = 18.**

Semantics:

- `intercept_delta_coords_{s1,s6}`: **learner-frame** target-future minus source-now, divided by `BOARD_SIZE`. Mirrors the current `delta_coords` convention exactly.
- `intercept_distance_{s1,s6}`: magnitude of the rotated normalized delta, i.e. `sqrt(dx² + dy²)` post-normalization.
- `intercept_turns_{s1,s6}`: `(distance_in_raw_coords / s) / MAX_STEPS`, mirroring the current `turns_to_arrival` normalization.
- `sun_cross_at_intercept_{s1,s6}`: `shot_crosses_sun_xy(src_now_raw, target_future_raw)`, evaluated on the **future** aim line (predictive).
- `crosses_now`: `shot_crosses_sun_xy(src_now_raw, target_now_raw)`, evaluated on the **current** aim line (legality-aligned, matches what `apply_trajectory_shield_to_turn_batch_v2` checks for sun blocking).

### Q2. `tgt_ships` projection

Locked by spec: **keep as snapshot**. Add a `# TODO(M5): forward-projected target ships per anchor` comment near the `target_ships` entry in `src/features/catalog/edge.py`.

### Q3. Top-K sort order under ADR-002

| Option | Strategy | Pros | Cons |
|---|---|---|---|
| **A — Recommended** | Keep snapshot-distance sort. Add explicit `# TODO(thin-trajectory-shield-follow-up)` near `lexsort` in `src/jax/features.py:186-189` documenting the inconsistency. | Golden-test stable; small diff surface; defers a separable change | Top-K can drop intercept-favorable targets in the long-anchor regime |
| **B** | Sort by `min(intercept_distance_s1, intercept_distance_s6)` | Picks edges the policy actually wants | Reorders top-K vs current behavior; golden tests churn; ADR-002 amendment needed |
| **C** | Multi-key sort: snapshot distance primary, mean intercept distance tiebreaker | Backward-compatible for static planets | Tiebreaker only fires on exact ties; little observable difference |

**Recommendation: A.** ADR-002 does not pin "snapshot distance" as the sort key, so this is technically a free choice — but golden stability is the dominant cost driver for M4. Phase 2 commit message must reference the TODO so the follow-up milestone owns the change.

### Q4. Throughput measurement methodology

| Option | Tool | Pros | Cons |
|---|---|---|---|
| **A — Recommended** | `scripts/benchmark_jax_rl.py --warmup 2 --updates 20` × 3 reps, fixed seed, model=gnn_pointer, both formats. Pinned baseline commit and M4 commit. Compare median `env_steps_per_sec`. | Already wired; matches the rollout loop M4 actually uses | Single-machine results |
| **B** | Custom microbench around `_edge_features` only | Isolates encoder cost | Misses rollout integration; not the spec gate |

**Recommendation: A** with **±5% non-regression tolerance** (not uplift). The widened `_edge_features` signature adds two scalar tensors to the JIT trace; a small overhead is plausible and acceptable.

### Q5. Non-rotating planet handling

The reused `_planet_positions_at_step_jax` already returns current position when its `rotates` flag is false (driven by `ROTATION_RADIUS_LIMIT` and `active`). For non-rotating planets the intercept block collapses to snapshot geometry at both anchors, which is the desired behavior.

Add a unit test asserting `intercept_delta_coords_s1 == intercept_delta_coords_s6 == snapshot_delta_coords` for a synthetic non-rotating planet, modulo the `BOARD_SIZE` normalization.

### Q6. A/B vs A/B/A for the win gate

| Option | Strategy | Pros | Cons |
|---|---|---|---|
| **A — Recommended** | Paired A/B at 3 seeds × 500 updates × 2 formats. Use same seeds for both arms; report paired-seed deltas. If lift is borderline (1.5–2.5%), escalate to **6 seeds** rather than re-running baseline. | Cheapest path to a credible directional read; paired seeds remove between-seed noise; if borderline, more seeds give better statistical power per compute unit than a third arm | 3 seeds is moderate statistical power |
| **B** | A/B/A: baseline → M4 → baseline-replica at 3 seeds each | Detects drift / transient effects | Triples baseline cost; drift across `main` HEAD between A and A' is the actual risk and is better controlled by pinning commits |
| **C** | Single-seed A/B at 1 seed × 1000 updates | Cheapest possible | No statistical claim |

**Recommendation: A** (paired A/B). At observed throughput (~1300 env_steps_per_sec on 32-env mix), one 500-update run is ~13 minutes; 3 seeds × 2 formats × 2 arms ≈ 2.6 hours of compute. Borderline-result escalation to 6 seeds adds ≈1.3 hours. A/B/A buys little additional confidence at this scale because both A and A' would run on the same pinned commit anyway.

## Phased Implementation Plan

```mermaid
flowchart LR
    P1[Phase 1: Catalog + schema + dataclass plumbing] --> P2[Phase 2: Encoder integration + goldens]
    P2 --> P3[Phase 3: Docs + fixture sweep]
    P3 --> P4[Phase 4: Ablation runbook + reward gate]
```

### Phase 1 — Catalog, schema bump, dataclass plumbing (1–2 days)

Addresses architect items **A5 (Hydra tuple coercion)** and **A6 (checkpoint metadata parser)** and the catalog redesign.

**Files touched**

- `src/config/schema.py`
  - Add `intercept_anchors: tuple[float, float] = (1.0, 6.0)` to `TaskConfig`.
  - Do NOT add the `trajectory_shield_static_only` flag (it belongs to the deferred milestone).
- `conf/task/default.yaml` — add `intercept_anchors: [1.0, 6.0]` (Hydra YAML idiom).
- `src/features/catalog/_types.py`
  - Extend `EdgeRowAssemblyContext` with per-anchor tensors: `intercept_delta_x_per_anchor`, `intercept_delta_y_per_anchor`, `intercept_distance_per_anchor`, `intercept_turns_per_anchor`, `sun_cross_at_intercept_per_anchor`, each shaped `(P, K, num_anchors)`.
  - Add `crosses_now: jax.Array` (replaces existing `crosses` semantically).
  - Drop now-replaced raw fields: `delta_x`, `delta_y`, `distance`, `turns`, `crosses`.
- `src/features/catalog/edge.py`
  - Define new entries in the order shown in Q1's table. Per-anchor compute functions destructure the anchor index from the context tensor: `def _feat_intercept_distance_s1(ctx): return ctx.intercept_distance_per_anchor[..., 0:1]`.
  - Drop the old `delta_coords`, `distance`, `sun_crossing`, `turns_to_arrival` entries.
- `src/artifacts/checkpoint_compat.py`
  - Bump `schema_version` to `4` in `feature_metadata()`.
  - Add `"intercept_anchors": tuple(map(float, env_cfg.intercept_anchors))` (A5: explicit tuple coercion at write time).
  - Extend `METADATA_KEYS` with `"intercept_anchors"`.
  - Patch `checkpoint_feature_metadata` parser (currently lines 167–191): the default `else: parsed[key] = int(value)` branch will raise on a tuple. Add an explicit branch:

    ```python
    elif key == "intercept_anchors":
        parsed[key] = tuple(float(v) for v in value)
    ```

    (A6 fix.)
  - In `validate_checkpoint_feature_compatibility`, add `intercept_anchors` to the mismatch comparison using element-wise float equality with tolerance `1e-6`.
  - **Bump schema floor (A2)**: change `validate_checkpoint_feature_compatibility` at line 313 from `if stored_schema is not None and int(stored_schema) < 3` to `< 4`. Update the error message from `"Schema v3 (single-source feature catalog) is required"` to `"Schema v4 (intercept-anchor edge features) is required; v3 → v4 migration required — retrain or run an explicit conversion."`. The string `"schema_version=2"` in the existing v2-rejection test (`tests/test_checkpoint_compat.py:55`) now matches v3 the same way; the rename test parameter to `schema_version=3` so it covers the new cutoff.
  - **Bump inferrer (C5)**: change `infer_feature_metadata_from_state_dict` at line 260 from `return {"schema_version": 3, ...}` to `4`. The function infers metadata from raw policy weight shapes; under v4 the only inferable metadata is still `(planet_feature_dim, edge_feature_dim, global_feature_dim)`, and labeling them `schema_version=4` lets the validator treat them as current-schema rather than rejecting on a stale floor.
  - **Validator cutoff decision (C6)**: under v4, any checkpoint with `schema_version < 4` is rejected. This subsumes the v1 legacy check at lines 267–274 (which only fires on the v1-specific shape markers like `self_feature_dim`); the `< 4` floor at line 313 catches v2 and v3 explicitly. Document this in a module-level docstring update at the top of `checkpoint_compat.py`: "Floor schema_version is 4 (intercept-anchor edges). Earlier versions are not loadable; migrate by retraining."
- `tests/test_checkpoint_compat.py` — Phase 1 touches the in-Phase 1 surface only; the broader fixture sweep is Phase 3. Concretely in Phase 1:
  - Line 26: `metadata["schema_version"] == 3` → `== 4`.
  - Line 28: `metadata["edge_feature_dim"] == 12` → `== 18`.
  - Lines 49–55 (`test_validate_rejects_v2_schema_version`): repurpose to `test_validate_rejects_v3_schema_version`; set `stored["schema_version"] = 3`; update regex to `"schema_version=3"`. This exercises the new floor.
  - Line 90 area: add a positive case asserting `intercept_anchors == (1.0, 6.0)`; add a mismatch case where stored `intercept_anchors == (1.0, 4.0)` triggers `validate_checkpoint_feature_compatibility` to raise.
- `tests/test_feature_catalog_drift.py`, `tests/test_feature_registry.py` — pin new dim and ordered field list.

**Exit criteria**

- `EDGE_FEATURE_CATALOG.base_dim == 18`.
- `feature_metadata(task_cfg)["schema_version"] == 4` and contains `intercept_anchors == (1.0, 6.0)` as a tuple.
- `feature_metadata` round-trips through `checkpoint_feature_metadata` without raising.
- `validate_checkpoint_feature_compatibility` rejects any checkpoint with `schema_version < 4`, including a stored v3 checkpoint.
- `infer_feature_metadata_from_state_dict` returns `schema_version: 4` (not 3) when inferring from weight shapes.
- `make test-domain-features` and `make test-domain-config` pass.
- `grep -rn "edge_feature_dim == 12\|E=12\|E = 12" src/ tests/ docs/` returns no untouched matches (docs follow in Phase 3).

**Estimated effort:** 1–2 days.

### Phase 2 — Encoder integration & golden tests (2 days)

Addresses architect items **A4 (widen `_edge_features` signature)**, **A7 (top-K sort TODO)**, and the per-edge τ question (architect item #1 vs critic item #10).

#### Design note: per-edge τ approach

Architect and critic disagreed on the per-edge τ implementation. Architect proposed a sibling function (`planet_positions_at_per_planet_step_jax`) plus a per-edge `jnp.take`. Critic claimed broadcasting handles per-edge τ natively without vmap.

**Reconciliation: both are partially right; we adopt a third path.** The orbital position formula is element-wise once you have per-edge orbital constants. Broadcasting works **only after** per-edge gather of `(initial_x, initial_y, current_x, current_y, radius, active)` for the target planet of each edge — at that point `start_angle`, `orbit_radius`, `rotates`, and `τ` all have shape `(P, K)` and the formula becomes a pure element-wise expression. No vmap, no `(P, K, P)` intermediate.

The existing `_planet_positions_at_step_jax` in `src/game/trajectory_shield.py` is **planet-shaped, not edge-shaped**: it takes a scalar (or `(P,)`) `step_index` and operates over all `MAX_PLANETS`. Calling it for per-edge τ would either require vmap (architect's concern about an `(P, K, P)` intermediate) or wouldn't broadcast cleanly (because source dim ≠ slot dim).

**Resolution:** extract the underlying formula into a shape-polymorphic primitive in `src/jax/feature_primitives.py` and call it from `_edge_features` with edge-shaped inputs. The trajectory-shield's `_planet_positions_at_step_jax` remains untouched in M4 (read-only constraint); the follow-up milestone will refactor it to call the same primitive.

```python
# src/jax/feature_primitives.py
def orbital_position_at_step_jax(
    start_angle, orbit_radius, angular_velocity, step_index,
    rotates, static_x, static_y,
):
    """Closed-form orbital position at a (fractional) step offset.

    Shape-polymorphic: all leading dims broadcast normally. For the planet-shaped
    callers in trajectory_shield (follow-up milestone), inputs are (MAX_PLANETS,);
    for per-edge use, callers gather per-edge constants first and feed (P, K)-shaped
    inputs. No vmap required.
    """
    angle = start_angle + angular_velocity * step_index.astype(jnp.float32)
    x = jnp.where(rotates, BOARD_CENTER[0] + orbit_radius * jnp.cos(angle), static_x)
    y = jnp.where(rotates, BOARD_CENTER[1] + orbit_radius * jnp.sin(angle), static_y)
    return x, y
```

The trajectory-shield's `_planet_positions_at_step_jax` is **not modified** in M4 (read-only milestone constraint). It continues to inline the same formula. The follow-up `thin-trajectory-shield` milestone will refactor it to call `orbital_position_at_step_jax` and complete the deduplication.

#### Design note: `crosses_now` rename

The existing edge field is named `crosses` (computed via `shot_crosses_sun_xy(src_now, target_now)`). M4 renames it to `crosses_now` to disambiguate from the new per-anchor `sun_cross_at_intercept_{s1,s6}` (computed against target-future positions). Because the schema v3 → v4 bump already invalidates existing checkpoints, **the rename is outright**: no co-existence, no transitional aliases. Catalog entry name, slice key in `EDGE_FEATURE_SCHEMA`, and the variable in `_edge_features` all use the new name from day one.

#### Design note: edge-mask legality gating

`src/jax/features.py:238` currently reads:

```python
edge_mask = ordered_valid & (~crosses) & owned_source[:, None]
```

After the rename this becomes:

```python
edge_mask = ordered_valid & (~crosses_now) & owned_source[:, None]
```

This is semantically identical to today. **The legality-aligned `crosses_now` drives the edge mask, not `sun_cross_at_intercept_{s1,s6}`.** The per-anchor predictive sun-crossing fields are pure feature signal — they expose information to the policy without altering legality. This separation is intentional: the dynamic shield (unchanged this milestone) is the legality oracle, and `crosses_now` mirrors the snapshot-line check the shield performs.

**Files touched**

- `src/jax/feature_primitives.py`
  - Add `orbital_position_at_step_jax(start_angle, orbit_radius, angular_velocity, step_index, rotates, static_x, static_y)` as described above. Shape-polymorphic; works for both planet-shaped (follow-up milestone) and edge-shaped (this milestone) callers.
  - No edits to `src/game/trajectory_shield.py` (read-only constraint). The follow-up milestone refactors `_planet_positions_at_step_jax` to call the primitive.
- `src/jax/features.py`
  - Widen `_edge_features` signature: `def _edge_features(game, env_cfg, scale, theta_ref_value)` — pass the whole game state so `game.step` and `game.angular_velocity` are available (A4). Update the `encode_turn` call site at line 69 to match (delete `planets`, `fleets`, `player` positional args; derive from `game`).
  - Per-edge gather of target orbital constants happens **once**, outside the anchor loop:
    ```python
    tgt_ids = order  # the existing top-K ordering result, shape (P, K)
    tgt_initial_x = jnp.take(game.initial_planets.x, tgt_ids, axis=0)  # (P, K)
    tgt_initial_y = jnp.take(game.initial_planets.y, tgt_ids, axis=0)
    tgt_radius_per_edge = jnp.take(planets.radius, tgt_ids, axis=0)
    tgt_active_per_edge = jnp.take(planets.active, tgt_ids, axis=0)
    init_dx = tgt_initial_x - BOARD_CENTER[0]
    init_dy = tgt_initial_y - BOARD_CENTER[1]
    orbit_radius = jnp.sqrt(init_dx * init_dx + init_dy * init_dy)
    rotates = (orbit_radius + tgt_radius_per_edge < ROTATION_RADIUS_LIMIT) & tgt_active_per_edge
    start_angle = jnp.arctan2(init_dy, init_dx)  # (P, K)
    ```
  - For each anchor `s ∈ tuple(env_cfg.intercept_anchors)`:
    1. `tau = jnp.maximum(snapshot_distance_raw / s, 0.0)` — shape `(P, K)`.
    2. `step_index = game.step.astype(jnp.float32) + tau` — shape `(P, K)`.
    3. `tgt_future_x, tgt_future_y = orbital_position_at_step_jax(start_angle, orbit_radius, game.angular_velocity, step_index, rotates, tgt_x, tgt_y)` — all inputs broadcast cleanly; no vmap.
    4. Apply `rotate_to_learner_frame` to both source-now and target-future raw coords, then divide by `BOARD_SIZE` for `intercept_delta_coords_{s1,s6}`.
    5. `intercept_distance` = magnitude of the rotated normalized delta.
    6. `intercept_turns` = `(distance_raw / s) / MAX_STEPS`, clipped to `[0, 1]`.
    7. `sun_cross_at_intercept` = `shot_crosses_sun_xy(src_now_raw, tgt_future_raw)` with the launch angle derived from the future-aim line.
  - Compute `crosses_now` exactly as the existing `crosses` is computed today (current-aim sun crossing). **Outright rename**: variable, catalog entry, and `EDGE_FEATURE_SCHEMA` slice name all use `crosses_now`.
  - Edge-mask line stays semantically identical, with the renamed field:
    ```python
    edge_mask = ordered_valid & (~crosses_now) & owned_source[:, None]
    ```
  - **Top-K selection stays on snapshot distance + snapshot sun crossing.** Add a comment immediately above the existing `lexsort` (currently `src/jax/features.py:186-189`):

    ```python
    # TODO(thin-trajectory-shield-follow-up): re-rank top-K by intercept proximity.
    # Snapshot sort retained for M4 golden-test stability; ADR-002 amendment will
    # accompany the re-rank when the follow-up milestone lands.
    ```

    (A7 fix.) Commit message must reference this TODO.
- `src/jax/policy.py:PlanetEdgeBackboneEncoder` — change `edge_feature_dim: int = 12` default to `18`, and ensure `build_gnn_pointer_policy` reads it from the catalog rather than hard-coding.
- `tests/test_feature_encoding_golden.py` — add four golden cases:
  1. Static (non-rotating) planet: `intercept_delta_coords_s1 == intercept_delta_coords_s6 == snapshot delta` modulo normalization (Q5).
  2. Rotating planet with non-zero `angular_velocity`: `intercept_distance_s1 > intercept_distance_s6` direction is expected when target is moving away in the source's frame; pin numeric values.
  3. Sun-cross flip: a planet whose orbit takes it across the sun line — `sun_cross_at_intercept_s1` differs from `crosses_now`.
  4. `tau` clipping: very far target so `tau_raw / MAX_STEPS > 1`; assert clipped to 1.
- `tests/test_jax_env.py`, `tests/test_jax_env_dispatch.py`, `tests/test_jax_policy_gnn.py` — already derive expected dim from `edge_feature_dim(cfg)`, so shapes flip automatically once Phase 1 lands. Run as a smoke.

**Exit criteria**

- `encode_turn` produces 18-dim edges; golden values committed.
- `make test-domain-features` passes.
- `uv run python -m src.train print_resolved_config=true` shows `intercept_anchors=[1.0, 6.0]` and resolved `edge_feature_dim=18`.
- `tests/test_jax_policy_gnn.py` runs (CPU-only path; no rollout).
- The TODO comment is present at `src/jax/features.py:lexsort` site and referenced in the commit message.

**Estimated effort:** 2 days (math + goldens).

### Phase 3 — Docs and fixture sweep (½ day)

Addresses critic's **docs drift** and **test-fixture pinning** items (C7).

**Files touched — docs**

- `docs/feature-encoding-v2.md` — update line 196 (`E=12` → `E=18`), line 217 section header (`Edge features (E = 12)` → `Edge features (E = 18)`), line 266 (schema example `schema_version: 2` → `schema_version: 4`, add `intercept_anchors: [1.0, 6.0]`), line 338 checklist (`P=13, E=12, G=46 locked` → `P=13, E=18, G=46 locked`).
- `docs/feature-encoding-v2-design.md` — update line 42 (`P=13, E=12, G=46` → `P=13, E=18, G=46`).
- `docs/feature-encoding-v2-pointer.md` — re-grep for `E=12` / `edge` references after Phase 1; update any stale references.
- `docs/adding-observation-features.md` — update line 81 (`schema_version=3` → `schema_version=4`).

**Files touched — test fixtures and scripts**

- `tests/test_checkpoint_compat.py` — Phase 1 already updated the lines exercised by Phase 1 logic. Phase 3 re-verifies the complete sweep, in particular:
  - Line 26: `schema_version == 3` → `== 4`.
  - Lines 49–55: `test_validate_rejects_v2_schema_version` renamed to `_rejects_v3_schema_version` with `stored["schema_version"] = 3` and regex `"schema_version=3"`.
  - Line 90: `inferred["schema_version"]` (if asserted) → `4`.
  - The dim-mismatch test continues to work because it mutates `planet_feature_dim`; if it incidentally asserted `12`, update to `18`.
- `tests/test_kaggle_submission_packager.py`:
  - Line 182: `"schema_version": 3` → `4`.
  - Line 184: `"edge_feature_dim": 12` → `18`.
  - Line 198: `artifact["feature_metadata"]["schema_version"] == 3` → `== 4`.
- `scripts/spike_feature_encoding_v2_phase0.py`:
  - Line 36: `EDGE_FEATURE_DIM = 12` → `18`. The spike script is a Phase 0 budget tool; the constant must match canonical catalog dim or the budget output is misleading.

**Exit criteria**

- `grep -rn "E ?= ?12\|edge_feature_dim ?== ?12\|edge_feature_dim ?= ?12\|EDGE_FEATURE_DIM ?= ?12" docs/ tests/ scripts/ src/` returns zero matches.
- `grep -rn "schema_version ?== ?3\|schema_version=3\|schema_version: 3\|\"schema_version\": 3" docs/ tests/ scripts/ src/` returns zero matches.
- `make test-fast` passes.

**Estimated effort:** ½ day.

### Phase 4 — Ablation runbook & reward gate (3–5 days; mostly wall-clock)

**Procedure**

1. **Pin baseline commit.** Use the current `main` head before any Phase 1 changes land. Capture baseline metadata in `artifacts/m4/baseline_pin.json` (commit SHA, throughput baseline, config hash).
2. **Encoder reset cost is unavoidable.** Schema v3 → v4 invalidates existing `gnn_pointer` checkpoints. Baseline must be retrained from scratch on the pinned commit. At observed throughput (~1300 env_steps_per_sec on 32-env mix configs, per `docs/feature-encoding-v2-phase0-results.md`), one 500-update training run is ≈13 minutes wall time.
3. **Baseline variance characterization (C9 — runs before the full A/B).** The `≥2% episode_reward_mean lift` gate currently lacks an empirical variance basis; the existing phase0 results doc only has a 25-update smoke. Before launching the paired A/B:
   - Run **1 seed × 100 updates × 3 RNG reps** on the pinned baseline commit (`model=gnn_pointer format=mix_2p_4p_8env`, different seeds 11/13/17, otherwise identical config to the planned A/B).
   - Compute sample standard deviation σ of `episode_reward_mean` at update 100 across the 3 reps.
   - Project the standard error of the 3-seed × 500-update mean delta as `SE ≈ σ × sqrt(2/3) × (variance_at_500 / variance_at_100)` (latter factor conservatively estimated as 0.6 based on RL convergence — re-tune if the 25-update smoke run had any longer arms).
   - **Decision rule**: if estimated `2 × SE` on the per-seed delta exceeds the 2% lift threshold (i.e., the gate is statistically indistinguishable from noise at N=3), escalate to **6 seeds for the full A/B** before running it. Document σ, projected SE, and the decision in `artifacts/m4/variance_characterization.json`.
   - **Cost**: 3 reps × 100 updates ≈ 8 minutes. Trivial relative to the full A/B budget.
4. **Compute budget.**

   | Run set | Seeds | Formats | Updates | Per-run wall | Subtotal |
   |---|---:|---:|---:|---:|---:|
   | Variance characterization (step 3) | 3 reps | 1 (mix) | 100 | ~3 min | ~9 min |
   | Baseline arm | 3 | 2 (2p, 4p) | 500 | ~13 min | ~1.3 h |
   | M4 arm | 3 | 2 | 500 | ~13 min | ~1.3 h |
   | Throughput benchmark | 3 reps × 2 commits × 2 formats | — | 20 | ~1 min | ~24 min |
   | Borderline escalation (6 seeds) | +3 | 2 | 500 | ~13 min | ~1.3 h |
   | **Default budget (no escalation)** | | | | | **~3 hours** |

5. **M4 training**: 3 seeds × 500 updates × 2 formats with `task.intercept_anchors=[1.0, 6.0]`. Use the same seeds as baseline (paired comparison). If step 3 escalated to 6 seeds, use 6 paired seeds throughout.
6. **Throughput**: `uv run python scripts/benchmark_jax_rl.py --overrides model=gnn_pointer task.player_count=2 --warmup 2 --updates 20` × 3 reps × 2 formats × 2 commits. Record median `env_steps_per_sec`.
7. **Gate evaluation.**
   - **H1 (submission validity, unchanged behavior)**: validate the M4 checkpoint with `scripts/validate_kaggle_docker_submission.py --player-count both --episode-steps 500` for 100 episodes per format. Expected to pass without changes since the dynamic shield is unchanged. Required: zero rejections.
   - **H2 (throughput non-regression, ±5%)**: `median(m4_env_steps_per_sec) ≥ 0.95 × median(baseline_env_steps_per_sec)` for both formats.
   - **H3 (training stability)**: no NaN/inf in `total_loss`, no entropy collapse (entropy floor > 0.1 of starting entropy by update 500).
   - **W1 (reward win gate)**: `mean(m4_episode_reward_mean over updates 450–500) − mean(baseline_episode_reward_mean over updates 450–500) ≥ 0.02 × |baseline|` for both 2p and 4p, paired across seeds.
   - **Borderline rule**: if W1 falls in `[1.5%, 2.5%]`, run 3 additional seeds (escalation to N=6) on both arms before declaring pass/fail. If step 3 already escalated upfront, this rule is satisfied automatically.
8. **Report**: write `docs/m4-intercept-edge-results.md` with the gate table, per-seed paired deltas, throughput numbers, and the variance characterization (σ, projected SE, escalation decision).

**Exit criteria**

- H1, H2, H3 all pass.
- W1 result documented (pass, fail, or borderline-with-escalation outcome).
- `docs/m4-intercept-edge-results.md` committed.
- If H2 fails: investigate Phase 2's `_edge_features` widening; if tracing overhead is the cause, profile and optimize. If H1 or H3 fail: revert Phase 2 (Phase 1 catalog/schema is keep-only).

**Estimated effort:** 3–5 days mostly wall-clock and analysis. **Requires user approval** to run the full training suite per AGENTS.md (`make test` is not used here; training is `uv run python -m src.train`).

## Test Strategy

### Unit tests (CPU-only, fast tier)

| Test file | Action | Coverage |
|---|---|---|
| `tests/test_feature_catalog_drift.py` | update | Catalog dim 18, ordered field list with new anchor block + `crosses_now` |
| `tests/test_feature_registry.py` | update | `edge_feature_dim == 18`, slice lookup for new feature names |
| `tests/test_feature_encoding_golden.py` | add 4 cases | Static planet anchor equality (Q5), rotating planet anchor divergence, sun-cross flip, `tau` clipping |
| `tests/test_checkpoint_compat.py` | update fixtures + add cases | `schema_version=4`, `intercept_anchors` round-trip, tuple coercion in parser (A6), mismatch raises |
| `tests/test_kaggle_submission_packager.py` | update fixture | `edge_feature_dim=18` (line 184) |
| `tests/test_jax_env.py`, `tests/test_jax_env_dispatch.py`, `tests/test_jax_policy_gnn.py` | smoke | Already derive dim from `edge_feature_dim(cfg)`; verify shapes flow through |

All CPU-only and must run inside `make test-fast`.

### Integration / slow-tier tests

- `tests/test_jax_rollout.py`, `tests/test_jax_ppo.py`, `tests/test_jax_curriculum.py` — slow tier; user-approved `make test` only.
- `tests/test_trajectory_shield.py` — **unchanged**. The dynamic shield is untouched; existing tests must continue to pass without modification. This is a regression signal.

Per AGENTS.md: agents do **not** run slow-tier tests during normal iteration. Phase 4 user runs `make test` once with explicit approval before training gates.

### Submission validation

```bash
uv run python scripts/validate_kaggle_docker_submission.py \
  --checkpoint artifacts/m4/checkpoints/seed0_final.pkl \
  --player-count both \
  --episode-steps 500 \
  --timeout-seconds 1.0
```

Acceptance: zero rejections (expected free, since the dynamic shield is unchanged).

## Risk Register

| # | Risk | Likelihood | Impact | Mitigation | Invalidates which gate? |
|---|---|---|---|---|---|
| R1 | Top-K snapshot sort drops edges the policy would have picked given intercept geometry, capping the reward lift. | Medium | Medium | Add the explicit TODO at the lexsort site; the deferred milestone owns the re-rank. If W1 fails directionally, this is the first hypothesis to test by trying option B (intercept-sorted top-K) as a one-off in a branch. | W1 only. |
| R2 | Encoder reset cost: schema v4 invalidates existing checkpoints, forcing a paired baseline retrain. | Certain | Low | Acknowledged in Phase 4 budget. At ~13 min per run, the cost is manageable. Document the pinned baseline commit so reruns are reproducible. | None (cost only). |
| R3 | Hydra coerces `intercept_anchors` to `ListConfig`, breaking equality checks or tuple-only call sites. | High | Medium | Normalize at every read with `tuple(env_cfg.intercept_anchors)` (A5). Add a regression test in `test_checkpoint_compat.py` that constructs `TaskConfig` via Hydra composition (not direct dataclass) and verifies `feature_metadata` round-trips. | Phase 1 acceptance. |
| R4 | Docs drift: E=12 / schema v3 references survive in `docs/feature-encoding-v2*.md` and confuse future readers / agents. | Medium | Low | Phase 3 grep sweep with explicit grep commands in exit criteria. | None (clarity only). |
| R5 | Test fixtures pinning E=12 (especially `tests/test_kaggle_submission_packager.py:183-184` and `tests/test_checkpoint_compat.py:28,92`) break `make test-fast` after Phase 1. | High | Low | Phase 1 itself updates these. Phase 3 re-sweeps as a safety net. | Phase 1 acceptance. |
| R6 | Per-edge τ vmap over `planet_positions_at_step_jax` introduces meaningful JIT trace cost, regressing H2. | Medium | Medium | Use Option α (vmap over per-edge τ with `jnp.take` gather) which keeps the work inside one trace. Benchmark in Phase 4. If H2 fails specifically here, fall back to inlining the formula in `_edge_features` while keeping `planet_positions_at_step_jax` as the canonical name — accepting modest duplication. | H2. |
| R7 | The `_planet_positions_at_step_jax` underscore-prefixed import looks like a private-API violation. | Low | Low | Phase 2 re-exports under a public name (`planet_positions_at_step_jax`) in `feature_primitives.py`. The follow-up milestone physically relocates the implementation, at which point the underscore name disappears. | None (code-hygiene only). |
| R8 | The intercept lift is real but small (<2%); W1 falls in the borderline band. | Medium | Medium | Phase 4 borderline rule: escalate to 6 seeds rather than retreating. The 6-seed escalation costs another ~1.3 hours, well inside budget. | W1 only. |

## ADR (Architecture Decision Record)

**Decision (iter-2):** Replace `TurnBatch` edge geometry with intercept geometry at two anchor fleet speeds (`s ∈ {1.0, 6.0}`), splitting the current `crosses` field into legality-aligned `crosses_now` and predictive `sun_cross_at_intercept` per anchor. Edge feature dim grows from 12 to 18; checkpoint schema bumps from v3 to v4 with new `intercept_anchors` metadata. **The dynamic trajectory shield is unchanged this milestone.** Shield thinning is registered as the deferred `thin-trajectory-shield` follow-up.

**Drivers:**

1. Bucket-conditional intercept geometry is the cheapest fix for the known `turns_to_arrival` fleet-speed-blindness.
2. Iter-1's combined feature+shield scope inflated risk dramatically — critic identified feature/shield disagreement, static-path guard bypass, and legality-net duplication as critical. Decoupling moves all three concerns out of M4.
3. The encoding-v2 single-source catalog is recent and stable; this is the right time to bump schema before deferred per-planet τ lookahead (M5).

**Alternatives considered:**

- **Iter-1 plan: combined feature + thin-shield milestone.** Rejected by architect+critic consensus: too many independent gates, shield-feature legality disagreement is a critical safety surface, and rollback requires either deleting code or threading a guard flag.
- **Iter-1 plan variant: keep dynamic shield helpers as dead code behind a flag.** Rejected as part of the descoping: even the flag introduces a static-path bypass risk that the critic flagged.
- **Per-planet position lookahead at fixed τ (original brief).** Rejected: spec explicitly defers to M5.
- **One anchor only + bucket-fraction scalar.** Rejected in interview Round 2.
- **A/B/A reward arm.** Rejected: paired-seed A/B with same-commit baselines gives equivalent confidence at ⅓ the compute; borderline escalation to 6 seeds gives better statistical power per compute unit than a third arm.
- **Re-implement orbit lookahead in `feature_primitives.py` from scratch.** Rejected per critic guidance. Synthesis adopted: extract a shape-polymorphic `orbital_position_at_step_jax` primitive in `feature_primitives.py`, call it from `_edge_features` with edge-shaped inputs; trajectory-shield's `_planet_positions_at_step_jax` stays untouched in M4 (read-only) and is refactored to call the primitive in the follow-up milestone.
- **`task.intercept_features_enabled=true|false` flag for same-checkpoint A/B (architect antithesis).** Rejected for two reasons:
  1. **Encoder weight incompatibility.** The first-layer dense in `PlanetEdgeBackboneEncoder` (`edge_enc_0` kernel) has input dim `edge_feature_dim`. With the flag toggled, the kernel must be either E=12 or E=18; a single checkpoint cannot serve both branches cleanly. The flag would degenerate to either two parallel parameter blocks (defeating the A/B-on-same-checkpoint goal) or feature padding (which biases the A/B against the disabled branch).
  2. **Test surface and merge risk.** Carrying two encoder paths through Phase 2–4 would require dual goldens, dual catalog branches, and a guard in `feature_metadata` / `validate_checkpoint_feature_compatibility`. The win — saving one paired retrain at ~1.3 hours of compute — is not worth the multi-day implementation and review overhead. The Principle 1 commitment ("Replace, don't dual-encode") applies.
  
  Equivalent confidence is reached at lower total cost by the paired-seed A/B with separate baseline retrain (Phase 4 step 4 budget shows ~3 hours total).

**Consequences:**

Positive:
- Encoder has explicit bucket-conditional intercept geometry.
- `crosses_now` keeps a legality-aligned snapshot signal in the feature vector so the policy can mirror the shield's decision when useful.
- Schema bump to v4 is forward-compatible with the deferred shield-thinning milestone (no further bump needed there).
- M4 scope is small enough to land cleanly in ~1 week implementation + ~3 hours training.

Negative:
- Existing reward-sweep checkpoints are invalidated (acknowledged).
- Edge dim grows from 12 to 18; encoder param count rises modestly.
- The orbit lookahead has a transient cross-package coupling (`feature_primitives.py` imports a `_`-prefixed name from `trajectory_shield.py`) until the follow-up milestone consolidates.

**Reversibility:**

- **Phase 1 (catalog/schema):** Reversible by reverting commits; checkpoints from before this phase still load only because they pre-date schema v4 (the validator rejects them on load, which is the desired migration behavior).
- **Phase 2 (encoder integration):** Reversible by reverting `_edge_features` and the `feature_primitives.py` re-export. Goldens are per-feature so geometry regressions are detectable.
- **Phase 3 (docs/fixtures):** Trivially reversible.
- **Phase 4 (training/gates):** Pure measurement; no code reversibility concerns.

**Follow-ups (queued, not part of M4):**

- `thin-trajectory-shield` milestone (deferred status): drop dynamic per-bucket shield scan, add `task.trajectory_shield_static_only` flag, optionally relocate `_planet_positions_at_step_jax` into `feature_primitives.py` and replace the import.
- M5 (per-planet τ lookahead features) and intercept-sorted top-K re-rank under an ADR-002 amendment.

## Total Estimated Effort

- Phase 1: 1–2 days
- Phase 2: 2 days
- Phase 3: ½ day
- Phase 4: 3–5 days (mostly wall-clock; ~3 hours actual compute, rest is analysis and reporting)
- **Total: ~6–8 days** of mixed implementation, analysis, and training wait time. Pure implementation is ~4 days (vs ~6 days in iter-1).

## Open Questions — Resolution Log

All iter-2 open questions resolved by iter-3 fixup pass. Recorded for traceability:

1. **Per-edge τ JIT trace cost.** Resolved in Phase 2 design note: extract `orbital_position_at_step_jax` shape-polymorphic primitive; per-edge gather of orbital constants happens once, anchor loop is pure element-wise. No vmap, no `(P, K, P)` intermediate.
2. **`_planet_positions_at_step_jax` underscore reuse.** Resolved: do **not** import the underscore name. Add the new primitive directly in `feature_primitives.py`; trajectory-shield's private function stays inlined in M4 and is refactored by the follow-up milestone.
3. **A/B vs A/B/A.** Resolved: paired A/B at 3 seeds is the default; Phase 4 step 3 adds an empirical variance characterization that triggers up-front 6-seed escalation if power is insufficient. Borderline-result escalation remains as a secondary backstop.
4. **`crosses_now` rename co-existence.** Resolved: outright rename, no co-existence. Schema bump invalidates checkpoints anyway, so a transitional alias buys nothing.
