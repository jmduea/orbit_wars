# Feature Registry Single-Source Spec

Generated: 2026-05-25
Workflow: deep-interview
Final ambiguity: 13%

## Goal

Make adding a new RL observation feature a **single-source declaration** (name + compute function) instead of manually updating registry lists, dimension constants, encoder stack order, and scattered magic indices. Dimensions, slice positions, and tensor assembly must be **derived mechanically** from the declaration so stack order cannot drift.

## Context

Brownfield repo with v2 planet-edge-global encoding:

| Layer | Current owner | Pain |
|-------|---------------|------|
| Schema | `src/features/registry.py` — `FeatureItem` lists | Manual ordered lists |
| Dims | `src/game/constants.py` — `BASE_*_FEATURE_DIM` | Duplicated; validated at import |
| Encoding | `src/jax/features.py` — `_planet_features`, `_edge_features`, `_global_frame` | Manual `jnp.stack` / `concatenate` order must match registry |
| Policy | `src/jax/policy.py` | Hardcoded indices (`planet_features[..., 1]`) |
| Checkpoints | `src/artifacts/checkpoint_compat.py` | Dim metadata derived from registry helpers |

Prior trace (`.omg/specs/deep-dive-trace-feature-encoding.md`, manifest `feature-encoding-trace`) flagged registry-vs-manual-stack drift as high risk. Feature encoding v2 greenfield is complete; this is a **developer-ergonomics refactor** of the plumbing, not a new encoding schema.

## Interview Transcript

### Round 1 — Goal (Ambiguity 100% → ~50%)

**Q:** Primary outcome when adding a feature?

**A:** **Single declaration** — define once (name + compute fn); dims/slices derived automatically.

### Round 2 — Success Criteria (Ambiguity ~50% → ~24%)

**Q:** What must be true for success?

**A (selected):**

- Adding a feature touches **≤2 files**
- **No manual dim constants** — `BASE_*_FEATURE_DIM` derived from registry
- **Stack order can't drift** — registry order and encoder output mechanically linked
- **Policy uses named slices** — no magic indices

**Not selected:** checkpoint auto-compat with old runs; per-feature golden test hooks.

### Round 3 — Constraints (Ambiguity ~24% → ~13%)

**Q:** What bounds the refactor?

**A (selected):**

- **All three groups** — planet, edge, global use the same pattern
- **Migrate all existing features** in one cutover (no parallel legacy path)
- **Breaking checkpoints OK** — bump `schema_version`, retrain required

**Not selected:** incremental new-features-only migration; preserve current layout/dims.

## Assumptions Exposed & Resolved

| Assumption | Resolution |
|------------|------------|
| User wants codegen or plugin discovery | Rejected — single-source in code, not scaffold tooling |
| Checkpoint backward compatibility required | Rejected — schema_version bump acceptable |
| Incremental adoption | Rejected — full cutover of all existing features |
| Python encoder parity | N/A — JAX-only canonical path (`encode_turn`); no Python encoder to maintain |

## Non-Goals

- Changing feature *semantics* or removing features (unless required to fit the new assembly pattern with identical values)
- Codegen CLI or plugin auto-discovery as the primary UX
- Per-feature golden test framework (user did not select)
- Rewiring `ObservationNormalizer` or `model.normalize_observations` (separate initiative)

## Proposed Design (Ontology)

### Key entities

| Entity | Role |
|--------|------|
| `FeatureSpec` | Frozen record: `name`, `size`, `active`, group (`planet` \| `edge` \| `global`) |
| `FeatureCompute` | Callable bound to a `FeatureSpec`; returns `(…, size)` array for one group context |
| `FeatureGroupCatalog` | Ordered list of specs + compute fns for one group; exposes `base_dim`, `base_slice`, `assemble(context)` |
| `TurnBatch` | Unchanged policy input contract (shapes derived from catalogs + task config) |

### Assembly flow

```mermaid
flowchart LR
  defs[FeatureSpec + compute fn per group]
  cat[FeatureGroupCatalog.assemble]
  batch[TurnBatch tensors]
  policy[policy.py named slices]

  defs --> cat --> batch --> policy
```

1. Each group defines an **ordered catalog** of `(FeatureSpec, compute_fn)` pairs in **one module** (target: `src/features/definitions.py` or split `planet.py` / `edge.py` / `global.py`).
2. `FeatureGroupRegistry` (or successor) is **built from** the catalog — not a separate hand-maintained list.
3. `_planet_features` / `_edge_features` / `_global_frame` become thin wrappers: build context dataclass → `catalog.assemble(context)` → `(N, base_dim)` or `(N, K, base_dim)` or `(base_dim,)`.
4. `BASE_PLANET_FEATURE_DIM`, `BASE_EDGE_FEATURE_DIM`, `BASE_GLOBAL_FEATURE_V2_DIM` in `constants.py` are **removed** and replaced by `catalog.base_dim` (or module-level aliases computed at import from catalog).
5. Policy reads geometry via `planet_feature_schema(env_cfg).base_slice("orbit_radius")` etc.

### Adding a new feature (target UX)

1. Append one `(FeatureSpec, compute_fn)` entry to the appropriate group catalog (**1 file**).
2. Optional: add a focused unit test for the compute fn or a registry slice assertion (**2nd file**).

No edits to `constants.py`, manual `jnp.stack` blocks, or policy magic indices.

## Acceptance Criteria

1. **Single-source catalogs** exist for planet, edge, and global groups; no duplicate `FeatureItem` lists detached from compute fns.
2. **Derived dimensions** — removing `BASE_*_FEATURE_DIM` manual constants; import-time validation uses `catalog.base_dim`.
3. **Mechanical assembly** — `_planet_features`, `_edge_features`, `_global_frame` delegate to catalog `assemble`; no hand-ordered feature stacks.
4. **Drift guard** — test asserts encoded tensor slices match `registry.base_slice(name)` for every active feature in each group.
5. **Policy named slices** — GNN pointer orbit geometry uses registry slices, not literals `1` and `2`.
6. **Full migration** — all current v2 features preserved at identical values on golden fixtures (`tests/test_feature_encoding_golden.py` updated, not deleted).
7. **Checkpoint bump** — `schema_version` → `3` in `checkpoint_compat.py`; dim metadata still auto-derived.
8. **≤2-file rule** — documented worked example in spec or module docstring showing add-feature workflow.
9. **Tests** — `make test-domain-features` and `make test-fast` pass.

## Risks

| Risk | Mitigation |
|------|------------|
| Global history/delta features need prior-frame context | Context dataclass carries `history`, `previous_global`, `previous_ships`; compute fns opt into history deps |
| JAX `jnp.stack` performance | Benchmark negligible; assembly still one stack per group |
| Large refactor touch surface | Keep semantic golden tests; no behavior change intended |
| Edge case: multi-dim features (`owner_slot` size=4) | `FeatureSpec.size` already supports; compute fn returns last axis of width `size` |

## Related Work

- Manifest: `feature-encoding` (complete), `feature-encoding-trace` (draft trace — informs drift risk)
- Does not block or duplicate `jax-train-split` (draft)
