# Adding Observation Features

Short guide for extending the v2 planet / edge / global encoder. Each feature is declared **once** in a group catalog; dimensions, slice order, and `encode_turn` assembly are derived automatically.

Background: [feature-encoding-v2.md](feature-encoding-v2.md)

## Where to edit

| Group | Catalog file | Context type |
|-------|--------------|--------------|
| Planet (per-planet row) | `src/features/catalog/planet.py` | `PlanetAssemblyContext` |
| Edge (per source→target row) | `src/features/catalog/edge.py` | `EdgeRowAssemblyContext` |
| Global (one vector per decision) | `src/features/catalog/global_.py` | `GlobalAssemblyContext` |

Registry schemas (`PLANET_FEATURE_SCHEMA`, etc.) in `src/features/registry.py` are **built from these catalogs** — do not add parallel `FeatureItem` lists elsewhere.

Shared geometry / owner helpers live in `src/jax/feature_primitives.py`. Top-K edge ranking stays in `src/jax/features.py`; catalogs only assemble row values.

## Typical workflow (≤2 files)

### 1. Add a compute function and catalog entry

**Planet example** — append to `PLANET_FEATURE_ENTRIES` in `planet.py`:

```python
def _feat_my_signal(ctx: PlanetAssemblyContext) -> jnp.ndarray:
    return (ctx.planets.ships / ctx.scale)[..., None]

FeatureCatalogEntry(FeatureDefinition("my_signal"), _feat_my_signal),
```

**Edge example** — same pattern in `edge.py` using `EdgeRowAssemblyContext` (tensors are `(MAX_PLANETS, K, …)`).

**Global example** — same pattern in `global_.py` using `GlobalAssemblyContext` (each piece is a 1D segment concatenated into the frame).

### 2. (Optional) Add a test

- Unit test for the compute fn, or
- Extend `tests/test_feature_catalog_drift.py` if you need slice-level regression.

Run:

```bash
make test-domain-features
make test-fast
```

## Shape rules

- **`FeatureDefinition.size`** must match the **last axis** of the compute output.
- **Scalar features** (`size=1`): return `[..., None]` so planet/edge rows are `(N, 1)` or `(N, K, 1)`.
- **Multi-slot features** (`size=4`): return a single tensor with trailing width 4 (e.g. owner one-hot), not four separate catalog entries.
- **Global group**: return a 1D array of length `size`; the catalog concatenates on axis 0.

Order in the `*_FEATURE_ENTRIES` tuple **is** tensor column order. Renaming or reordering changes checkpoint input dims.

## When you need more than one file

Extend the context builder in the same catalog module if the feature needs new shared scratch:

- Planet: `build_planet_context()` in `planet.py`
- Global: `build_global_context()` in `global_.py` (preferred for bincounts / deltas)
- Edge: pre-gathered fields are built in `_edge_features()` in `src/jax/features.py`; add fields to `EdgeRowAssemblyContext` in `_types.py` only when the orchestrator must supply new tensors.

## Reading features elsewhere

Use named slices, not literal indices:

```python
from src.features.registry import PLANET_FEATURE_SCHEMA

orbit = batch.planet_features[..., PLANET_FEATURE_SCHEMA.base_slice("orbit_radius")]
```

Policy code should follow the same pattern (`src/jax/policy.py` GNN pointer is the reference).

## Checkpoints and compatibility

- Feature dims are embedded in checkpoint metadata (`schema_version`, `planet_feature_dim`, …).
- Layout changes require **retraining**; bump `schema_version` in `src/artifacts/checkpoint_compat.py` when you intentionally break old checkpoints.
- Current production encoder uses **`schema_version=3`** (single-source catalogs). Checkpoints with `schema_version < 3` are rejected on load.

## Quick checklist

- [ ] Entry added to the correct `*_FEATURE_ENTRIES` tuple
- [ ] Output shape matches `FeatureDefinition.size`
- [ ] No manual `jnp.stack` / duplicate registry list
- [ ] No new magic indices in policy or action builders
- [ ] `make test-domain-features` passes
