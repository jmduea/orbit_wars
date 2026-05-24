# Ralplan: Source Reorganization

Date: 2026-05-24

## Decision

Reorganize `src/` into operational domain packages while preserving the canonical training command `uv run python -m src.train`.

## Decision Drivers

- Make ownership boundaries visible in the filesystem.
- Avoid retaining the flat namespace as a compatibility layer for internals.
- Keep behavior changes out of scope and preserve existing dirty work.
- Split ambiguous modules only when required to avoid misleading package ownership or import cycles.

## Chosen Approach

Use domain packages with package-level facades where a domain has a natural public API:

- `src/config/`: config schema, Hydra composition/validation, config-owned exports.
- `src/game/`: constants, game types, Python environment.
- `src/features/`: feature registry, feature encoding/history, normalization.
- `src/jax/`: JAX environment, feature encoding, policy/model, PPO, device helpers, JAX training internals.
- `src/training/`: curriculum, seed scheduling, entrypoint support.
- `src/opponents/`: opponent policies and pools.
- `src/artifacts/`: checkpoint compatibility/retention, replay, artifact pipeline, run paths.
- `src/telemetry/`: telemetry logger and metric registry.

Keep `src/train.py` as the stable executable module. Do not keep top-level files such as `src/jax_ppo.py` or `src/features.py` solely as compatibility shims.

## Alternatives Considered

- Full compatibility shims for every old flat module: rejected because it preserves the confusing flat surface.
- JAX/non-JAX split first: rejected because the user's primary goal is domain ownership, not runtime backend separation.
- Split all ambiguous files immediately: rejected because it risks turning a structural refactor into a behavior redesign.

## Implementation Plan

1. Create package directories and move files to domain-owned names.
2. Add `__init__.py` facades for stable domain APIs.
3. Update imports in `src/`, `tests/`, `scripts/`, docs, and Hydra sweep configs.
4. Preserve user-local changes in `src/jax_ppo.py`, configs, and curriculum tests during moves.
5. Run import-oriented checks, then targeted pytest groups, then full pytest if feasible.
6. Fix only behavior-neutral failures caused by the reorganization.

## Risk Controls

- Use `git status` and diffs before and after moves.
- Prefer mechanical import rewrites before manual changes.
- If package cycles appear, split the smallest ambiguous module surface needed to break the cycle.
- Validate `uv run python -m src.train print_resolved_config=true`.

