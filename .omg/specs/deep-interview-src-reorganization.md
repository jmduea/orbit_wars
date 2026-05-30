# Source Reorganization Deep Interview Spec

Date: 2026-05-24
Ambiguity: 18%

## Goal

Reorganize the flat `src/` package into domain-owned subpackages so contributors can quickly tell what each part of the codebase owns and where new behavior should live.

The reorganization should optimize first for ownership boundaries, not merely reducing the number of top-level files.

## Target Package Model

Use operational domain packages:

- `src/config/` for Hydra composition, structured config schemas, config validation, and config-owned registries.
- `src/game/` for core game constants, types, parsing, and shared environment concepts.
- `src/features/` for feature schemas, feature encoding, history, and normalization concerns.
- `src/jax/` for JAX environment, JAX feature encoding, JAX policy/model code, PPO implementation, device handling, and accelerated training internals.
- `src/training/` for training orchestration, curriculum, seed scheduling, and training entrypoint support.
- `src/opponents/` for opponent construction, opponent policies, and opponent pool management.
- `src/artifacts/` for checkpoint compatibility, checkpoint retention, replay generation, artifact pipeline, and run path/manifests.
- `src/telemetry/` for telemetry logging and metrics filtering/registry behavior unless a specific registry is more clearly config-owned.

Keep `src.train` stable enough that the canonical command remains:

```bash
uv run python -m src.train
```

## Constraints

- A clean internal import break is acceptable.
- Update repository imports, tests, scripts, docs, and Hydra sweep command references as part of the first pass.
- Preserve public-ish entrypoints where practical, especially `python -m src.train`.
- Do not preserve old flat-module compatibility aliases merely for removed internals.
- Prefer package ownership over call-site convenience.
- Split ambiguous files when keeping them whole would preserve unclear ownership or create package cycles.
- Do not split ambiguous files just because they are large; the split must improve ownership clarity.
- Keep behavior changes out of scope unless required by the reorganization.

## Ambiguous Module Policy

Default approach:

1. Move modules into the package that owns their primary responsibility.
2. Inspect imports for package cycles or misleading ownership.
3. Split only when the package model would otherwise be tangled or semantically false.

Known modules to evaluate carefully:

- `trajectory_shield.py`: currently spans game safety, runtime feature/action selection, and JAX policy types. Split only if one owner would make the others awkward or cyclic.
- `curriculum.py`: likely training-owned, but verify opponent-pool boundaries.
- `metric_registry.py`: may be telemetry-owned or config-owned depending on whether validation/schema ownership dominates.
- `jax_train.py`: likely training entry orchestration plus JAX-specific training internals; preserve `src.train` while moving internals to the right packages.

## Non-Goals

- Do not redesign PPO, feature encoding, reward semantics, Hydra responsibility groups, or checkpoint compatibility behavior.
- Do not restructure `mcp-server/`; it is a separate TypeScript project.
- Do not delete local workflow/runtime artifacts such as `.omg/`, `.omc/`, `.understand-anything/`, `outputs/`, `wandb/`, or `artifacts/`.
- Do not introduce broad compatibility shims for every old flat import unless needed for a deliberately stable public entrypoint.

## Acceptance Criteria

- `src/` is organized into the target domain packages.
- All imports in `src/`, `tests/`, `scripts/`, docs, and relevant config/sweep files are updated.
- `uv run python -m src.train print_resolved_config=true` still works.
- Relevant targeted tests pass after the move:
  - `uv run --group dev pytest tests/test_config_consolidation.py tests/test_curriculum.py tests/test_telemetry.py`
  - `uv run --group dev pytest tests/test_env.py tests/test_features.py tests/test_feature_history.py tests/test_jax_env.py tests/test_jax_env_parity.py`
  - `uv run --group dev pytest tests/test_jax_policy.py tests/test_jax_ppo.py`
  - `uv run --group dev pytest tests/test_replay.py tests/test_artifact_pipeline.py tests/test_run_paths.py`
- Full verification target is `uv run --group dev pytest`.
- Documentation references to old module paths or `src.train` commands are accurate after the refactor.

## Assumptions Resolved

- The user wants an implementation-oriented refactor, not only a plan.
- The first pass should update source, tests, scripts, docs, and sweep references.
- Domain ownership is more important than JAX/non-JAX separation alone.
- Clean internal package paths are acceptable.
- Stable-ish public entrypoints are preferred, especially the training command.
- Ambiguous modules should be split when necessary to stamp out unclear ownership, but the refactor should not become an unnecessary redesign.

## Interview Transcript

- Round 1: User chose domain ownership boundaries as the primary optimization.
- Round 2: User accepted a clean internal break while preferring stable-ish public entrypoints.
- Round 3: User chose full source plus docs/scripts updates as the first-pass completion target.
- Round 4: User chose operational domain packages: config, game, features, jax, training, opponents, artifacts, telemetry.
- Round 5: User preferred splitting ambiguous files when real ownership issues emerge, while avoiding overcomplication.
- Round 6: User chose the guardrail: move first, split only to avoid bad boundaries or cycles.

