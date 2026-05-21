# Orbit Wars Agent Guide

## Repository Shape

- Python 3.12 reinforcement-learning project managed with `uv`; dependencies live in `pyproject.toml`.
- Canonical training entrypoint is Hydra-first: `uv run python -m src.train` with responsibility-group overrides.
- The runtime source is under `src/`; tests are under `tests/`; operational docs are under `docs/`.
- Hydra config is the source of truth under `conf/`. Responsibility groups include `model/`, `task/`, `reward/`, `training/`, `format/`, `opponents/`, `curriculum/`, `telemetry/`, and `artifacts/`.
- `mcp-server/` is a separate TypeScript Node project for the OMG MCP workflow server. Treat it as its own package with its own `package.json`, `tsconfig.json`, `src/`, and `test/` tree.

## Core Commands

- Install/sync Python dependencies: `uv sync --group dev`
- Run all Python tests: `uv run --group dev pytest`
- Run training with Hydra: `uv run python -m src.train model=attention training.total_updates=1000`
- Print resolved training config without training: `uv run python -m src.train print_resolved_config=true`
- Build MCP server: from `mcp-server/`, run `npm run build`
- Test MCP server: from `mcp-server/`, run `npm test`

## Python Subsystems

- `src/conf_schema.py` defines structured dataclass config defaults. When adding or renaming config fields, update this first.
- `src/config.py` composes and validates Hydra configs. It rejects legacy conflicts such as `ppo.rollout_groups`, `ppo.phases`, `ppo.num_envs_2p`, and `ppo.num_envs_4p`.
- `src/train.py` is a thin Hydra entrypoint that converts OmegaConf to `TrainConfig` and delegates to `src/jax_train.py`.
- `src/jax_train.py`, `src/jax_env.py`, `src/jax_features.py`, `src/jax_policy.py`, and `src/jax_ppo.py` are the JAX training path. Be careful with shape-defining config such as player count, candidate count, feature history, ship buckets, and model dimensions because these affect JIT compilation and checkpoint compatibility.
- `src/env.py` and `src/features.py` keep the Python environment/feature path. JAX and Python behavior are compared by parity tests, so mirror semantic changes across both paths when applicable.
- `src/feature_registry.py` owns ordered feature schemas and dimension checks against constants. Feature additions usually require updates in registry, encoders, JAX encoders, tests, and checkpoint compatibility expectations.
- `src/checkpoint_compat.py` validates checkpoint feature metadata; preserve this when changing feature dimensions or history behavior.

## Hydra And Experiment Rules

- Prefer editing `conf/` over adding ad hoc config files elsewhere. The old `configs/` layout is intentionally gone.
- Select config by responsibility group. Prefer `model=...`, `task.*`, `reward.*`, `training.*`, `format=...`, `opponents=...`, `curriculum=...`, `telemetry.*`, and `artifacts.*`.
- Use normal Hydra assignment for existing keys, such as `ppo.total_updates=2000`; use `+key=value` only for intentionally absent dynamic keys.
- Opponent behavior should usually be selected via `opponents=<profile>`. Avoid sweeping profile-owned fields like `self_play_enabled`, `self_play_pool_size`, `self_play_snapshot_interval`, and opponent curriculum internals unless deliberately editing the profile.
- Mixed 2p/4p JAX training uses `training_format.rollout_groups` and curriculum phases. Do not reintroduce deprecated `ppo.*` rollout group knobs.

## Testing Expectations

- For config/schema changes, run `uv run --group dev pytest tests/test_config_consolidation.py tests/test_curriculum.py tests/test_telemetry.py`.
- For environment, feature, or reward changes, run the relevant `tests/test_env.py`, `tests/test_features.py`, `tests/test_feature_history.py`, `tests/test_jax_env.py`, and `tests/test_jax_env_parity.py` coverage.
- For policy/PPO changes, run `tests/test_jax_policy.py` and `tests/test_jax_ppo.py`.
- For evaluation script changes, run `tests/test_evaluate.py`.
- Full Python verification is `uv run --group dev pytest`; expect JAX tests to be heavier than pure config/unit tests.

## MCP Server Notes

- `mcp-server/src/index.ts` registers tool groups for state, PRD, workflow, memory, checkpoint, model routing, bridge, and ultragoal support.
- Tests import compiled files from `mcp-server/dist/`, so run `npm run build` before `npm test` after TypeScript edits.
- Utility functions in `mcp-server/src/utils.ts` intentionally guard JSON parsing, file size, symlink reads/writes, and mode names. Preserve those safety checks when extending tools.

## Generated And Local Artifacts

- Ignore local training outputs and telemetry when making code changes: `outputs/`, `wandb/`, `artifacts/`, and Hydra run directories are runtime artifacts.
- `.omg/`, `.omc/`, and `.understand-anything/` may contain workflow or analysis state. Do not delete or rewrite them unless the task explicitly targets those systems.
- Keep future guidance concise and repository-specific. Prefer adding facts here only when they affect how agents safely edit, test, or run this repo.
