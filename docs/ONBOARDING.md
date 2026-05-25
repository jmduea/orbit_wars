# Orbit Wars — Onboarding Guide

*Generated from the Understand-Anything knowledge graph (analyzed 2026-05-22, commit `e9ab5ab`).*

---

## Project Overview

**Orbit Wars** (`orbit-wars`) is a Hydra-first Python reinforcement-learning project for the Orbit Wars game. It trains policies with JAX/PPO, uses structured configuration and experiment operations, and ships an OMG MCP workflow server for agent tooling.

| | |
|---|---|
| **Languages** | Python (primary), TypeScript, YAML, TOML, shell, Makefile |
| **Frameworks** | Hydra, JAX, Flax, Optax, pytest, Weights & Biases, MCP |
| **Package manager** | [uv](https://github.com/astral-sh/uv) (Python 3.12) |

**First commands**

```bash
uv sync --group dev
uv run --group dev pytest
uv run python -m src.train model=attention training.total_updates=1000
uv run python -m src.train print_resolved_config=true   # dry-run config only
```

Canonical agent guidance also lives in `AGENTS.md` and `README.md`.

---

## Architecture Layers

The repo is organized into six layers. Start with **Training Runtime** and **Hydra Configuration**; treat **OMG MCP Server** as a separate package.

### 1. Training Runtime (`src/`)

Environment dynamics, feature encoders, policies, PPO, training orchestration, telemetry, artifacts, and checkpoints.

| File | Role | Complexity |
|------|------|------------|
| `train.py` | Thin Hydra entry → `TrainConfig` → `jax_train` | simple |
| `config.py` | Composes/validates Hydra configs | **complex** |
| `conf_schema.py` | Dataclass config defaults (edit first for schema changes) | **complex** |
| `jax_train.py` | JAX training loop orchestration | **complex** |
| `env.py` / `jax_env.py` | Python vs vectorized JAX environment | **complex** |
| `features.py` / `jax_features.py` | Feature encoders (keep in parity) | **complex** |
| `feature_registry.py` | Ordered feature schemas, dimension checks | moderate |
| `jax_policy.py` / `jax_ppo.py` | Model + PPO updates | **complex** |
| `curriculum.py` / `opponents.py` | Staged self-play and opponent profiles | moderate |
| `telemetry.py` | W&B and run metrics | moderate |
| `artifact_pipeline.py` | Async artifact handling | **complex** |
| `checkpoint_compat.py` | Checkpoint feature metadata validation | moderate |
| `trajectory_shield.py` | Action masking / trajectory constraints | **complex** |
| `constants.py`, `game_types.py`, `normalization.py`, `seed_scheduler.py`, `jax_device.py` | Shared types, norms, device setup | simple–moderate |

### 2. Hydra Configuration (`conf/`)

Responsibility groups: `model/`, `task/`, `reward/`, `training/`, `format/`, `opponents/`, `curriculum/`, `telemetry/`, `artifacts/`. Root `conf/config.yaml` composes them. Model presets include `attention`, `mlp`, `gnn_pointer`, and entity-transformer variants.

Override with normal Hydra syntax, e.g. `training.total_updates=2000`. Prefer `opponents=<profile>` over sweeping profile-internal fields.

### 3. Test Suite (`tests/`, `mcp-server/test/`)

Pytest validates config, curriculum, features, JAX parity, PPO, telemetry, artifacts, and trajectory shield. MCP tests live under `mcp-server/test/` and import from `dist/` — run `npm run build` in `mcp-server/` after TS changes.

### 4. Operations and Docs (`docs/`, `scripts/`, `.github/`)

- `docs/experiments.md`, `docs/hydra_migration.md`, `docs/config_migration.md` — experiment and config history
- `scripts/benchmark_jax_rl.py`, `scripts/compare_attention_candidates.py` — benchmarks and comparisons
- `.github/` — OMG agents, skills, hooks (synced to `.cursor/` via `scripts/sync_omg_cursor.py`)

### 5. OMG MCP Server (`mcp-server/`)

TypeScript MCP server: state, PRD, memory, checkpoint, bridge (Claude/OMC import), model routing, ultragoal. Entry: `mcp-server/src/index.ts`.

### 6. Miscellaneous

Local caches, session checkpoints under `.omg/`, and generated training outputs (`outputs/`, `wandb/`, `artifacts/`) — not source of truth.

---

## Key Concepts

1. **Hydra-first training** — YAML responsibility groups compose into dataclasses; `src/train.py` only bridges Hydra → JAX. Config shape drives JIT and checkpoint compatibility.

2. **Dual environment paths** — `env.py` / `features.py` (reference Python) mirror `jax_env.py` / `jax_features.py` (training). Parity tests (`test_jax_env_parity.py`) require semantic alignment when you change game state or features.

3. **Feature registry as contract** — `feature_registry.py` defines ordered schemas and dimensions. New features touch registry, both encoders, tests, and `checkpoint_compat.py`.

4. **Mixed 2p/4p training** — `format.rollout_groups` and curriculum stages control mixed player counts; avoid reintroducing flat rollout knobs.

5. **Opponent profiles** — Select via `opponents=<profile>` (`self_play_curriculum`, `latest_only`, etc.) rather than ad hoc nested overrides.

6. **Checkpoint compatibility** — Checkpoints embed feature metadata; old flat config shapes are rejected. Shape-defining fields (player count, candidates, history, ship buckets, model dims) are breaking changes.

7. **OMG workflow bridge** — `.github/` is the catalog; Cursor uses `.cursor/` (regenerate with `sync_omg_cursor.py`). MCP workflow tools need a built `mcp-server/dist/`.

---

## Guided Tour

Recommended learning path from the knowledge graph tour:

| Step | Focus | Read / run |
|------|--------|------------|
| **1** | Project overview | `README.md`, `AGENTS.md` |
| **2** | Training entry | `src/train.py` → `config.py` → `jax_train.py` |
| | *Lesson* | Hydra composes YAML groups before dataclasses and JAX consume them |
| **3** | Environment & features | `env.py`, `jax_env.py`, `features.py`, `jax_features.py`, `feature_registry.py` |
| **4** | Policy & PPO | `jax_policy.py`, `jax_ppo.py`, `normalization.py`, `trajectory_shield.py` |
| **5** | Configuration surface | `conf/config.yaml`, `conf/model/*`, `conf/task/default.yaml`, `conf/training/default.yaml` |
| **6** | Verification & ops | `docs/*`, `scripts/benchmark_jax_rl.py`, targeted pytest modules |
| **7** | OMG MCP server | `mcp-server/src/index.ts`, tool modules, `npm run build` + `npm test` |

---

## File Map (by layer)

### Training Runtime — quick reference

- **Entry & config:** `train.py`, `config.py`, `conf_schema.py`
- **JAX core:** `jax_train.py`, `jax_env.py`, `jax_features.py`, `jax_policy.py`, `jax_ppo.py`
- **Python parity:** `env.py`, `features.py`
- **Training support:** `curriculum.py`, `opponents.py`, `opponent_pool.py`, `replay.py`, `run_paths.py`, `metric_registry.py`, `checkpoint_retention.py`
- **Safety & I/O:** `trajectory_shield.py`, `artifact_pipeline.py`, `checkpoint_compat.py`, `telemetry.py`

### Hydra Configuration

- `conf/config.yaml` — root composition
- `conf/model/*.yaml` — architecture presets
- `conf/task/`, `conf/reward/`, `conf/training/`, `conf/format/`, `conf/opponents/`, `conf/curriculum/`, `conf/telemetry/`
- `pyproject.toml`, `Makefile` — project tooling

### Test Suite (high-signal)

| Test file | What it guards |
|-----------|----------------|
| `test_config_consolidation.py` | Hydra/schema wiring |
| `test_curriculum.py` | Curriculum stages |
| `test_jax_env_parity.py` | Python ↔ JAX env parity |
| `test_jax_ppo.py`, `test_jax_policy.py` | RL update path |
| `test_trajectory_shield.py` | Action shield semantics |
| `test_artifact_pipeline.py` | Artifact async pipeline |
| `test_telemetry.py` | Metrics/W&B integration |

### OMG MCP Server

| File | Purpose |
|------|---------|
| `index.ts` | Tool registration |
| `state-tools.ts`, `prd-tools.ts`, `workflow-tools.ts` | Workflow state & PRD |
| `memory-tools.ts`, `checkpoint-tools.ts` | Memory & session checkpoints |
| `ultragoal-tools.ts` | Durable goals (most complex MCP module) |
| `bridge/*` | Claude/OMC session import/export |
| `utils.ts` | JSON/size/symlink guards |

### Operations scripts

| Script | Purpose |
|--------|---------|
| `benchmark_jax_rl.py` | JAX RL throughput benchmarks |
| `compare_attention_candidates.py` | Compare attention configs from logs |
| `run_artifact_worker.py` | Artifact worker process |
| `validate_kaggle_docker_submission.py` | Kaggle submission validation (**complex**) |

---

## Complexity Hotspots

Approach these with extra care — they are marked **complex** in the graph or are central to correctness:

### Training runtime (core RL path)

- `src/config.py`, `src/conf_schema.py` — config schema changes ripple everywhere
- `src/env.py`, `src/jax_env.py` — game dynamics and vectorization
- `src/features.py`, `src/jax_features.py` — observation space (checkpoint-breaking)
- `src/jax_train.py`, `src/jax_policy.py`, `src/jax_ppo.py` — training loop and learning
- `src/artifact_pipeline.py`, `src/trajectory_shield.py`, `src/metric_registry.py`

### Tests that mirror complex behavior

- `tests/test_jax_env_parity.py`, `tests/test_jax_ppo.py`, `tests/test_trajectory_shield.py`, `tests/test_artifact_pipeline.py`

### MCP & tooling

- `mcp-server/src/ultragoal-tools.ts`
- `mcp-server/test/bridge-tests.mjs`
- `scripts/validate_kaggle_docker_submission.py`

**Safer starting points:** `src/train.py`, `src/constants.py`, `src/game_types.py`, `tests/conftest.py` (all **simple**).

---

## Suggested verification matrix

When you touch a subsystem, run the focused tests from `AGENTS.md`:

| Change area | Tests |
|-------------|--------|
| Config / schema | `test_config_consolidation.py`, `test_curriculum.py`, `test_telemetry.py` |
| Env / features | `test_env.py`, `test_features.py`, `test_feature_history.py`, `test_jax_env.py`, `test_jax_env_parity.py` |
| Policy / PPO | `test_jax_policy.py`, `test_jax_ppo.py` |
| Full check | `uv run --group dev pytest` |

---

## Regenerating this guide

Run `/understand` to refresh `.understand-anything/knowledge-graph.json`, then `/understand-onboard` to update this document.
