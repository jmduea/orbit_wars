# Orbit Wars — Onboarding Guide

*Generated from the Understand-Anything knowledge graph (analyzed 2026-05-25, commit `56ea682`). Scope: project RL code only — OMG/Cursor mirrors, `mcp-server/`, and training outputs are excluded via `.understandignore`.*

---

## Project Overview

**Orbit Wars** (`orbit-wars`) is a Hydra-first Python 3.12 reinforcement-learning project for the Orbit Wars game. Policies train with JAX/Flax PPO; configuration is split into responsibility groups under `conf/`; experiments log to Weights & Biases; checkpoints and replays flow through an artifact pipeline.

| | |
|---|---|
| **Languages** | Python (primary), YAML, TOML, JSON, Markdown, shell, Makefile |
| **Frameworks** | Hydra, JAX, Flax, Optax, pytest, Weights & Biases |
| **Package manager** | [uv](https://github.com/astral-sh/uv) (Python 3.12) |

**First commands**

```bash
uv sync --group dev
uv run --group dev pytest
uv run python -m src.train model=attention training.total_updates=1000
uv run python -m src.train print_resolved_config=true   # resolve config without training
```

Canonical references: `README.md` (user-facing) and `AGENTS.md` (agent/editor guidance).

---

## Architecture Layers

The scoped graph organizes **117 file-level nodes** into six layers. Read **Training Runtime** and **Hydra Configuration** first; use **Test Suite** to validate any change.

### 1. Training Runtime (`src/`)

JAX/Python RL stack: Hydra entrypoint, game simulation, feature encoding, PPO policies, opponents, curriculum, telemetry, and checkpoint artifact pipelines.

| Package | Role |
|---------|------|
| `src/train.py` | Thin Hydra CLI → typed config → `src/jax/train.py` |
| `src/config/` | Dataclass schema (`schema.py`) and Hydra composition/validation (`runtime.py`) |
| `src/game/` | Reference Python env, types, constants, trajectory shield |
| `src/features/` | Feature registry, encoding, normalization |
| `src/jax/` | Vectorized env, features, policy, PPO, training orchestrator |
| `src/opponents/` | Opponent pool and runtime mixing |
| `src/training/` | Curriculum stages and seed scheduling |
| `src/telemetry/` | Metric registry and experiment logger |
| `src/artifacts/` | Checkpoints, replay eval, async pipeline, path layout |

### 2. Hydra Configuration (`conf/`)

Responsibility-group YAML composing models, tasks, rewards, training budgets, formats, opponents, curriculum, telemetry, artifacts, and W&B sweep grids.

| Group | Examples |
|-------|----------|
| `model/` | `attention`, `mlp`, `gnn_pointer`, `planet_graph_transformer`, entity-transformer presets |
| `task/`, `reward/`, `training/` | Shape, reward shaping, PPO budget |
| `format/` | 2p/4p and mixed rollout topologies |
| `opponents/`, `curriculum/` | Self-play profiles and staged progression |
| `telemetry/`, `artifacts/` | Metric groups and checkpoint cadence |
| `sweeps/wandb/` | Campaign templates (capacity, budget, reward, throughput, …) |

Override with normal Hydra syntax, e.g. `training.total_updates=2000`. Prefer `opponents=<profile>` over sweeping profile-internal fields.

### 3. Test Suite (`tests/`)

Pytest guards Hydra wiring, JAX env/policy/PPO behavior, feature dimensions, curriculum, telemetry, artifacts, trajectory shield, and Kaggle packaging.

### 4. Documentation (`docs/`)

Experiment guides, baseline sweep notes, Hydra/config migration history, and this onboarding doc.

### 5. Scripts (`scripts/`)

Operational CLIs: JAX benchmarks, artifact workers, Kaggle Docker validation, attention log comparison, IDE tooling sync.

### 6. Project Root

`README.md`, `AGENTS.md`, `pyproject.toml`, `Makefile`, and Understand-Anything graph metadata under `.understand-anything/`.

---

## Key Concepts

1. **Hydra-first training** — YAML groups compose into dataclasses in `src/config/`; `src/train.py` only bridges Hydra to JAX. Shape-defining fields drive JIT compilation and checkpoint compatibility.

2. **Package layout** — Game rules live in `src/game/`, observations in `src/features/`, acceleration in `src/jax/`. Do not look for legacy flat modules (`src/env.py`, `src/config.py`); the graph and imports use the package paths.

3. **Python ↔ JAX parity** — `src/game/env.py` and `src/features/encoding.py` are reference implementations; `src/jax/env.py` and `src/jax/features.py` must stay semantically aligned. `tests/test_jax_env_parity.py` is the guardrail.

4. **Feature registry as contract** — `src/features/registry.py` fixes ordered schemas and dimensions. New features require registry, both encoders, tests, and `src/artifacts/checkpoint_compat.py`.

5. **Mixed 2p/4p training** — `format.rollout_groups` and `src/training/curriculum.py` control player-count mix and stage gates; avoid reintroducing flat rollout knobs.

6. **Opponent profiles** — Select via `opponents=<profile>` (`self_play_curriculum`, `noop_only`, `latest_only`, etc.) rather than ad hoc nested overrides.

7. **Checkpoint compatibility** — Checkpoints embed feature metadata; incompatible shapes or old flat config layouts are rejected in `checkpoint_compat.py`.

8. **Campaign outputs** — New runs use `outputs/campaigns/<campaign>/runs/<run_id>/` with Hydra snapshots, manifests, and checkpoints grouped by experimental question (`output.campaign=`).

---

## Guided Tour

Recommended path from the knowledge graph (10 steps). Each step links file-level nodes you can open in the [Understand dashboard](/understand-dashboard) after `/understand`.

| Step | Title | What to read |
|------|--------|----------------|
| **1** | Hydra Training Entry | `src/train.py` — canonical `uv run python -m src.train`; maps CLI overrides to responsibility groups |
| **2** | Config Schema & Runtime | `src/config/schema.py`, `src/config/runtime.py` — dataclass contract and Hydra validation |
| **3** | JAX Training Orchestrator | `src/jax/train.py` — devices, vectorized envs, PPO loop, curriculum, logging, artifacts |
| **4** | Game Rules & Reference Env | `src/game/constants.py`, `types.py`, `env.py` — ground truth for game dynamics |
| **5** | Observation Features | `src/features/registry.py`, `encoding.py`, `src/jax/features.py` — observation layout for policies |
| **6** | JAX Environment & PPO | `src/jax/env.py`, `src/jax/rollout/`, `src/jax/ppo_update.py` — batched stepping and policy updates |
| **7** | Policy Architectures | `src/jax/policy.py` — Flax modules selected by `conf/model/*.yaml` |
| **8** | Safe Launches & Opponents | `src/game/trajectory_shield.py`, `src/opponents/*`, `src/training/curriculum.py` |
| **9** | Telemetry & Experiment Logging | `src/telemetry/metric_registry.py`, `logger.py` |
| **10** | Checkpoints & Artifact Pipeline | `src/artifacts/pipeline.py`, `replay.py`, `checkpoint_compat.py` |

**Hydra lesson (step 1):** Groups compose at runtime; overrides like `model=attention training.total_updates=1000` nest without editing defaults on disk.

**JAX lesson (step 6):** `jit`/`vmap` batch independent envs; PPO relies on static shapes from task config (players, candidates, history frames) to avoid recompilation.

---

## File Map

### Training Runtime — by package

| Path | Summary |
|------|---------|
| `src/train.py` | Hydra entry; resolves `TrainConfig` and delegates to JAX training |
| `src/config/schema.py` | Dataclass defaults for all responsibility groups |
| `src/config/runtime.py` | OmegaConf resolvers, composition, validation |
| `src/game/env.py` | Python reset/step, rewards, opponent hooks |
| `src/game/trajectory_shield.py` | Blocks illegal fleet launches during rollouts |
| `src/features/registry.py` | Ordered feature schemas and dimension checks |
| `src/features/encoding.py` | Self/candidate/global tensors with history |
| `src/features/normalization.py` | Feature normalization helpers |
| `src/jax/train.py` | Main training control plane |
| `src/jax/env.py` | Vectorized game simulation |
| `src/jax/features.py` | JIT feature encoder matching Python layout |
| `src/jax/policy.py` | MLP, attention, entity transformer, GNN pointer |
| `src/jax/rollout/` | Rollout collection, transition types, diagnostics |
| `src/jax/ppo_update.py` | PPO loss and batch utilities |
| `src/jax/device.py` | Device selection utilities |
| `src/opponents/pool.py` | Opponent source definitions |
| `src/opponents/runtime.py` | Mixing random, scripted, self-play, snapshots |
| `src/training/curriculum.py` | Stage progression and win-rate gates |
| `src/training/seed_scheduler.py` | Reseeding schedule for rollouts |
| `src/telemetry/metric_registry.py` | Named metric groups for logging |
| `src/telemetry/logger.py` | W&B / JSONL flattening |
| `src/artifacts/pipeline.py` | Async checkpoint and replay jobs |
| `src/artifacts/replay.py` | Policy evaluation against opponents |
| `src/artifacts/checkpoint_compat.py` | Feature metadata and config-shape validation |
| `src/artifacts/run_paths.py` | Campaign/run directory layout |
| `src/artifacts/checkpoint_retention.py` | Retention policy for checkpoints |

#### Seed scheduler — config and observability

Periodic rollout reseeding is **off by default** (`training.reseed_every_updates: 0`, `training.reseed_on_plateau: false`). Enable periodic swaps with Hydra, for example:

```bash
uv run python -m src.train training.reseed_every_updates=50
```

Plateau-triggered reseeding uses `training.plateau_metric`, `plateau_window`, and `plateau_delta`. When `heldout_eval_seed_set` is non-empty, reseeds draw from a shuffled pool instead of random jumps.

During JAX training (`src/jax/train.py`), the scheduler runs **before each rollout**. When a swap fires, the PRNG key is replaced and a single-entry list is attached to the update record:

| Field | Meaning |
|-------|---------|
| `reseed_events` | List of `{update, old_seed, new_seed, reason, policy}`; empty when no swap |
| `seed_scheduler_policy` | Policy that would apply on the next reseed (`incremental`, `random_jump`, `shuffled_pool`) |
| `seed_scheduler_plateau_metric` | Metric name monitored for plateau detection |

These fields land in the run JSONL (`campaigns/*/runs/*/logs/*_jax.jsonl`) and in W&B when telemetry is enabled (`events` metric group).

**Verify locally**

- Fast unit coverage: `uv run --group dev pytest tests/test_seed_scheduler.py -m "not slow and not jax"`
- End-to-end smoke (slow/JAX compile): `uv run --group dev pytest tests/test_jax_seed_scheduler.py -m slow`

### Hydra Configuration — high-signal files

| Path | Role |
|------|------|
| `conf/config.yaml` | Root composition of responsibility groups |
| `conf/task/default.yaml` | Player count, candidates, feature history |
| `conf/training/default.yaml` | PPO budget, batching, optimizer |
| `conf/model/*.yaml` | Architecture capacity presets |
| `conf/format/*.yaml` | 2p/4p and mixed rollout groups |
| `conf/opponents/*.yaml` | Opponent family profiles |
| `conf/curriculum/*.yaml` | Staged progression |
| `conf/sweeps/wandb/*.yaml` | W&B multirun templates |

### Test Suite — what each file guards

| Test | Focus |
|------|--------|
| `test_config_consolidation.py` | Hydra group wiring and schema |
| `test_curriculum.py` | Curriculum stage transitions |
| `test_features.py`, `test_feature_history.py`, `test_feature_registry.py` | Observation layout |
| `test_jax_env.py`, `test_jax_env_parity.py` | JAX env and Python parity |
| `test_jax_policy.py`, `test_jax_ppo.py` | Policy and PPO path |
| `test_trajectory_shield.py` | Launch shield semantics |
| `test_telemetry.py`, `test_metric_registry.py` | Metrics and logging |
| `test_artifact_pipeline.py`, `test_replay.py`, `test_run_paths.py` | Artifacts and paths |
| `test_jax_train_timing.py` | Training loop timing smoke |
| `test_seed_scheduler.py` | Seed scheduler policies and parse helpers |
| `test_jax_seed_scheduler.py` | JAX JSONL `reseed_events` integration smoke |
| `test_kaggle_submission_packager.py` | Submission packaging |

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/benchmark_jax_rl.py` | JAX RL throughput benchmarks |
| `scripts/compare_attention_candidates.py` | Compare attention configs from logs |
| `scripts/run_artifact_worker.py` | Artifact worker process |
| `scripts/validate_kaggle_docker_submission.py` | Kaggle Docker submission validation |
| `Makefile` | `setup`, `test-fast`, and domain test targets |

### Documentation

| Doc | Topic |
|-----|--------|
| `docs/experiments.md` | Experiment conventions |
| `docs/baseline_sweep.md` | Baseline sweep methodology |
| `docs/config_migration.md`, `docs/hydra_migration.md` | Config history |

---

## Complexity Hotspots

Files marked **complex** in the graph — read tests alongside code before changing behavior.

### Core RL path (`src/`)

- `src/config/schema.py`, `src/config/runtime.py` — schema and validation ripple through training
- `src/game/env.py`, `src/jax/env.py` — game dynamics and vectorization
- `src/features/registry.py`, `src/features/encoding.py`, `src/jax/features.py` — observation contract (checkpoint-breaking)
- `src/jax/train.py`, `src/jax/policy.py`, `src/jax/rollout/`, `src/jax/ppo_update.py` — training loop and learning
- `src/game/trajectory_shield.py`, `src/opponents/runtime.py` — safety and opponent mixing
- `src/telemetry/metric_registry.py` — metric surface area
- `src/artifacts/pipeline.py`, `replay.py`, `checkpoint_compat.py`, `run_paths.py` — persistence and compatibility

### Tests mirroring complex behavior

- `tests/test_jax_env.py`, `test_jax_env_parity.py`, `test_jax_ppo.py`
- `tests/test_curriculum.py`, `test_trajectory_shield.py`, `test_telemetry.py`

### Scripts

- `scripts/run_artifact_worker.py`, `validate_kaggle_docker_submission.py`

**Safer starting points:** `src/train.py`, `src/game/constants.py`, `src/game/types.py`, `tests/conftest.py` (simple complexity in the graph).

---

## Suggested verification matrix

Canonical **agent** test-selection rules live in `AGENTS.md` § *Test Selection For Coding Agents*. Quick reference:

| Change area | Quick command | Files / notes |
|-------------|---------------|---------------|
| Config / schema | `make test-domain-config` | `test_config_consolidation.py`, `test_telemetry.py`, `test_metric_registry.py`, `test_run_paths.py` |
| Env / features | `make test-domain-features` | Python feature tests; add `make test-domain-jax-env` when JAX env mirrors change |
| JAX env | `make test-domain-jax-env` | serial `-m "jax and not slow"`; parity stays in slow tier |
| Policy / PPO | `make test-domain-policy` | serial `-m "jax and not slow"` |
| Curriculum | `make test-domain-curriculum` | CPU subset of `test_curriculum.py`, `test_jax_train_timing.py` |
| Artifacts | `make test-domain-artifacts` | `test_artifact_pipeline.py`, `test_replay.py`, `test_kaggle_submission_packager.py` |
| Default agent iteration | `make test-fast` | CPU-only, serial, `-m "not slow and not jax"` |
| JAX follow-up | `make test-jax` | after edits under `src/jax/` |
| Pre-merge only | `make test` | all tests including slow; serial; ask user on WSL2 |

IDE tip: set `"python.testing.pytestArgs": ["-m", "not slow and not jax"]` in `.vscode/settings.json` so the test explorer defaults to the CPU-safe tier.

---

## Regenerating this guide

1. Run `/understand` to refresh `.understand-anything/knowledge-graph.json`.
2. Run `/understand-onboard` to regenerate this document.
3. Commit `docs/ONBOARDING.md` so the team shares the same map.

Graph stats at generation time: **691 nodes**, **1408 edges**, **115 files** analyzed, **239 paths** excluded by `.understandignore`.
