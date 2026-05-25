# Orbit Wars Agent Guide

## Repository Shape

- Python 3.12 reinforcement-learning project managed with `uv`; dependencies live in `pyproject.toml`.
- Canonical training entrypoint is Hydra-first: `uv run python -m src.train` with responsibility-group overrides.
- The runtime source is under `src/`; tests are under `tests/`; operational docs are under `docs/`.
- Hydra config is the source of truth under `conf/`. Responsibility groups include `model/`, `task/`, `reward/`, `training/`, `format/`, `opponents/`, `curriculum/`, `telemetry/`, and `artifacts/`.
- `mcp-server/` is a separate TypeScript Node project for the OMG MCP workflow server. Treat it as its own package with its own `package.json`, `tsconfig.json`, `src/`, and `test/` tree.

## Core Commands

- Install/sync Python dependencies: `uv sync --group dev`
- Run tests: see **Test Selection For Coding Agents** below; default iteration is `make test-fast`
- Run training with Hydra: `uv run python -m src.train model=attention training.total_updates=1000`
- Print resolved training config without training: `uv run python -m src.train print_resolved_config=true`
- Build MCP server: from `mcp-server/`, run `npm run build`
- Test MCP server: from `mcp-server/`, run `npm test`

## Python Subsystems

- `src/conf_schema.py` defines structured dataclass config defaults. When adding or renaming config fields, update this first.
- `src/config.py` composes and validates Hydra configs against the canonical responsibility-group schema; do not add compatibility aliases for removed runtime fields.
- `src/train.py` is a thin Hydra entrypoint that converts OmegaConf to `TrainConfig` and delegates to `src/jax_train.py`.
- `src/jax_train.py`, `src/jax_env.py`, `src/jax_features.py`, `src/jax_policy.py`, and `src/jax_ppo.py` are the JAX training path. Be careful with shape-defining config such as player count, candidate count, feature history, ship buckets, and model dimensions because these affect JIT compilation and checkpoint compatibility.
- `src/env.py` and `src/features.py` keep the Python environment/feature path. JAX and Python behavior are compared by parity tests, so mirror semantic changes across both paths when applicable.
- `src/feature_registry.py` owns ordered feature schemas and dimension checks against constants. Feature additions usually require updates in registry, encoders, JAX encoders, tests, and checkpoint compatibility expectations.
- `src/checkpoint_compat.py` validates checkpoint feature metadata and rejects checkpoints that embed the old flat runtime config shape; preserve this when changing feature dimensions or config ownership.

## Hydra And Experiment Rules

- Prefer editing `conf/` over adding ad hoc config files elsewhere. The old `configs/` layout is intentionally gone.
- Select config by responsibility group. Prefer `model=...`, `task.*`, `reward.*`, `training.*`, `format=...`, `opponents=...`, `curriculum=...`, `telemetry.*`, and `artifacts.*`.
- Use normal Hydra assignment for existing keys, such as `training.total_updates=2000`; use `+key=value` only for intentionally absent dynamic keys.
- Opponent behavior should usually be selected via `opponents=<profile>`. Avoid sweeping profile-owned fields under `opponents.self_play`, `opponents.snapshot`, or stage-local curriculum internals unless deliberately editing that profile.
- Mixed 2p/4p JAX training uses `format.rollout_groups` and curriculum stages. Do not reintroduce flat or duplicate rollout group knobs.

## Test Selection For Coding Agents

Use tiered Makefile targets. **Serial execution only.** Never run `pytest -n`, `pytest -n auto`, or install/use `pytest-xdist` — parallel JAX/CUDA workers have crashed WSL2; `tests/conftest.py` rejects xdist.

### Default rule

During implementation, run the **smallest tier that covers your edit**. Escalate only when the change crosses subsystem boundaries or you are preparing to finish/merge.

| When | Command | ~tests | Safe on WSL2? |
|------|---------|--------|---------------|
| After most edits (default) | `make test-fast` | 89 CPU | Yes |
| After JAX code edits | `make test-jax` | 18 serial JAX | Caution — single-process CUDA |
| Before claiming task complete / merge | `make test` | 156 incl. 49 slow | Slow (~15 min), serial — ask user first on WSL2 |

Do **not** run `make test` routinely while iterating. Reserve it for final verification or when the user explicitly requests full coverage.

### Pick a tier from files touched

| If you changed… | Run first | Also run when… |
|-----------------|-----------|----------------|
| `conf/`, `src/conf_schema.py`, `src/config.py`, telemetry/metrics | `make test-domain-config` | curriculum config wiring → add `make test-domain-curriculum` |
| `src/features/`, `src/feature_registry.py`, Python encoding | `make test-domain-features` | JAX mirrors in `src/jax/features.py` → add `make test-jax` or `make test-domain-policy` if shapes affect policy inputs |
| `src/game/`, `src/env.py`, `src/jax/env.py` | `make test-domain-features` + `make test-domain-jax-env` | semantic env changes → user should run full suite before merge (parity is `@pytest.mark.slow`) |
| `src/jax/policy.py`, `src/jax/ppo_update.py`, `src/jax/rollout/`, `src/jax/train_state.py`, opponents JAX actions | `make test-domain-policy` | training-loop integration → full suite before merge |
| `src/training/curriculum.py`, curriculum config | `make test-domain-curriculum` | rollout/opponent mixing → add `make test-domain-policy`; stage promotion integration is slow-tier only |
| `src/artifacts/`, replay, checkpoint paths | `make test-domain-artifacts` | checkpoint feature metadata → also `make test-domain-config` |
| `src/jax_train.py`, `src/jax/train.py` | `make test-domain-policy` + `make test-domain-curriculum` | always plan on full suite before merge |
| `mcp-server/` | `npm run build && npm test` (from `mcp-server/`) | unrelated to Python tiers |

When unsure which domain applies, fall back to `make test-fast`, then add `make test-jax` if any JAX path changed.

### Targeted single-file runs

Prefer Makefile domain targets. For a single test file after a focused fix:

```bash
uv run --group dev pytest tests/test_config_consolidation.py -m "not slow and not jax"
uv run --group dev pytest tests/test_jax_ppo.py -m "jax and not slow"
```

Do not run a slow test in isolation unless the user asked for it — e.g. avoid `-k sweep_campaign_samples_compose_full`, `-k training_loop_logs_curriculum`, or `tests/test_jax_env_parity.py` during routine agent work.

### What lives in the slow tier

`@pytest.mark.slow` (49 tests) includes expensive checks agents should **not** run by default:

- Full Hydra sweep Cartesian product (`test_wandb_sweep_campaign_samples_compose_full`)
- Full training loop (`test_training_loop_logs_curriculum_events_on_same_update`)
- JAX env parity suite (`tests/test_jax_env_parity.py`)
- End-to-end rollout/PPO smokes, curriculum rollout integration, heavy JAX env steps

These run only via `make test` (no `-m` filter). Pytest prints a yellow warning when the full suite is selected.

### Agent workflow checklist

1. **Implement** — edit the smallest surface needed.
2. **Verify** — run the matching domain target(s) from the table above; use `make test-fast` when the mapping is unclear.
3. **JAX follow-up** — if imports under `src/jax/` or JAX test files changed, run `make test-jax`.
4. **Pre-merge gate** — tell the user full suite (`make test`) is required for parity/rollout/sweep coverage; run it only with user approval on WSL2/NVIDIA hosts.
5. **Report** — cite which commands you ran; do not claim full coverage after `test-fast` alone.

### Hard prohibitions for agents

- Never use `pytest-xdist`, `-n auto`, or `-n <N>`.
- Never run `make test` as a default “let me check my work” step mid-task.
- Never run bare `uv run --group dev pytest` without understanding it executes **all 156 tests** including slow tier.
- Do not “optimize” verification by skipping tests outside your edit when the change affects shared config, feature dimensions, or JAX shapes — escalate tiers instead.

## Testing Expectations

- **Daily dev loop (CPU-safe):** `make test-fast` — `-m "not slow and not jax"`; serial only.
- **JAX quick check:** `make test-jax` — `-m "jax and not slow"`; serial only.
- **Before sharing/merging:** `make test` — full suite including `@pytest.mark.slow`; serial only; confirm with user on WSL2.
- **Domain targets:** `make test-domain-config`, `test-domain-features`, `test-domain-jax-env`, `test-domain-policy`, `test-domain-artifacts`, `test-domain-curriculum`.
- **xdist blocked:** `tests/conftest.py` raises if parallel workers are requested.

## MCP Server Notes

- `mcp-server/src/index.ts` registers tool groups for state, PRD, workflow, memory, checkpoint, model routing, bridge, and ultragoal support.
- Tests import compiled files from `mcp-server/dist/`, so run `npm run build` before `npm test` after TypeScript edits.
- Utility functions in `mcp-server/src/utils.ts` intentionally guard JSON parsing, file size, symlink reads/writes, and mode names. Preserve those safety checks when extending tools.

## OMG Workflow Bridge

- `.github/` remains the OMG source catalog for Copilot instructions, skills, agents, prompts, and hooks.
- **Cursor (native):** project config lives in `.cursor/` — rules, skills, subagents, hooks, and MCP. Regenerate from `.github/` with `uv run python scripts/sync_omg_cursor.py` after editing agents, skills, prompts, or `copilot-instructions.md`.
- **Understand-Anything (Cursor):** enable `/understand`, `/understand-dashboard`, `/understand-chat`, etc. with `bash scripts/install_understand_anything_cursor.sh` (links plugin skills from `~/.understand-anything/repo` into `.cursor/skills/`). Re-run after `sync_omg_cursor.py` if OMG skills were refreshed. Alternatively install globally via **Cursor Settings → Plugins** → `https://github.com/Lum1104/Understand-Anything`.
- **Codex:** mirrored project skills in `.agents/skills/`, custom agents in `.codex/agents/`, hooks in `.codex/hooks.json`, and the OMG MCP server from `.codex/config.toml`.
- The deep-interview → ralplan → omg-autopilot path depends on `mcp-server/dist/` being current; after TypeScript MCP edits, run `npm run build` in `mcp-server/` before relying on the workflow.

## Generated And Local Artifacts

- Ignore local training outputs and telemetry when making code changes: `outputs/`, `wandb/`, `artifacts/`, and Hydra run directories are runtime artifacts.
- `.omg/`, `.omc/`, and `.understand-anything/` may contain workflow or analysis state. Do not delete or rewrite them unless the task explicitly targets those systems.
- Spec/plan lifecycle truth lives in `.omg/workflow-manifest.json`. Before treating `.omg/specs/` or `.omg/plans/` markdown as backlog, call `omg_workflow_manifest_list(active_only=true)` or run `uv run python scripts/omg_workflow_manifest.py active`.
- Keep future guidance concise and repository-specific. Prefer adding facts here only when they affect how agents safely edit, test, or run this repo.
