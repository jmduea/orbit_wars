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
- Print resolved training config without training: `uv run ow train print_resolved_config=true`
- Launch Kaggle training (standalone): `uv run ow train kaggle format=mix_2p_4p_16env`
- Kaggle ops: `uv run ow train kaggle preflight|status|sync`
- Legacy Kaggle script: `uv run python scripts/kaggle_runner.py` (deprecated: `kaggle_wandb_population.py`)
- Build MCP server: from `mcp-server/`, run `npm run build`
- Test MCP server: from `mcp-server/`, run `npm test`

## Python Subsystems

- `src/config/schema.py` defines structured dataclass config defaults. When adding or renaming config fields, update this first, then `conf/` overrides.
- `src/config/runtime.py` composes and validates Hydra configs into `TrainConfig`; do not add compatibility aliases for removed runtime fields. Dataclass defaults can differ from `conf/` YAML (e.g. `TaskConfig.candidate_count`); treat resolved Hydra config as runtime truth.
- `src/train.py` is a thin Hydra entrypoint that delegates to `src/jax/train.py`.
- JAX training lives under `src/jax/`: `train.py` (loop shell), `env.py`, `features.py`, `policy.py`, `ppo_update.py`, and `rollout/` (collect + metrics). Opponent action builders are in `src/opponents/jax_actions/`. Shape-defining config (player count, candidate count, feature history, ship buckets, model dimensions) affects JIT compilation and checkpoint compatibility.
- Feature encoding is JAX-only (planet-edge schema): `src/jax/features.py` implements `encode_turn`; `src/features/registry.py` owns feature schemas and dims; `src/features/extractor.py` is the shared entry point for Kaggle obs and JAX env (coerces via `jax_game_from_observation`). Do not reintroduce v1 self/candidate/global encoders or `_v2` suffix modules.
- Python game logic remains in `src/game/`; optional Python opponent inference uses `src/opponents/runtime.py`.
- `src/artifacts/checkpoint_compat.py` validates checkpoint feature metadata; preserve compatibility checks when changing feature dimensions or config ownership.
- Microbatched rollouts (`training.rollout_microbatch_envs < num_envs`) merge metrics in `jax.lax.scan`: use `_merge_metric_dicts` inside the scan (stable key set) and `_finalize_cross_chunk_rate_metrics` after — never add derived rate keys inside the scan body.

## Hydra And Experiment Rules

- Prefer editing `conf/` over adding ad hoc config files elsewhere. The old `configs/` layout is intentionally gone.
- Select config by responsibility group. Prefer `model=...`, `task.*`, `reward.*`, `training.*`, `format=...`, `opponents=...`, `curriculum=...`, `telemetry.*`, and `artifacts.*`.
- Use normal Hydra assignment for existing keys, such as `training.total_updates=2000`; use `+key=value` only for intentionally absent dynamic keys.
- Opponent behavior should usually be selected via `opponents=<profile>`. Avoid sweeping profile-owned fields under `opponents.self_play`, `opponents.snapshot`, or stage-local curriculum internals unless deliberately editing that profile.
- Mixed 2p/4p JAX training uses `format.rollout_groups` and curriculum stages. Do not reintroduce flat or duplicate rollout group knobs.

## Test Selection For Coding Agents

Use tiered Makefile targets. **Serial execution only.** Never run `pytest -n`, `pytest -n auto`, or install/use `pytest-xdist` — parallel JAX/CUDA workers have crashed WSL2; `tests/conftest.py` rejects xdist.

### Default rule

During implementation, run **`make test-fast` only**. Do not run JAX compilation, rollout, or training-smoke tests while iterating — they dominate wall time on WSL2/CUDA hosts.

| When | Command | ~tests | Safe on WSL2? |
|------|---------|--------|---------------|
| After most edits (default) | `make test-fast` | CPU-only | Yes |
| After JAX edits (optional, user-requested) | `make test-jax` | ~5 lightweight JAX | Caution |
| Before merge / user asks | `make test` | full incl. slow | Slow (~15 min); ask first |

Do **not** run `make test` or `make test-jax` routinely while iterating. JAX env dispatch, rollout collect, PPO update smokes, JIT/vmap encode, and training-loop tests live in the **slow** tier and run only via `make test` (or when the user explicitly asks).

### Pick a tier from files touched

| If you changed… | Run first | Also run when… |
|-----------------|-----------|----------------|
| `conf/`, `src/config/schema.py`, `src/config/runtime.py`, telemetry/metrics | `make test-domain-config` | curriculum config wiring → `make test-domain-curriculum` |
| `src/features/`, `src/jax/features.py`, registry, extractor | `make test-domain-features` | user asks for JAX compile coverage |
| `src/game/`, `src/jax/env.py` | `make test-domain-features` | user asks; env step smokes are slow-tier |
| `src/jax/policy.py`, `src/jax/ppo_update.py`, `src/jax/rollout/`, `src/opponents/jax_actions/` | `make test-domain-policy` or `make test-fast` | user asks for rollout/training smokes before merge |
| `src/training/curriculum.py`, curriculum config | `make test-domain-curriculum` | training-loop integration is slow-tier only |
| `src/artifacts/`, replay, checkpoint paths | `make test-domain-artifacts` | also `make test-domain-config` when metadata changes |
| `src/jax/train.py` | `make test-fast` | full suite before merge, user approval only |
| `mcp-server/` | `npm run build && npm test` (from `mcp-server/`) | unrelated to Python tiers |

When unsure which domain applies, use `make test-fast` only.

### Targeted single-file runs

Prefer Makefile domain targets. For a focused fix without JAX compile:

```bash
uv run --group dev pytest tests/test_config_consolidation.py -m "not slow and not jax"
uv run --group dev pytest tests/test_feature_encoding_golden.py -m "not slow and not jax"
```

Do not run slow-tier tests in isolation unless the user asked — e.g. `tests/test_jax_rollout.py`, `tests/test_jax_curriculum.py`, `test_encode_v2_jit_vmap_smoke`, or `tests/test_jax_env_parity.py`.

### What lives in the slow tier

`@pytest.mark.slow` includes expensive checks agents should **not** run during iteration:

- Full Hydra sweep Cartesian product (`test_wandb_sweep_campaign_samples_compose_full`)
- JAX env parity suite (`tests/test_jax_env_parity.py`)
- **JAX compile / training smokes:** `tests/test_jax_rollout.py`, `tests/test_jax_curriculum.py`, `tests/test_jax_scripted_opponents.py`, batched env reset/step, end-to-end rollout+PPO, JIT/vmap encode smoke
- Full training loop and heavy rollout integration in `test_jax_ppo.py`

These run only via `make test` (no `-m` filter). Pytest prints a yellow warning when the full suite is selected.

### Agent workflow checklist

1. **Implement** — edit the smallest surface needed.
2. **Verify** — run `make test-fast` (or the matching domain target if it is also CPU-only).
3. **Do not** run `make test-jax` or rollout/training smokes unless the user explicitly requests JAX compile coverage.
4. **Pre-merge gate** — tell the user full suite (`make test`) covers rollout/training/JIT; run only with user approval on WSL2/NVIDIA hosts.
5. **Report** — cite which commands you ran; do not claim compile/training coverage after `test-fast` alone.

### Hard prohibitions for agents

- Never use `pytest-xdist`, `-n auto`, or `-n <N>`.
- Never run `make test` or `make test-jax` as a default “let me check my work” step mid-task.
- Never run bare `uv run --group dev pytest` without understanding it executes **all** tests including slow/JAX-compile tier.
- Do not run JAX rollout, PPO update, training-loop, or JIT/vmap smokes during routine agent work.

## Testing Expectations

- **Daily dev loop (CPU-safe):** `make test-fast` — `-m "not slow and not jax"`; serial only.
- **JAX lightweight (user-requested only):** `make test-jax` — metric/action-builder checks; no rollout/training smokes.
- **Before sharing/merging:** `make test` — full suite including JAX compile/training smokes; user approval on WSL2.
- **Domain targets:** `make test-domain-config`, `test-domain-features`, `test-domain-policy`, `test-domain-artifacts`, `test-domain-curriculum` (CPU-only where noted).
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

## Human Roadmap (single funnel)

- **`docs/ROADMAP.md`** is the only priority index (≤3 **Now**, ≤5 **Done**). **`docs/brain_dump.md` is retired** — do not capture or triage there.
- **Free-form chat:** users need not invoke `/work-intake`. On any implementation request, run **`roadmap.py begin "<user message>"`** first, then obey `may_implement` / `next_steps` → planning if needed → `claim` → `approve-impl` → code → `wrap-up` + `check-session --require-clean`.
- **Enforcement:** Cursor pre-tool hook blocks `src/`, `conf/`, `tests/` edits without `.omg/state/impl-gate.json`. `ORBIT_WARS_IMPL_GATE` defaults to **on** (`scripts/agent_env.sh`).
- **After editing ROADMAP:** `make roadmap-check`. **Human Now wins** over manifest backlog.
- **New work:** add **Later** row first; open GitHub issues after phase 3 (execution plan), not at idea time.

- **Multi-agent:** one `roadmap.py claim --issue N --path …` per implementing agent; `roadmap.py claims` before work; **before push:** move closed work to ROADMAP **Done** and run `make roadmap-check`; `wrap-up --issue N --evidence` after `gh issue close` (requires **Done** row + CLOSED + evidence); `check-session` before stopping.
- Set `ORBIT_WARS_AGENT_ID` per Cursor session (e.g. `cursor-a`, `cursor-b`) to avoid claim collisions.
- **Agent packages:** `.omg/workflow-manifest.json` active entries only; register/link when promoting to **Now**.

## Generated And Local Artifacts

- Ignore local training outputs and telemetry when making code changes: `outputs/`, `wandb/`, `artifacts/`, and Hydra run directories are runtime artifacts.
- `.omg/`, `.omc/`, and `.understand-anything/` may contain workflow or analysis state. Do not delete or rewrite them unless the task explicitly targets those systems.
- Spec/plan lifecycle truth lives in `.omg/workflow-manifest.json`. Before treating `.omg/specs/` or `.omg/plans/` markdown as backlog, call `omg_workflow_manifest_list(active_only=true)` or run `uv run python scripts/omg_workflow_manifest.py active`.
- Keep future guidance concise and repository-specific. Prefer adding facts here only when they affect how agents safely edit, test, or run this repo.

## Learned User Preferences

- Prefer unified v2-only feature encoding and module layout; remove legacy v1 paths rather than maintaining parallel encoders.
- Favor full rework over incremental shims when simplifying encoding, rollout, or training modules.
- Daily dev loop: `make test-fast` or a domain Makefile target — not bare full `pytest` and not slow/JAX-compile smokes unless explicitly requested.
- Never use `pytest-xdist` or parallel pytest workers on WSL2/CUDA hosts.
- Before treating `.omg/specs/` or `.omg/plans/` markdown as active backlog, consult `.omg/workflow-manifest.json` (or `omg_workflow_manifest_list(active_only=true)`).
- Commit verified work locally without asking; **do not push** to remote unless the user explicitly requests it.
- Do not start test runs (`make test-fast`, domain targets, or `pytest`) when another agent/session is already running tests, or when the user says verification is already done — check the terminals folder first.

## Learned Workspace Facts

- Canonical feature path: Kaggle/JAX obs → `FeatureExtractor` → `encode_turn` (planet-edge `TurnBatch`); golden tests live in `tests/test_feature_encoding_golden.py`.
- JAX concerns are split: rollout collection in `src/jax/rollout/collect.py`, PPO update in `src/jax/ppo_update.py`, opponent builders in `src/opponents/jax_actions/`.
- `model.normalize_observations` appears in model YAMLs but is not wired into JAX training; treat as dead config until implemented or removed.
- Hydra dataclass defaults in `src/config/schema.py` can differ from `conf/` YAML; verify with `uv run python -m src.train print_resolved_config=true`.
- Understand-Anything scans honor `.understandignore` for excluding non-project adjacent paths.
- OMG Cursor config is generated from `.github/` via `uv run python scripts/sync_omg_cursor.py`; re-run after editing agents, skills, or `copilot-instructions.md`.
