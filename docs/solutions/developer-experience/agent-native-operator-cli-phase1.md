---
title: Agent-native operator CLI phase 1 (runs, eval status, session context)
date: 2026-06-01
category: developer-experience
module: cli
problem_type: developer_experience
component: tooling
severity: medium
applies_when:
  - "Coding agents need to inspect outputs/campaigns without manual path spelunking"
  - "Agent-native audit scored CRUD and context injection low"
  - "Training or artifact pipeline changes are silent in the terminal"
tags:
  - agent-native
  - ow-cli
  - operator-introspection
  - outputs-campaigns
  - make-agent-context
related_components:
  - src/cli/runs.py
  - src/cli/run_status.py
  - scripts/agent_context.py
  - docs/AGENT_CAPABILITIES.md
---

# Agent-native operator CLI phase 1 (runs, eval status, session context)

## Context

A 2026-06-01 agent-native audit scored Orbit Wars ~87% action parity (same `ow` CLI as humans) but **0/8 CRUD**, weak dynamic context, and silent async artifact work. Agents could train and benchmark but lacked list/show/status commands and session-scoped threshold/roadmap context.

Phase 1 shipped in PR [#164](https://github.com/jmduea/orbit_wars/pull/164) on `main` (merge `0844515`). Feature branches created before that merge need `git merge main` or rebase to get `ow runs`, `make agent-context`, and related files. Phase 2 benchmark gates and Phase 3 refactors (YAML gate loader, `ow benchmark gate run`, `ow sweep`) shipped on `main` — see `docs/audits/agent-native-status.md`. Post-#184 operator CRUD and seed interval **50** are in `docs/solutions/developer-experience/seed-scheduler-calibration-agent-native-operator-phase2.md`.

Parallel work during that PR used stash + branch isolation — see `docs/solutions/workflow-issues/git-stash-recovery-after-parallel-branch-cleanup.md`.

## Guidance

### Session context (run once per agent session)

```bash
make agent-context
```

Emits JSON: preflight threshold excerpt from `docs/benchmarks/preflight-calibration.json`, ROADMAP Now/Next, recent `outputs/indexes/runs.jsonl` rows, doc pointers. Implemented in `scripts/agent_context.py` (no JAX import).

### Inspect campaign runs

```bash
uv run ow runs list --limit 10
uv run ow runs show --run outputs/campaigns/<campaign>/runs/<run_id>
uv run ow runs logs --run outputs/campaigns/<campaign>/runs/<run_id> --tail 5
```

Filesystem-backed: reads `manifest.json`, tails `logs/*_jax.jsonl`. Dispatch: `ow runs` in `src/cli/__init__.py` → `src/cli/runs.py`.

### Artifact queue and promotion snapshot

```bash
uv run ow eval status --run outputs/campaigns/<c>/runs/<id>
uv run ow eval worker --run <path> --verbose
```

`status` summarizes `queue/optional_jobs/*.json`, promotion manifest path, last JSONL marker. `--verbose` prints per-job start/done in `src/artifacts/worker_runner.py`.

### Discovery surfaces

| Surface | Role |
|---------|------|
| `docs/AGENT_CAPABILITIES.md` | Task prompts + config-vs-code boundary |
| `make help` | Test tiers, preflight shortcuts, `agent-context` |
| `uv run ow --help` | Links eval/benchmark/runs help + agent doc |
| Empty `ow eval` / `ow benchmark` | Subcommand menu (exit 0) |

### Training observability

Local `ow train` prints:

- `orbit_train_start` — `run_dir`, `log_path`, `queue_dir`, wandb on/off (`src/jax/train/loop.py`)
- `orbit_train_complete` — final update, log path, `jax_ckpt_last.pkl` hint
- `artifact_worker_started` — paths to `queue/worker.stdout.log` when autostart fires

JSONL under `logs/*_jax.jsonl` remains the durable metric source; terminal lines are a sampled view.

Benchmark sweeps that subprocess `ow train` (`calibrate-seed-scheduler`, preflight calibration) use a separate harness — see `docs/solutions/developer-experience/benchmark-subprocess-training-observability.md`.

## Why This Matters

Agents and humans already shared `outputs/campaigns/` (high shared-workspace score). The gap was **operator ergonomics**: no first-class read/list/status without knowing layout. Phase 1 closes the highest-leverage introspection holes without refactoring `ow benchmark learn-proof` into primitives.

## When to Apply

- Debugging a train run: start with `ow runs show` + `ow eval status`, not blind `tail` guesses.
- Onboarding a coding agent: point to `AGENT_CAPABILITIES.md` + `make agent-context`.
- After hybrid promotion (`artifacts=hybrid_promotion`): use `eval status` and `worker --verbose` to see queued tournament/docker jobs.

## Examples

**Agent prompt: inspect hybrid promotion**

```text
Run `uv run ow eval status --run outputs/campaigns/<c>/runs/<id>` and summarize
queued/running jobs. If worker autostarted, read queue/worker.stderr.log.
```

**Verify train smoke left artifacts**

```text
Run a 5-update smoke, then `uv run ow runs list` and confirm the new run_dir
has logs/*_jax.jsonl and orbit_train_complete on stdout.
```

## Related Issues

- Audit follow-up plan: `docs/plans/2026-06-01-agent-native-operator-cli-plan.md`
- Phase 2+3 shipped: `docs/solutions/developer-experience/seed-scheduler-calibration-agent-native-operator-phase2.md`, `docs/audits/agent-native-status.md`
- GPU shell enforcement (project hooks): `docs/solutions/developer-experience/cursor-before-shell-gpu-terminal-contention.md`
- Benchmark CLI package split (PR #202): `docs/solutions/architecture-patterns/benchmark-cli-package-split-agent-native-parity.md`
