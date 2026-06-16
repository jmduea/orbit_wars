---

## title: Colab train host for preflight shortlist to remote long runs

date: 2026-06-07
last_updated: 2026-06-16
category: architecture-patterns
module: orchestration
problem_type: architecture_pattern
component: tooling
severity: medium
applies_when:

- "Local W&B preflight sweeps need a GPU host for multi-hour or multi-thousand-update training"
- "Choosing between ow train kaggle and a simpler remote tarball bootstrap path"
- "Wiring a third train host into ow train alongside local and kaggle"
tags:
- colab
- remote-training
- train-host
- preflight-sweep
- google-colab-cli
- kaggle-runner-contrast
related_components:
- src/cli/train_hosts.py
- src/cli/colab_runner.py
- src/orchestration/colab_runner.py
- src/orchestration/colab_cli.py
- src/orchestration/remote_package.py
- src/orchestration/remote_worker.py
- scripts/colab_worker_entry.py
- docs/colab_runner.md
- docs/kaggle_runner.md

# Colab train host for preflight shortlist to remote long runs

## Context

Orbit Wars trains with Hydra + JAX PPO. Operators run **local W&B preflight sweeps** (`conf/wandb_sweep/fixed/preflight.yaml`) to rank hyperparameters, then need **remote GPU compute** for long training without blocking a local machine.

An earlier path used `**ow train kaggle`** (`src/orchestration/kaggle_runner.py`, `docs/kaggle_runner.md`). It was set aside because:

- Kaggle script kernels execute only `script.py`; the repo must be embedded as base64 inside that single file (`embedded-payload-v6` in `src/orchestration/kernel_package.py`).
- JAX/CUDA bootstrap on Kaggle is brittle (`src/orchestration/kaggle_jax.py`, image-specific driver discovery).
- Default hardware is P100; cold JAX compiles are slow.
- W&B-on-Kaggle is deprecated; standalone mode only.

[google-colab-cli](https://github.com/googlecolab/google-colab-cli) (0.5.9, June 2026) enables terminal-driven Colab VM lifecycle (`colab new`, `colab exec`, `colab upload/download`, `colab stop`). Colab accepts a **tarball + separate bootstrap script** — no single-file embedding — and offers T4/L4/A100 GPUs.

**Shipped:** PR [#226](https://github.com/jmduea/orbit_wars/pull/226) merged at `7f7a38c`. U0 spike and U6 operator proof passed on T4 (10 updates, `exit_code` 0). Plan: `docs/solutions/architecture-patterns/colab-train-host-preflight-long-run.md`.

## Guidance

### Train host router

`ow train` routes three hosts via `src/cli/train_hosts.py`:

| Host     | Entry                      | Remote shape               |
| -------- | -------------------------- | -------------------------- |
| `local`  | `uv run ow train …`        | N/A                        |
| `kaggle` | `uv run ow train kaggle …` | Embedded kernel package    |
| `colab`  | `uv run ow train colab …`  | Tarball + Python bootstrap |

Colab subcommands: `preflight`, `prepare`, `launch`, `status`, `sync`, `shortlist`, `stop`, `monitor`.

For multi-hour runs, use **`--monitor-after-launch`** (poll sync, stale detection, local checkpoint eval) — see [`colab-long-run-monitor-sync-recovery.md`](../workflow-issues/colab-long-run-monitor-sync-recovery.md).

### Shared packaging layer

Colab reuses shared remote-worker primitives instead of duplicating Kaggle kernel logic:

- `**src/orchestration/remote_package.py`** — renders `orbit_wars.tgz` + `worker-env.json` (`REMOTE_PACKAGE_SOURCE_MODE = "remote-tarball-v1"`). Includes `src/`, `conf/`, `pyproject.toml`, `uv.lock`, and `data/jax_map_pool/` when present.
- `**src/orchestration/remote_worker.py**` — bootstrap helpers shared by Colab and Kaggle workers.
- `**scripts/colab_worker_entry.py**` — remote entry: `uv sync --group dev`, JAX GPU check, `uv run ow train …`, writes `worker-summary.json`.

Orchestration: `src/orchestration/colab_runner.py`, `src/orchestration/colab_cli.py`. CLI dispatch: `src/cli/colab_runner.py`.

### Operator workflow (canonical)

Preflight stays **local**; Colab is for long runs after shortlist selection:

```bash
# 1. Local W&B preflight (unchanged)
uv run ow sweep create --backend wandb \
  --sweep-yaml conf/wandb_sweep/fixed/preflight.yaml
wandb agent <entity>/orbit_wars/<sweep_id>

# 2. Shortlist winner
uv run ow train colab shortlist --sweep-id <sweep_id> \
  --out outputs/colab_runner/shortlist.json

# 3. Long run on Colab GPU (see workflow-issues/colab-long-run-monitor-sync-recovery.md for fixed-path recipe)
uv run ow train colab launch \
  --from-shortlist outputs/colab_runner/shortlist.json --rank 0 \
  --gpu T4 --timeout 86400 \
  --monitor-after-launch \
  --interval-seconds 300 --stale-seconds 900 \
  training.total_updates=2000 \
  output.campaign=colab_long \
  training=2p_32 \
  curriculum=scripted_heavy

# 4. Poll and pull artifacts locally
uv run ow train colab status --session <slug>
uv run ow train colab sync --session <slug>
uv run ow train colab stop --session <slug>
```

Hydra overrides after Colab flags use the same token rules as `ow train kaggle` — pass as **separate CLI arguments**, not one space-joined string.

Synced outputs land under `outputs/colab_runner/synced/<campaign>/` (checkpoints + `logs/*_jax.jsonl`). Inspect with `ow runs show --run outputs/colab_runner/synced/<campaign>/runs/<run_id>`.

### Colab CLI integration details (U6 fixes)

These landed during operator proof and are required for reliable automation:

| Issue                                | Fix                                                                            |
| ------------------------------------ | ------------------------------------------------------------------------------ |
| `colab exec` without session context | All `colab upload/download/exec/stop/status` use `--session` flags (CLI 0.5.9) |
| Shell bootstrap quoting failures     | Bootstrap is **Python** (uploaded via `colab exec -f`), not a shell heredoc    |
| Directory download unsupported       | `sync` archives the remote campaign dir to a tarball before `colab download`   |

### Scope boundaries

- **Colab does:** remote training compute (checkpoints, JSONL logs).
- **Colab does not:** eval, Docker packaging validation, tournament ladders, or Kaggle submit — those stay **local** after `sync`.
- **v1:** one Colab session at a time; sync before `stop` when checkpoints matter.
- **Preflight sweeps:** never run `wandb agent` on Colab in v1.

### Kaggle runner contrast

| Dimension     | `ow train kaggle`                     | `ow train colab`                                                    |
| ------------- | ------------------------------------- | ------------------------------------------------------------------- |
| Payload       | Base64 embedded in single `script.py` | Tarball + `worker-env.json`                                         |
| Bootstrap     | Kaggle image + `kaggle_jax.py`        | `uv sync --group dev` + JAX GPU check                               |
| Default GPU   | P100                                  | T4 (configurable)                                                   |
| W&B on remote | Deprecated / standalone only          | Optional via `worker-env.json` (`WANDB_API_KEY`)                    |
| Shortlist     | `ow train kaggle shortlist`           | `ow train colab shortlist` (same `wandb_sweeps.shortlist_from_api`) |
| Status        | Kaggle kernel slug                    | Colab session slug (`ow-colab_<campaign>-<sha>`)                    |

Kaggle runner remains in the repo for diagnostics; Colab is the preferred remote long-run host after preflight.

## Why This Matters

Without a tarball-based remote host, operators either block local GPUs for days or fight Kaggle's single-file embedding and brittle JAX bootstrap. Colab keeps the **same Hydra override surface** as local training while separating concerns: W&B ranks configs locally; Colab runs the winning recipe; local tooling runs gates and submit.

Re-implementing packaging per host duplicates `remote_package.py` and drifts worker behavior. The shared layer ensures map-pool data, lockfiles, and bootstrap diagnostics stay consistent.

## When to Apply

- After a local W&B preflight sweep when `training.total_updates` or wall time exceeds comfortable local GPU budget.
- When evaluating remote training backends — prefer Colab unless Kaggle kernel constraints are explicitly required.
- When extending train hosts — add subcommands to `train_hosts.py`, reuse `remote_package.py` / `remote_worker.py`, emit JSON on stdout for agent loops (human progress on stderr).
- **Not** for preflight sweeps, admission gates, or submit-valid proof — those run locally.

## Examples

### Prerequisites

```bash
uv tool install google-colab-cli
colab auth
uv run ow train colab preflight
```

### Smoke proof (U6, PASS 2026-06-07)

```bash
uv run ow train colab launch --gpu T4 --timeout 7200 \
  training.total_updates=10 curriculum=noop_only output.campaign=colab_smoke \
  task=shield_cheap \
  telemetry.wandb.enabled=false
uv run ow train colab sync --session ow-colab_smoke-12c2f68
```

Results: `exit_code` 0, cold update 1 `rollout_s≈70s`, steady-state update 10 `rollout_s≈1.5s`, checkpoints synced to `outputs/colab_runner/synced/colab_smoke/`.

### Anti-patterns

**Wrong:** Run preflight W&B sweep on Colab to "save local GPU."

**Right:** Local `wandb agent` on preflight YAML → `ow train colab shortlist` → `ow train colab launch`.

**Wrong:** Pipe `ow train colab launch` through `tail`/`head` — hides progress until exit.

**Right:** JSON on stdout; watch stderr or use `ow train colab status`.

## Related

- Long-run operations (monitor, sync tolerance, recovery, recipe selection): [`../workflow-issues/colab-long-run-monitor-sync-recovery.md`](../workflow-issues/colab-long-run-monitor-sync-recovery.md)
- Operator reference: `[docs/colab_runner.md](../../colab_runner.md)`
- Kaggle contrast: `[docs/kaggle_runner.md](../../kaggle_runner.md)`
- Implementation plan: `[docs/solutions/architecture-patterns/colab-train-host-preflight-long-run.md](../architecture-patterns/colab-train-host-preflight-long-run.md)`
- SSOT spine (local preflight → packaging → long train): `[ssot-training-pipeline-config-to-kaggle-submission.md](ssot-training-pipeline-config-to-kaggle-submission.md)` — Colab is an alternate **long-train host** for step 5, not a replacement for local gates/submit
- Agent-native CLI patterns: `[../developer-experience/agent-native-operator-cli-phase1.md](../developer-experience/agent-native-operator-cli-phase1.md)`
