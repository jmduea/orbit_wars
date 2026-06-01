---
title: Benchmark calibration subprocesses looked hung with no terminal progress
date: 2026-06-01
category: developer-experience
module: jax-training
problem_type: developer_experience
component: development_workflow
severity: medium
applies_when:
  - "Running long GPU sweeps via ow benchmark that spawn ow train subprocesses"
  - "W&B disabled for calibration or preflight benchmark arms"
  - "First JAX compile on WSL2 GPU before update 1 completes"
symptoms:
  - "calibrate-seed-scheduler prints only CUDA driver warnings then silence for many minutes"
  - "No way to tell if training is progressing vs hung without inspecting run artifacts"
  - "discover picks newest run dir by mtime including failed runs with empty JSONL"
tags:
  - calibration
  - benchmark
  - observability
  - subprocess
  - seed-scheduler
  - log-every
  - jax-compile
related_components:
  - src/jax/preflight_calibration.py
  - src/jax/seed_scheduler_calibration.py
  - src/jax/train/loop.py
  - src/cli/benchmark.py
---

# Benchmark calibration subprocesses looked hung with no terminal progress

## Context

GPU calibration sweeps (`ow benchmark calibrate-seed-scheduler`, preflight calibration) spawn `ow train` as a subprocess for each arm. W&B is intentionally off (`telemetry.wandb.enabled=false`) to keep sweeps lightweight. On WSL2, the first update can take several minutes while JAX compiles; without explicit terminal progress, operators assume the process is hung and interrupt it (session history).

Direct `ow train` already gained `orbit_train_start` / per-update lines in PR #164, but **benchmark harnesses** used a separate code path that did not surface child output.

## Guidance

Long-running benchmark training must provide **either W&B telemetry or live terminal progress**. For subprocess-driven sweeps, implement all of the following:

### 1. Stream the child process

Replace fire-and-forget `subprocess.run()` with line-buffered streaming and unbuffered Python:

```python
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    env=env,
)
for line in proc.stdout:
    print(line, end="", flush=True)
```

Print a labeled banner and full command before launch so it is obvious which arm is running.

Canonical implementation: `run_ow_train()` in `src/jax/preflight_calibration.py` (shared by preflight gates, preflight calibration, and seed-scheduler calibration).

### 2. Force per-update logging for benchmark arms

Workstation profile defaults to `log_every: 10`. Calibration bases should override:

```yaml
training.log_every=1
```

In `SEED_SCHED_TRAIN_BASE` and `PREFLIGHT_TRAIN_BASE` so the child prints a progress line every update once the loop starts.

### 3. Startup banner when W&B is off

In `run_jax_training`, print campaign, run id, update range, `log_every`, log path, and a note that **update 1 may stall during JAX compile**. This sets expectations before the first metric line.

### 4. Discover completed runs, not empty failures

`latest_run_dir()` picks newest mtime, including interrupted runs with empty `logs/*_jax.jsonl`. Use `latest_completed_run_dir()` — newest run whose JSONL is non-empty — when discovering calibration campaigns for analysis.

### 5. Escape hatch while debugging

If you cannot restart with the fix yet, progress is still visible in artifacts:

```bash
tail -f outputs/campaigns/<campaign>/runs/<run_id>/logs/*_jax.jsonl
```

Benign WSL2 stderr such as `Could not get kernel mode driver version` from JAX GPU init is **not** a hang indicator.

## Why This Matters

A 500-update calibration arm can take hours. Silence through compile plus `log_every=10` means **50+ minutes** with no terminal output even when training is healthy. Operators Ctrl+C (exit 130), leaving partial campaigns that break discover/analysis.

Separately, treating CUDA warnings as failure sends debugging down the wrong path while the real issue is observability.

## When to Apply

- Adding any new `ow benchmark` sweep that calls `run_ow_train()`
- Disabling W&B for batch benchmark or CI training smokes
- Reviewing operator UX for GPU jobs on WSL2 or headless agents

Do **not** assume inherited stdout from `subprocess.run()` is enough — the child may print nothing until `update % log_every == 0`.

## Examples

**Before (looked hung):**

```text
E0601 ... cuda_executor.cc:1526] Could not get kernel mode driver version ...
[silence for 5–50+ minutes]
```

**After (PR #165):**

```text
Seed-scheduler calibration: 3 training arm(s), 500 updates each, output_root=outputs

=== seed-scheduler calibration arm 1/3 opponent=self_play_only reseed=25 ... ===
uv run ow train training=workstation ...
JAX training starting: campaign=... updates=1-500 log_every=1 wandb=off log=...
Terminal progress: one line per log_every update(s). First update may stall during JAX compile.
update=1 steps=4096 ... sps=...
update=2 ...
```

**Verify a run is alive without terminal fix:**

```bash
wc -l outputs/campaigns/seed_sched_cal_*/runs/*/logs/*_jax.jsonl
```

## Related

- Fix: PR [#165](https://github.com/jmduea/orbit_wars/pull/165) (`aa3686c`)
- Active calibration plan: `docs/plans/2026-06-01-003-feat-seed-scheduler-calibration-plan.md`
- Direct `ow train` operator lines (different path): `docs/solutions/developer-experience/agent-native-operator-cli-phase1.md` § Training observability
- Unrelated throughput topic: `docs/solutions/performance-issues/launch-hygiene-incremental-carry-throughput.md`
