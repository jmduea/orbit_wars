---
title: Long ow CLI jobs — stderr progress and no tail/head pipes
date: 2026-06-03
category: developer-experience
module: cli
problem_type: workflow_issue
component: development_workflow
severity: medium
applies_when:
  - "Running ow benchmark gate run, tournament-proof, or calibration for many minutes"
  - "Agents or operators pipe ow output to tail, head, or 2>&1 | tail"
  - "stdout must remain machine-readable JSON for --out paths"
symptoms:
  - "Terminal shows no output until the subprocess exits"
  - "Agent assumes the job hung and interrupts a healthy GPU run"
tags:
  - ow-cli
  - stderr
  - benchmark-progress
  - agent-workflow
  - observability
related_components:
  - src/jax/benchmark_progress.py
  - src/jax/preflight.py
  - src/jax/preflight_calibration.py
  - src/cli/benchmark.py
---

# Long ow CLI jobs — stderr progress and no tail/head pipes

## Context

Gate runs, tournament proof, and calibration subprocesses can run tens of minutes (JAX compile, Docker validation, `kaggle_environments` matches). Piping `ow` stdout through `tail` or `head` buffers output until the pipe closes, so the session looks frozen. PR [#186](https://github.com/jmduea/orbit_wars/pull/186) adds **`emit_benchmark_progress`** on **stderr** so humans and agents see liveness while **stdout** stays clean for `--out` JSON.

This complements subprocess **train** streaming in `docs/solutions/developer-experience/benchmark-subprocess-training-observability.md` (child `ow train` lines). Here the focus is **top-level benchmark/gate commands** and anti-patterns for piping.

## Guidance

### Stream progress on stderr, JSON on stdout

```python
# src/jax/benchmark_progress.py
def emit_benchmark_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
```

Preflight gates and calibration arms call this before/after `ow train` subprocesses (`src/jax/preflight.py`, `src/jax/preflight_calibration.py`). Gate and tournament-proof CLIs accept **`--verbose`** to surface stage banners on stderr.

### Do not pipe ow through tail or head

**Avoid:**

```bash
uv run ow benchmark gate run curriculum_staged 2>&1 | tail -20   # buffers; looks hung
uv run ow benchmark tournament-proof ... | tail -f                  # same
```

**Prefer:**

```bash
uv run ow benchmark gate run curriculum_staged --verbose --out /tmp/gate4.json
uv run ow benchmark tournament-proof --eval-checkpoint ... --verbose --out /tmp/gate5.json
uv run ow runs watch --run outputs/campaigns/<c>/runs/<id>
uv run ow eval status --run <path> --watch
tail -f outputs/campaigns/<c>/runs/<id>/logs/*_jax.jsonl   # artifacts only, not ow stdout
```

### When W&B is off, still prove liveness

For training inside benchmarks, keep `training.log_every=1` on calibration bases and use `run_ow_train()` streaming (see benchmark-subprocess doc). For gate/tournament wrappers, rely on **`--verbose`** stderr lines and artifact tails, not stdout truncation.

## Why This Matters

Agents default to `| tail` for long commands; with Python block buffering and pipe backpressure, **no lines appear until exit**, triggering false "hung" interrupts on expensive GPU work. Separating stderr progress from stdout JSON keeps automation parseable without sacrificing operator feedback.

## When to Apply

- Documenting operator steps in `AGENTS.md`, `docs/AGENT_CAPABILITIES.md`, or runbooks
- Adding new `ow benchmark` subcommands that run >2 minutes
- Debugging "silent" gate or tournament-proof runs in Cursor/automation

## Examples

**Before:** `ow benchmark gate run win_proof_tournament` with no flags and stdout piped to `tail -5` — silent until Docker + ladder complete.

**After:** `--verbose --out /tmp/gate5.json` — stderr shows Docker gate start, tournament stage, timestamps; stdout file receives final JSON only at end.

## Related

- Train subprocess streaming: `docs/solutions/developer-experience/benchmark-subprocess-training-observability.md`
- Gate 5 funnel (uses same operator habits): `docs/solutions/architecture-patterns/gate5-unified-tournament-submit-valid-funnel.md`
- `tests/test_benchmark_progress.py`
