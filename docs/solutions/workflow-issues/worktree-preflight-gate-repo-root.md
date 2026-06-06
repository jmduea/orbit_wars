---
title: Run preflight gates against a git worktree with --repo-root
date: 2026-06-06
category: workflow-issues
module: benchmark
problem_type: workflow_issue
component: development_workflow
severity: medium
applies_when:
  - "Cherry-pick admission runs ow train in orbit_wars-throughput-anchor or another worktree"
  - "Gate orchestration uses main's ow benchmark gate run but training code must be the worktree checkout"
  - "Hydra fails with unsafe relative output path when output.root is absolute"
tags:
  - worktree
  - preflight-gate
  - repo-root
  - output-root
  - admission-throughput
  - cherry-pick-manifest
---

# Run preflight gates against a git worktree with --repo-root

## Context

Cherry-pick admission compares **candidate training code** in a worktree (e.g. `orbit_wars-throughput-anchor` @ `throughput-baseline`) against a **learning-first baseline** captured on the pre-hygiene anchor. The worktree often lacks newer CLI subcommands (`ow benchmark gate run`, `admission-throughput`) that exist on `main`.

Running the gate from `main` with `--repo-root` keeps the **stable harness** (gate YAML, thresholds, throughput extract) on main while **`ow train` executes in the worktree** — the part that changes per cherry-pick.

## Guidance

### What runs where

| Piece | Checkout |
|-------|----------|
| `ow benchmark gate run`, gate YAML, calibration JSON | **main** |
| `ow train` subprocess (JAX train / rollout / PPO) | **worktree** (`--repo-root`) |
| `ow benchmark admission-throughput` (optional `--also-throughput`) | **main**, reads worktree `logs/*_jax.jsonl` |
| Run artifacts under `outputs/campaigns/preflight_beat_noop/` | **worktree** |

This does **not** test worktree-specific gate CLI code; it tests whether **worktree training code** learns and meets speed floors on the locked admission recipe.

### Command template

From **main** (not inside the worktree):

```bash
cd ~/projects/orbit_wars

uv run ow benchmark gate run beat_noop \
  --repo-root ~/projects/orbit_wars-throughput-anchor \
  --output-root ~/projects/orbit_wars-throughput-anchor/outputs \
  --train-overrides training=2p4p_32_split training.rollout_steps=256 task.candidate_count=3 \
    telemetry.wandb.enabled=true telemetry.wandb.group=preflight artifacts.replay.enabled=false \
  --out ~/projects/orbit_wars-throughput-anchor/outputs/benchmarks/cherry-pick/anchor_learn_proof.json \
  --also-throughput \
  --throughput-baseline docs/benchmarks/launch-hygiene-e2e-baseline-learning-first.json \
  --throughput-within-pct 10
```

Do **not** run `ow benchmark gate` inside a worktree that predates the gate subcommand — you get `invalid choice: 'gate'`.

### output.root must be repo-relative

Hydra resolver `_orbit_safe_rel` rejects **absolute** `output.root`. Passing
`output.root=/home/.../orbit_wars-throughput-anchor/outputs` fails with:

```text
InterpolationResolutionError: unsafe relative output path: '/home/.../outputs'
```

**Fix (main):** when `--repo-root` is set, emit `output.root=outputs` (relative to the worktree) while resolving filesystem paths for log discovery under the absolute `outputs/` directory.

Helpers: `_hydra_output_root()`, `_resolve_output_root()` in `src/jax/preflight.py`.

## Why This Matters

Without `--repo-root`, operators either cherry-pick the entire gate CLI onto every worktree or mistakenly believe running from main tests main training code. Without the relative `output.root` fix, the gate fails before the first update despite a correct mental model.

## When to Apply

- Nuclear cherry-pick manifest admission on `throughput-baseline` worktree.
- Any preflight gate that must exercise an older SHA's **training** path while keeping calibrated gate recipes on current `main`.
- Baseline capture on `orbit_wars-pre-hygiene` uses the same pattern (harness invokes train in that checkout).

## Examples

**Dry-run — confirm subprocess uses worktree and relative root:**

```bash
uv run ow benchmark gate run beat_noop --dry-run --verbose \
  --repo-root ~/projects/orbit_wars-throughput-anchor \
  --output-root ~/projects/orbit_wars-throughput-anchor/outputs \
  --train-overrides training=2p4p_32_split training.rollout_steps=256 task.candidate_count=3
```

Expect `output.root=outputs` in overrides and `uv run ow train` with `cwd` = worktree.

**Worktree missing gate CLI:**

```text
ow: error: invalid choice: 'gate'
```

→ Run from main with `--repo-root`, not from the worktree.

## Related

- [nuclear-cherry-pick-manifest-baseline-integration.md](nuclear-cherry-pick-manifest-baseline-integration.md)
- [cursor-before-shell-gpu-terminal-contention.md](../developer-experience/cursor-before-shell-gpu-terminal-contention.md) — light `http.server` no longer blocks gate commands
- `docs/benchmarks/cherry-pick-manifest.json` — `admission_profile`, baseline JSON path
- `docs/session-handoff/2026-06-05-cherry-pick-manifest.md` — operator steps
