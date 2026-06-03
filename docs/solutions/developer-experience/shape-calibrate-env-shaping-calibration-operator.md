---
title: Env shaping calibration via ow benchmark shape-calibrate
date: 2026-06-03
category: developer-experience
module: jax-training
problem_type: developer_experience
component: development_workflow
severity: medium
applies_when:
  - "Choosing or changing joint MDP shaping (reward profile, training opponents, reseed_every_updates) before long GPU trains"
  - "Running or re-analyzing shape_cal_* calibration campaigns on one GPU"
  - "Implementing or reviewing the shape-calibrate operator (planned; see docs/plans/2026-06-03-003-feat-shape-calibrate-plan.md)"
tags:
  - shape-calibrate
  - env-shaping
  - shaping-calibration
  - evaluate-gate-records
  - dual-contract
  - calibrate-seed-scheduler
  - preflight-calibration
  - reward-opponent-reseed
  - agent-native
  - ow-cli
related_components:
  - src/jax/shaping_calibration.py
  - src/jax/seed_scheduler_calibration.py
  - src/jax/preflight.py
  - src/cli/benchmark/
  - docs/benchmarks/shaping-calibration.json
  - docs/benchmarks/preflight-calibration.json
---

# Env shaping calibration via `ow benchmark shape-calibrate`

## Context

Orbit Wars spreads MDP shaping across Hydra groups (`conf/reward/`, `conf/opponents/`, `training.reseed_*`, task/shield, curriculum). The repo already has an **inner loop** (PPO on a chosen bundle) and partial **outer-loop** tooling: `ow benchmark calibrate-seed-scheduler` pins reseed interval from held-out eval; preflight Gates 2–4 read training JSONL trends; Gate 5 / hybrid `checkpoint_eval` prove submit-valid performance on held-out opponents.

What was missing in this session was a single **measure → decide → pin** operator for **joint** shaping (reward × opponents × reseed), with an explicit split between **training MDP** fitness and **reference MDP** proof—the ICML auto-env-shaping pattern (train on shaped env, score on unshaped or held-out reference).

This doc captures the **design** agreed in ideation, brainstorm, and plan (`docs/ideation/2026-06-03-searchable-measurable-env-shaping-ideation.md`, `docs/brainstorms/2026-06-03-shape-calibrate-requirements.md`, `docs/plans/2026-06-03-003-feat-shape-calibrate-plan.md`). Re-run `/ce-compound` after implementation ships to add verified CLI examples.

## Guidance

### Mirror `calibrate-seed-scheduler`, extend the search space

Use `src/jax/seed_scheduler_calibration.py` as the template:

- Campaign prefix `shape_cal_<reward>_<opponent>_reseed<N>_u<updates>` under `outputs/campaigns/`
- `run_ow_train` from `src/jax/preflight_calibration.py` (streaming subprocess, `training.log_every=1`)
- `discover_*_runs` + `latest_completed_run_dir` for `--analyze-only`
- Output `docs/benchmarks/shaping-calibration.json` with `decision` and per-cell `run_dir` / `log_path` / `checkpoint_path`

Grid: **reward profile × training opponent × `reseed_every_updates`**, hard cap **≤12** cells.

### Dual contract (inner trend, outer held-out)

1. **Inner filter:** After each cell train, call `evaluate_gate_records` on that cell’s `logs/*_jax.jsonl` for `beat_noop` and `beat_random` (specs from `build_gate_spec` + `docs/benchmarks/preflight-calibration.json`). **Eliminate** cells where **both** trends fail.
2. **Tier-1 rank:** Held-out **2p** `ow eval tournament` vs **noop** and **random** (seed-scheduler pattern). Rank by **`min(mean_noop, mean_random)`** so noop-only gaming does not win.
3. **Tier-2:** Top **3** survivors → unified **Stage-1** micro-bracket (`run_unified_ladder`, `stop_after_stage1=True`). Skip bracket if **&lt;2** survivors.
4. **Optional:** `--confirm-winner-tournament` runs full `ow benchmark tournament-proof` on the tier-2 winner only.

### Do not subprocess `ow benchmark gate run` per cell

`run_preflight_gate` in `src/jax/preflight.py` **retrains** in `preflight_*` campaigns. A multi-cell shaping sweep needs **one train per cell**; gate fitness must come from **`evaluate_gate_records` on existing JSONL**, not a second training pass.

### Smoke budget and thresholds

- Default **`--total-updates 50`** via planned `conf/training/shape_cal_smoke.yaml`. **`training=smoke` is only 2 updates**—too short for calibrated trend windows.
- Never invent gate or tournament floors; load from `docs/benchmarks/preflight-calibration.json` only.
- Train base: `task=shield_off`, `curriculum=off`, W&B and artifact pipeline off (mirror `SEED_SCHED_TRAIN_BASE`).

### Metric context on every cell

Document per cell in the calibration JSON:

- Opponent profile, reward profile, reseed interval
- Which metrics are **not** learning signals (self-play ~50%, `overall_win_rate` under wrong opponent mix, `episode_reward_mean` when dense shaping is on)

See `docs/solutions/logic-errors/planet-flow-sweep-gameable-objective.md` for the gameable-objective case study.

## Why This Matters

Ad-hoc W&B sweeps on raw training win rate produced policies that looked good in logs but failed held-out proof. Without a pinned `shaping-calibration.json`, agents re-invent thresholds, double-train via `gate run`, or promote on self-play noise. A bilevel operator makes shaping **searchable** (enumerable grid, campaign names) and **measurable** (calibrated gates + held-out eval + optional tournament).

## When to Apply

- Before changing default reward/opponent/reseed bundles as a **joint** MDP decision
- When adding a new `conf/reward/*` profile—run bounded calibration before merging as default
- When an agent cites training JSONL win rate alone—require gate trends **and** held-out eval
- When implementing `src/jax/shaping_calibration.py`—copy seed-scheduler; do not fork a parallel trainer

**Not for:** LLM reward codegen, shield/feature factorial grids, tournament on every cell, or automatic hybrid promotion from calibration alone (v1 non-goals).

## Examples

### Before (ad-hoc)

```bash
uv run ow train reward=ship_differential opponents=noop_only training.total_updates=300
# Promote based on W&B overall_win_rate
uv run ow benchmark gate run beat_noop   # retrains — different campaign than sweep
```

### After (target operator)

```bash
uv run ow benchmark shape-calibrate --dry-run \
  --reward-profiles terminal_only,ship_differential \
  --opponents noop_only,random_only \
  --reseed-intervals 25,50 --total-updates 50

uv run ow benchmark shape-calibrate --analyze-only
uv run ow benchmark shape-calibrate --confirm-winner-tournament
```

Decision artifact (intended shape, mirroring `docs/benchmarks/seed-scheduler-calibration.json`):

```json
{
  "gate": "shape_calibration",
  "decision": {
    "chosen_reward": "terminal_only",
    "chosen_opponents": "random_only",
    "chosen_reseed_interval": 50
  }
}
```

## Related documentation

- Benchmark CLI package (planned `shape-calibrate` subcommand): `docs/solutions/architecture-patterns/benchmark-cli-package-split-agent-native-parity.md`
- Ideation: `docs/ideation/2026-06-03-searchable-measurable-env-shaping-ideation.md`
- Requirements: `docs/brainstorms/2026-06-03-shape-calibrate-requirements.md`
- Plan: `docs/plans/2026-06-03-003-feat-shape-calibrate-plan.md`
- `docs/solutions/developer-experience/seed-scheduler-calibration-agent-native-operator-phase2.md` — same operator pattern, reseed-only axis
- `docs/solutions/developer-experience/benchmark-subprocess-training-observability.md` — `run_ow_train` during sweeps
- `docs/solutions/logic-errors/planet-flow-sweep-gameable-objective.md` — metric denominators and composite scores
- External: [ICML 2024 auto environment shaping](https://arxiv.org/html/2407.16186) — bilevel MDP design framing
