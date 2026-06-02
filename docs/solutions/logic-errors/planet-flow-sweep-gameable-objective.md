---
title: Planet Flow W&B sweep optimized launch-collapse instead of learning
date: 2026-06-02
category: logic-errors
module: planet-flow-ppo-sweep
problem_type: logic_error
component: testing_framework
symptoms:
  - "Bayes sweep run 1 reached overall_win_rate=1.0 with planet_flow_emitted_launch_count=0 and entropy near zero"
  - "W&B ranked the zero-launch policy as the sweep leader when optimizing overall_win_rate on noop_only"
  - "held_demand_rate stayed at 1.0 while planet_flow_demanded_mass_sum collapsed to near zero"
root_cause: logic_error
resolution_type: code_fix
severity: high
tags:
  - planet-flow
  - wandb-sweep
  - bayes-optimization
  - preflight
  - sweep-score
  - replay
related_components:
  - src/jax/train/sweep_score.py
  - src/jax/train/loop.py
  - conf/wandb_sweep/planet_flow_ppo_signal.yaml
  - conf/artifacts/planet_flow_proof.yaml
  - src/config/runtime.py
  - src/jax/preflight.py
---

# Planet Flow W&B sweep optimized launch-collapse instead of learning

## Problem

The first Planet Flow PPO signal W&B sweep (`planet_flow_ppo_signal`, Bayes on `overall_win_rate` vs `noop_only`, 300 updates) produced a degenerate optimum: the policy stopped launching entirely, beat noop by doing nothing, and still logged `overall_win_rate=1.0`. Bayes treated that collapse as the winning direction. Replay was also blocked because Planet Flow P0 runtime guards required `artifacts=disabled`.

## Symptoms

- `overall_win_rate` rose to 1.0 while `mean_active_launches_per_turn` and `planet_flow_emitted_launch_count` went to 0.
- PPO entropy collapsed toward zero; control-path launches stayed ~8000/update (16 envs × 500 rollout steps), so `-7998` launch deltas were **correct telemetry**, not missing metrics.
- `planet_flow_held_demand_rate=1.0` was misleading when `planet_flow_demanded_mass_sum` was ~0–4 (divide-by-tiny-numerator).
- Sweep v1 ID `3zeu25xq` must **not** be resumed — its observed runs are biased toward collapse.

## What Didn't Work

- **Optimizing raw `overall_win_rate` on `noop_only`.** Noop is a degenerate opponent; not launching is a valid winning strategy and is unrelated to Planet Flow learning signal.
- **Manual guardrails only.** Comments in `conf/wandb_sweep/planet_flow_ppo_signal.yaml` listed KL/entropy/control deltas as post-hoc filters, but Bayes still explored collapse policies during the sweep.
- **`artifacts=disabled` for all Planet Flow proof paths.** Operators could not inspect HTML replays during sweep, calibration, or learn-proof runs.

## Solution

### 1. Collapse-resistant sweep objective

Log `win_rate_delta_10` (10-update first/last window delta on `overall_win_rate`, aligned with preflight gates) and composite `planet_flow_sweep_score`:

```python
# src/jax/train/sweep_score.py — floors reject launch-collapse policies
planet_flow_sweep_score = win_rate_delta_10  # when floors pass
else PLANET_FLOW_SWEEP_SCORE_INELIGIBLE  # -1.0
```

Floors include minimum mean launches, demand mass, emitted launches, entropy, and max `approx_kl`. Ineligible configs score `-1.0` so Bayes rejects them.

W&B sweep metric: `conf/wandb_sweep/metric/planet_flow_sweep_score.yaml`.

### 2. Sweep v2 fixed axes

| Axis | v1 (broken) | v2 |
|------|-------------|-----|
| Metric | `overall_win_rate` | `planet_flow_sweep_score` |
| Opponent | `noop_only` | `random_only` |
| Updates | 300 | 200 |
| Artifacts | `disabled` | `planet_flow_proof` |

Regenerate and register a **new** sweep (do not resume `3zeu25xq`):

```bash
uv run ow make wandb_sweep=planet_flow_ppo_signal
uv run wandb sweep outputs/_meta/sweeps/planet_flow_ppo_signal.yaml
uv run wandb agent <entity>/planet-flow-policy/<new_sweep_id>
```

v2 example sweep: `40il23b3`.

### 3. Async local HTML replays (`artifacts=planet_flow_proof`)

`conf/artifacts/planet_flow_proof.yaml` enables pipeline + `replay_async` + `replay_backend=local` without Docker/tournament/promotion. Runtime guards in `src/config/runtime.py` allow this replay-only path; `build_checkpoint_agent` passes `allow_planet_flow=True`.

HTML replays:

```text
outputs/campaigns/<campaign>/runs/<run_id>/evaluations/replay_u*/replay/*.html
```

Checkpoints every 50 updates (`checkpoint_every: 50`).

### 4. Rate metric floor

`src/jax/train/metrics.py` gates `held_demand_rate` / `unreachable_demand_rate` on `PLANET_FLOW_MIN_DEMAND_MASS` (100.0) so near-zero demand does not print misleading `1.0` rates.

### 5. Hydra override ordering for preflight/calibration

`PREFLIGHT_TRAIN_BASE` includes `artifacts.artifact_pipeline.enabled=false`. When adding `artifacts=planet_flow_proof`, that explicit false can survive the config-group switch and leave replay enabled with pipeline disabled. **Trail an explicit re-enable:**

```python
"artifacts=planet_flow_proof",
"artifacts.artifact_pipeline.enabled=true",
```

Used in `src/jax/preflight.py` and `src/jax/preflight_calibration.py`.

## Why This Works

The failure was objective misalignment, not broken Planet Flow telemetry. `overall_win_rate` vs noop rewards inactivity; preflight learn-proof gates instead require **positive win-rate trend** plus stable PPO diagnostics and non-trivial Planet Flow activity. Encoding those floors into the W&B scalar (`planet_flow_sweep_score`) makes Bayes explore policies that actually launch and improve. Tuning on `random_only` avoids the noop degeneracy while learn-proof still validates `beat_noop` → `beat_random`. Async local replays make collapse vs real learning visually obvious without enabling unsupported Docker/tournament paths.

## Prevention

- **Before gating on a training metric**, confirm its denominator and what “chance” means for that opponent (see AGENTS.md metric gates).
- **Align W&B sweep objectives with preflight gates** — use trend metrics (`win_rate_delta`), not raw win rate on trivial opponents.
- **Add activity floors to composite sweep scores** when the action space allows “do nothing” wins.
- **Never resume a sweep known to have gameable observations** — register a new sweep ID after fixing the objective.
- **Use `artifacts=planet_flow_proof`** for Planet Flow sweep/calibration/learn-proof when visual replay inspection is needed.
- **Test Hydra compose** for proof paths: `compose_hydra_train_config([..., *PREFLIGHT_TRAIN_BASE, "artifacts=planet_flow_proof", "artifacts.artifact_pipeline.enabled=true"])`.

## Related Issues

- Plan: `docs/plans/2026-06-02-001-feat-planet-flow-sweep-v2-replay-plan.md`
- Prior sweep plan (v1 rationale, superseded objective): `docs/plans/2026-06-01-004-feat-planet-flow-ppo-sweep-plan.md`
- Benchmark subprocess observability (separate concern): `docs/solutions/developer-experience/benchmark-subprocess-training-observability.md`
- Sweep shortlist CLI deferred: GitHub issue #166
