# Preflight gate calibration

Gates 2–3 prove **JAX learning signal** (upward trend). Gate 5 proves **absolute win rate** on `kaggle_environments` via tournament eval.

## Why split?

Committed workstation benchmarks never reached 0.85 vs `noop_only` with default PPO hyperparameters. Example 100-update run means:

| Opponent | Seed | `overall_win_rate` (run mean) |
|----------|------|--------------------------------|
| `noop_only` | 42 | 0.20 |
| `noop_only` | 43 | 0.32 |
| `random_only` | 42 | 0.10 |
| `random_only` | 43 | 0.22 |

A 200-update preflight run (seed 42, `transformer_factorized_small`) showed learning without hitting 0.85 absolute last-window win rate:

| Metric | Value |
|--------|-------|
| First-10 win rate | 0.14 |
| Last-10 win rate | 0.35 |
| Win-rate delta | 0.21 |
| Best rolling-10 | 0.65 |

Absolute JAX win rate is a poor gate with untuned self-play PPO defaults. Trend is a better fast signal. Tournament eval is the win-proof layer.

## Per-model PPO profiles (gates + calibration)

Preflight **does not** use drifting `conf/training/base.yaml` PPO defaults. Each model maps to promoted hyperparameters in `docs/benchmarks/preflight-profiles.json` (loaded by `src/jax/preflight_profiles.py`). Gate envelopes (`training=2p_16`, `rollout_steps=128`, opponents, update counts) stay in code.

When a W&B sweep winner is promoted (e.g. `ppo_stability_kl`), update `ppo_overrides` in that JSON, then re-run calibration and learn-proof for that model.

```bash
uv run ow benchmark learn-proof --model transformer_factorized_small --through beat_random \
  --profile-path docs/benchmarks/preflight-profiles.json \
  --out outputs/preflight/learn_proof_report.json
```

Optional `--train-overrides` append after the profile (Planet Flow smoke winner pattern on the worktree branch).

## Calibrated thresholds

Source of truth: `docs/benchmarks/preflight-calibration.json` (regenerate with `make preflight-calibrate`).

Current mode: **`trend_plus_tournament`**

### Gates 2–3 (learning signal, JAX JSONL)

| Check | Threshold |
|-------|-----------|
| `win_rate_delta` (last 10 vs first 10 updates) | ≥ 0.05 (12-run grid; run `make preflight-calibrate` to refresh) |
| `approx_kl` (last 10) | ≤ 0.15 |
| `entropy` (last 10) | ≥ 1e-4 |

Gate 2 still runs 200 updates vs `noop_only`. Gate 3 runs 300 vs `random_only`.

### Gate 5 (win proof, tournament)

```bash
uv run ow benchmark learn-proof \
  --eval-checkpoint outputs/.../jax_ckpt_last.pkl \
  --baselines noop \
  --out outputs/preflight/win_proof_noop.json
```

| Baseline | Min win rate |
|----------|--------------|
| `noop` | 0.70 |
| `random` | 0.58 |

Uses `kaggle_environments.make("orbit_wars")`, not Docker. Docker remains on submission packaging / hybrid promotion.

## Refresh calibration

Live sweeps subprocess `ow train` with W&B disabled; see [`benchmark-subprocess-training-observability.md`](../solutions/developer-experience/benchmark-subprocess-training-observability.md) for terminal progress expectations.

Analyze completed calibration campaigns without retraining:

```bash
uv run ow benchmark calibrate --analyze-only --analyze-campaigns
make preflight-calibrate
```

Analyze a single existing JSONL:

```bash
uv run ow benchmark calibrate --analyze-only \
  --analyze-jsonl path/to/run_jax.jsonl:noop_only:42:200
```

Short live sweep (2 opponents × 2 seeds × 200/500 updates):

```bash
uv run ow benchmark calibrate \
  --seeds 42,43 \
  --updates 200,500 \
  --opponents noop_only,random_only
```

Makefile shortcut (analyze-only on the reference noop run):

```bash
make preflight-calibrate
```

## Recommended ladder

| Gate | Command | Proves |
|------|---------|--------|
| 0 | `make test-fast` | Wiring |
| 1 | `make preflight-sanity` | Optimization reproducibility |
| 2–3 | `make preflight-learn-proof` | JAX trend vs scripted opponents |
| 4 | `ow benchmark learn-proof --through curriculum_staged` | Curriculum promotions |
| 5 | `ow benchmark learn-proof --eval-checkpoint … --baselines noop` | Tournament win proof |

## Open work

- Expand calibration with `random_only` JSONL trajectories and multi-seed sweeps before tightening tournament floors.
- Optional Docker validation remains separate from this ladder.
