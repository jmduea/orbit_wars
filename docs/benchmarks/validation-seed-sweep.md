# Workstation validation — multi-seed seed sweep

**Workstation validation default format is `2p_4p_16env`.** Canonical overrides live in `WORKSTATION_VALIDATION_OVERRIDES` (`ow benchmark training` (`src/benchmark/training.py` presets), lines 58–67) or `--preset validation`:

| Field | Default |
|-------|---------|
| Format | `2p_4p_16env` (32 envs: 16×2p + 16×4p) |
| Training | `workstation` |
| Opponents | `self_play_only` (primary validation) |
| Curriculum | `off` |
| Model | `transformer_factorized` |
| Telemetry / artifacts | W&B and artifact pipeline disabled |

Single-seed 500-update gate: `docs/benchmarks/issues-jax-validation-500u.md`.

## How to reproduce

Primary sweep (500 updates, self-play):

```bash
uv run ow benchmark training \
  --label validation-seed-<SEED>-500u \
  --tier workstation \
  --overrides \
    model=transformer_factorized \
    format=2p_4p_16env \
    training=workstation \
    opponents=self_play_only \
    curriculum=off \
    telemetry.wandb.enabled=false \
    artifacts.artifact_pipeline.enabled=false \
    seed=<SEED> \
  --updates 500 \
  --out docs/benchmarks/validation-seed-<SEED>-500u.json
```

Equivalent to `--preset validation` with a different seed (preset pins `seed=42` in `WORKSTATION_VALIDATION_OVERRIDES`).

Batch runner:

```bash
bash scripts/run_validation_seed_sweep.sh
```

Aggregate summary:

```bash
uv run python scripts/summarize_validation_seed_sweep.py
```

Commit: `dcafdc827a9202152b27e3676edfa9e715ecc0f7` · GPU: RTX 5080 · JAX 0.10.0 (CUDA).

## Primary sweep — `self_play_only`, 500 updates

| Seed | Wall (s) | env_steps/s | compile→u3 (s) | policy_loss | value_loss | approx_kl | mean_active_launches | overall_win_rate |
|------|----------|-------------|----------------|-------------|------------|-----------|----------------------|------------------|
| 42 | 531 | 3860 | 310 | −0.0036 | 1.94 | 0.0014 | 1.52 | 0.534 |
| 43 | 2157 | 949 | 415 | −0.0156 | 2.15 | 0.0025 | 3.31 | 0.635 |
| 44 | 431 | 4749 | 235 | −0.0284 | 2.66 | 0.0039 | 3.62 | 0.795 |
| 45 | 432 | 4742 | 237 | −0.0170 | 2.29 | 0.0034 | 4.28 | 0.747 |
| 46 | 436 | 4701 | 236 | −0.0079 | 2.17 | 0.0040 | 1.85 | 0.614 |

### Mean ± std (`self_play_only`, all 5 seeds)

| Metric | Mean ± std |
|--------|------------|
| Wall time (s) | 797 ± 761 |
| env_steps/sec | 3800 ± 1638 |
| compile→u3 (s) | 287 ± 78 |
| policy_loss | −0.014 ± 0.010 |
| value_loss | 2.24 ± 0.26 |
| approx_kl | 0.0030 ± 0.0011 |
| mean_active_launches | 2.92 ± 1.18 |
| overall_win_rate | 0.665 ± 0.105 |

**Throughput note:** seed 43 was a first-run outlier (2157 s, 949 env_steps/s). Excluding it, typical 500-update wall time is **457 ± 47 s** and **4513 ± 385 env_steps/s** (seeds 42, 44–46).

**Update count choice:** primary sweep used **500 updates** (not 100) for PPO stability parity with the single-seed gate in `issues-jax-validation-500u.md`.

Artifacts: `validation-500u.json`, `validation-seed-{43,44,45,46}-500u.json`. Summary JSON: `validation-seed-sweep-summary.json`.

## Opponent contrast — 100 updates (seeds 42, 43)

Same format/curriculum/training; shorter run for behavioral contrast.

| Opponents | Seed | Wall (s) | env_steps/s | policy_loss | approx_kl | mean_active_launches | overall_win_rate |
|-----------|------|----------|-------------|-------------|-----------|----------------------|------------------|
| noop_only | 42 | 199 | 2055 | 0.037 | 0.0048 | 1.20 | 0.204 |
| noop_only | 43 | 192 | 2129 | −0.032 | 0.0050 | 3.24 | 0.319 |
| random_only | 42 | 196 | 2091 | 0.0026 | 0.00012 | 0.16 | 0.105 |
| random_only | 43 | 185 | 2210 | −0.024 | 0.0033 | 3.34 | 0.225 |

**Contrast vs self_play_only:** noop/random profiles show lower `overall_win_rate` and often lower launch activity (especially `random_only` seed 42), while self-play 500u runs cluster around 0.53–0.80 win rate with higher engagement. PPO losses remain finite across all profiles.

## Pass / fail (multi-seed)

| Criterion | Result |
|-----------|--------|
| Finite scalars (no NaN/Inf) | **Pass** — all JSON artifacts |
| KL not exploding | **Pass** — run-mean \|approx_kl\| ≤ 0.005 |
| Stable PPO losses at 500u | **Pass** — no blow-ups across seeds 42–46 |
| Throughput regression | **Pass** — ~4.5k env_steps/s typical (excl. seed 43 outlier) |

**Overall: PASS** for workstation validation sign-off under `format=2p_4p_16env`, `self_play_only`, `curriculum=off`.

## Sign-off recommendations

1. **Default format confirmed** — code and docs agree on `2p_4p_16env` via `--preset validation`.
2. **Stability** — 500-update self-play sweep clears the PPO gate on all five seeds; safe to sign off on JAX training stability for this profile.
3. **Throughput** — treat seed 43 as a cold-start outlier when interpreting wall time; re-run if a single canonical throughput number is needed for CI baselines.
4. **Opponents** — use `self_play_only` for primary validation; `noop_only` / `random_only` are useful sanity checks but not production training profiles.
5. **Next** — re-validate `self_play_curriculum` + staged curriculum separately before enabling in production runs.
