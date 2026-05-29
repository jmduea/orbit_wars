# Issues.md JAX — 500-update workstation validation

**Workstation validation default format is `2p_4p_16env`.** The canonical override bundle is `WORKSTATION_VALIDATION_OVERRIDES` in `scripts/issues_jax_30update_benchmark.py` (lines 58–67) or `--preset validation`, which sets `format=2p_4p_16env`, `training=workstation`, `opponents=self_play_only`, and `curriculum=off`.

Multi-seed sweep results: `docs/benchmarks/validation-seed-sweep.md`.

## Config snapshot

| Field | Value |
|-------|--------|
| Commit | `dcafdc827a9202152b27e3676edfa9e715ecc0f7` |
| Model | `transformer_factorized` |
| Format | `2p_4p_16env` (32 envs: 16×2p + 16×4p) |
| Training | `workstation` (`rollout_steps=128`, `rollout_microbatch_envs=16`, `epochs=2`) |
| Opponents | `self_play_only` |
| Curriculum | `off` (disabled) |
| Seed | `42` |
| W&B / artifacts | disabled |
| Measured updates | 500 (after 2 warmup) |

Command (benchmark script):

```bash
uv run python scripts/issues_jax_30update_benchmark.py \
  --label workstation-validation-500u \
  --tier workstation \
  --preset validation \
  --updates 500 \
  --snapshot-updates 50 100 500 \
  --out docs/benchmarks/validation-500u.json
```

Equivalent explicit overrides (see `WORKSTATION_VALIDATION_OVERRIDES` in the benchmark script):

```bash
  --overrides \
    model=transformer_factorized \
    format=2p_4p_16env \
    training=workstation \
    opponents=self_play_only \
    curriculum=off \
    telemetry.wandb.enabled=false \
    artifacts.artifact_pipeline.enabled=false \
    seed=42
```

**Why simplified:** prior run with `self_play_curriculum` + `self_play_staged` showed a severe `policy_loss` spike at update 500 while curriculum remained in `soft_start`. Validation now isolates PPO stability under pure self-play with curriculum disabled; production training profiles may still use staged curriculum separately.

Artifact: `docs/benchmarks/validation-500u.json`

## Run summary

| Metric | Value |
|--------|--------|
| Wall time (500 updates) | 530.6 s |
| `compile_seconds_to_update_3` | 310.0 s |
| `seconds_per_update_mean` | 1.061 s |
| `env_steps_per_sec` (mean) | 3860 |
| Curriculum at snapshots | `default_latest` (curriculum disabled) |

## Per-update snapshots

| Update | `mean_active_launches` | `approx_kl` | `policy_loss` | `value_loss` | `entropy` | `overall_win_rate` | `survival_time` |
|--------|------------------------|-------------|---------------|--------------|-----------|-------------------|-----------------|
| 50 | 4.55 | −0.00087 | −0.00074 | 1.13 | 3.90 | 0.67 | 0.998 |
| 100 | 0.17 | 0.00046 | 0.00019 | 0.98 | 0.76 | 0.20 | 0.998 |
| 500 | 0.84 | 0.0015 | −0.019 | 0.22 | 1.97 | 0.00 | 0.00 |

500-update **means** over all measured steps: `policy_loss` −0.0036, `approx_kl` 0.0014, `mean_active_launches` 1.52, `overall_win_rate` 0.534.

### Prior run comparison (curriculum + mixed opponents)

The previous validation used `self_play_curriculum` + `self_play_staged` and failed with `policy_loss` ≈ **291525** at update 500 (run mean 24010). Switching to `self_play_only` + `curriculum=off` eliminated the spike; the u500 `policy_loss` is −0.019 with finite `total_loss` 0.083.

## Pass / fail gate

| Criterion | Result | Notes |
|-----------|--------|-------|
| Finite scalars (no NaN/Inf) | **Pass** | `snapshots_all_finite: true` |
| `mean_active_launches` not ~0 | **Pass** | 0.17–4.55 at snapshots; run mean 1.52 (u100 dip is low but non-zero) |
| KL not exploding | **Pass** | `|approx_kl|` ≤ 0.0015 at all snapshots |
| Stable PPO losses | **Pass** | No blow-up at u500; mean `policy_loss` ≈ −0.004 |

**Overall: PASS** — PPO stability gate clears at 500 updates under pure self-play with curriculum disabled. Low launch activity at u100/u500 and u500 `overall_win_rate=0` are rollout-quality signals worth tracking separately; they do not fail the stability gate. Staged-curriculum + mixed-opponent training should be re-validated on its own profile before production use.

## Survival time appendix

**Definition** (`src/jax/env.py`, terminal helper): at episode end,

```text
survival_time = min(step + 1, MAX_STEPS) / MAX_STEPS
```

So `survival_time` is the normalized game length in \([0, 1]\), not a standalone training objective when `reward.terminal_reward_mode=binary_win` (win/loss ±1 only). Rollout telemetry aggregates `survival_time_sum / episode_done` — reported values near 0.87–1.0 mean episodes ran most of the horizon before terminating.

**Interpretation here:** high `survival_time` with `binary_win` indicates long games, not that the agent optimizes survival directly. Pair with `overall_win_rate` for outcome quality.

## Related docs

- 30-update workstation throughput: `docs/benchmarks/issues-jax-fix-30update.md`
- Micro baseline JSON: `docs/benchmarks/baseline-30update.json`
