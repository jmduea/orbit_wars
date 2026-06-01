# Seed scheduler calibration

Source JSON: [`seed-scheduler-calibration.json`](seed-scheduler-calibration.json)

**Status:** partial sweep (9/15 training arms complete). Tournament eval not run yet. Final default stays `-1` auto-scale until held-out eval completes.

## Sweep progress

| Opponent | Reseed interval | Training | Checkpoint | Reseed events |
|----------|-----------------|----------|------------|---------------|
| `noop_only` | 0, 25, 50, 100 | complete | yes | 0 / 20 / 10 / 5 |
| `random_only` | 0, 25, 50, 100 | complete | yes | 0 / 20 / 10 / 5 |
| `self_play_only` | 0 | complete | yes | 0 |
| `self_play_only` | 25, 50, 100 | **missing** | no | interrupted at u0 compile |

Grid target: intervals `{0, 25, 50, 100}` (= `total_updates//5`) × 3 opponents = 15 runs.

## Training-side signals (in-run `overall_win_rate`, seed 42)

All completed runs pass stability gates (`|approx_kl|` run mean ≪ 0.005, finite losses).

| Opponent | Reseed | Mean WR | Last-10 WR | Reseeds |
|----------|--------|---------|------------|---------|
| noop | 0 | 0.405 | 0.446 | 0 |
| noop | 25 | 0.330 | 0.349 | 20 |
| noop | 50 | 0.368 | 0.395 | 10 |
| noop | 100 | 0.409 | 0.317 | 5 |
| random | 0 | 0.384 | 0.419 | 0 |
| random | 25 | 0.344 | 0.311 | 20 |
| random | 50 | 0.388 | 0.479 | 10 |
| random | 100 | 0.383 | 0.368 | 5 |
| self_play | 0 | 0.653 | 0.606 | 0 |

Reseed scheduling is working (event counts match interval). In-training win rate alone is not the decision metric. Frequent reseed (25) looks worse on noop last-10 than baseline, but held-out seed tournament eval is required before changing defaults.

## Decision (pending)

```json
{
  "chosen_interval": null,
  "reason": "no interval passed stability on all opponents with eval data"
}
```

## Next steps

Finish missing training arms:

```bash
uv run ow benchmark calibrate-seed-scheduler \
  --opponents self_play_only \
  --reseed-intervals 25,50,100 \
  --no-include-total-fifth \
  --total-updates 500
```

Then run held-out tournament eval on all completed checkpoints (no retrain):

```bash
uv run ow benchmark calibrate-seed-scheduler \
  --analyze-only \
  --eval-existing \
  --out docs/benchmarks/seed-scheduler-calibration.json \
  --out-md docs/benchmarks/seed-scheduler-calibration.md
```

Refresh training-only analysis anytime:

```bash
uv run ow benchmark calibrate-seed-scheduler --analyze-only
```

## Default until final decision

`training.reseed_every_updates: -1` → `max(25, total_updates // 10)` (50 on workstation 500u).
