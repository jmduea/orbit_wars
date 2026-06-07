# Seed scheduler calibration

Source JSON: `docs/benchmarks/seed-scheduler-calibration.json`

## Decision

```json
{
  "chosen_interval": 50,
  "chosen_effective_interval": 50,
  "min_eval_win_rate": 0.75,
  "mean_eval_win_rate_std": 0.07856742013183861,
  "baseline_min_eval_win_rate": 0.75,
  "candidate_count": 4
}
```

## Reproduce

```bash
uv run ow benchmark calibrate-seed-scheduler \
  --opponents noop_only,random_only,self_play_only \
  --reseed-intervals 0,25,50,100 \
  --total-updates 500
```

