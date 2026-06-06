# Benchmarks and calibration artifacts

Committed JSON and runbooks under this folder define verification thresholds and baselines. **Do not invent round numbers** — derive pass/fail bars from these artifacts or recalibrate with `ow benchmark` commands documented in [`AGENTS.md`](../../AGENTS.md) (preflight section).

## Primary gate sources

| Artifact | Role |
| --- | --- |
| [preflight-calibration.json](preflight-calibration.json) | Gates 2–4 learning-signal windows; Gate 5 tournament floors; unified tournament combined floors |
| [preflight-profiles.json](preflight-profiles.json) | Per-model PPO overrides for preflight gate runs |
| [seed-scheduler-calibration.json](seed-scheduler-calibration.json) | Default `training.reseed_every_updates` calibration |
| [unified-tournament-calibration.json](unified-tournament-calibration.json) | Unified ladder combined score floors |
| [qualifier-seed-calibration.json](qualifier-seed-calibration.json) | Bracket qualifier seed calibration |
| [launch-hygiene-e2e-baseline.json](launch-hygiene-e2e-baseline.json) | Tier-2 launch-hygiene throughput baseline (primary preset) |
| [launch-hygiene-e2e-4p-baseline.json](launch-hygiene-e2e-4p-baseline.json) | 4p launch-hygiene baseline companion |
| [launch-hygiene-ablation.json](launch-hygiene-ablation.json) | Launch-hygiene ablation reference |

## Env parity A/B (comet / Kaggle generators)

Isolate whether post–#188 comet stepping or Kaggle reference generators explain env-step cost:

```bash
uv run ow benchmark env-parity-ab --repeats 3 --out /tmp/env-parity-ab.json
```

Arms: `legacy` (comet-free hot path), `train` (production default), `kaggle` (reference `generate_planets` + comet spawn via `pure_callback`, diagnostic only). Compare `deltas.train_vs_legacy_pct` in the JSON; optional Hydra: `task=env_legacy` vs `task=kaggle_parity`.

## Runbooks

| Doc | Topic |
| --- | --- |
| [preflight-calibration.md](preflight-calibration.md) | Why gates split learning signal vs absolute win proof; calibration commands |
| [seed-scheduler-calibration.md](seed-scheduler-calibration.md) | Reseed interval calibration methodology |
| [validation-seed-sweep.md](validation-seed-sweep.md) | Validation seed sweep notes |

## Historical and issue-specific JSON

Additional files in this directory (validation runs, issue reproductions, terminal-reward ablations, VRAM experiments) are retained for evidence but are **not** default gate sources unless a plan or runbook explicitly cites them.

## Maintenance

After changing a committed calibration artifact, update the matching runbook and re-run the relevant `ow benchmark calibrate-*` or `make preflight-calibrate` workflow before editing thresholds in `AGENTS.md` or gate YAML.
