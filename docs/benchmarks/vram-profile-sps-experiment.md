# VRAM profile — W&B + measured telemetry

Closes [#123](https://github.com/jmduea/orbit_wars/issues/123).

Primary campaign: W&B group `sps_experiment` (A100 40GB/80GB). Newer runs and
local calibration logs include `gpu_memory_peak_gb` from training telemetry.

## Summary

- Runs ingested: **44** (43 finished)
- Runs with measured VRAM: **1**

- Peak observed VRAM (measured runs): **14.64 GB**

## Source hardware

| GPU | Memory (GB) | Runs |
|-----|-------------|------|
| NVIDIA A100-SXM4-40GB | 40 | 37 |
| NVIDIA A100-SXM4-80GB | 80 | 4 |
| NVIDIA GeForce RTX 5080 | 16 | 1 |

## Measured comfort (preferred)

| GPU | Runs | Max peak (GB) | Comfort ceiling (90%) |
|-----|------|---------------|------------------------|
| NVIDIA GeForce RTX 5080 | 1 | 14.64 | 16.26 |

## Legacy proxy comfort (sps_experiment)

Used when `gpu_memory_peak_gb` is absent.

| Format | Opponent | Group envs | Max pressure | Best rs | Peak GB |
|--------|----------|------------|--------------|---------|---------|
| mix2p4p | noop | 32 | 32768 | 512 | — |
| mix2p4p | noop | 64 | 65536 | 768 | — |
| mix2p4p | selfplay | 32 | 32768 | 128 | 14.64 |
| mix2p4p | selfplay | 64 | 49152 | 768 | — |

## Workstation calibration (RTX 5080, measured)

Local run `vram_profile_calibration/rtx5080-rs128-mb16`:
`format=2p_4p_16env`, `training.rollout_steps=128`,
`training.rollout_microbatch_envs=16`, selfplay, 3 updates.

- **Peak VRAM: 14.64 GB** (~92% of 16 GB device)
- At this shape the run is near comfort limit; prefer **rs=128** (not rs=256) on 16 GB GPUs
- Full JSONL: `outputs/campaigns/vram_profile_calibration/runs/rtx5080-rs128-mb16/logs/`

## Scaled targets (16GB class GPUs)

| Target | Method | Suggested shape | Microbatch |
|--------|--------|-----------------|------------|
| NVIDIA GeForce RTX 5080 (workstation) | measured_peak_ratio | 32 env × rs=128 | 16 |
| NVIDIA Tesla P100 (Kaggle) | measured_peak_ratio | 32 env × rs=128 | 16 |

## Telemetry fields

- `gpu_memory_used_gb` — driver-reported use after each update
- `gpu_memory_total_gb` — device capacity (GiB)
- `gpu_memory_peak_gb` — running peak since run start
- `gpu_name` — logged once at run start

## Regenerate

```bash
uv run python scripts/summarize_sps_vram_profile.py --write-md \
  --calibration-jsonl path/to/run.jsonl
```
