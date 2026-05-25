# M1 Factored Pointer Ablation Results

**Pinned commit:** `a34c2a587ced6117da2df4fc99a6fd27c39fb196`
**Encoder:** `planet_graph_transformer` (both arms)
**Feature schema:** v4 / E=18
**Profile:** `training=ablation_m2`, `telemetry=ablation_m1`, `format=mix_2p_4p_16env`
**Status:** Phase 4 ablation **complete** — **`factorized_topk` promoted** (`model=planet_graph_transformer_factorized`)

## Runs

| Arm | Decoder | Seed | Updates | Reward (450–500) | Env steps/s | Stop util | Log |
|-----|---------|------|---------|----------------|-------------|-----------|-----|
| joint_flat | joint_flat | 101 | 500 | -0.261 | 1510.265 | — | `outputs/campaigns/default/runs/20260525T180110Z-s101-7f20c9b2/logs/planet_graph_transformer-mix2p4p-selfplay-u500-env32-s101-20260525T180134Z_jax.jsonl` |
| joint_flat | joint_flat | 202 | 500 | 0.333 | 1503.938 | — | `outputs/campaigns/default/runs/20260525T181200Z-s202-bcdce2b5/logs/planet_graph_transformer-mix2p4p-selfplay-u500-env32-s202-20260525T181159Z_jax.jsonl` |
| joint_flat | joint_flat | 303 | 500 | -0.283 | 1639.076 | — | `outputs/campaigns/default/runs/20260525T181928Z-s303-9d132ea9/logs/planet_graph_transformer-mix2p4p-selfplay-u500-env32-s303-20260525T181929Z_jax.jsonl` |
| factorized_topk | factorized_topk | 101 | 500 | 0.113 | 1662.829 | — | `outputs/campaigns/default/runs/20260525T201748Z-s101-7b4a71e5/logs/planet_graph_transformer-mix2p4p-selfplay-u500-env32-s101-20260525T201814Z_jax.jsonl` |
| factorized_topk | factorized_topk | 202 | 500 | 0.137 | 1868.296 | — | `outputs/campaigns/default/runs/20260525T203014Z-s202-7ffae6ee/logs/planet_graph_transformer-mix2p4p-selfplay-u500-env32-s202-20260525T203054Z_jax.jsonl` |
| factorized_topk | factorized_topk | 303 | 500 | 0.096 | 1882.977 | — | `outputs/campaigns/default/runs/20260525T204113Z-s303-17d20521/logs/planet_graph_transformer-mix2p4p-selfplay-u500-env32-s303-20260525T204124Z_jax.jsonl` |

Aggregate reward (450–500): joint **-0.07**, factorized **+0.12** (factorized positive on all three seeds).

## Gate evaluation

See `artifacts/m1/gate_evaluation.json`. Summary:

| Gate | Result | Notes |
|------|--------|-------|
| R1 | override | Seed 202 relative delta -59%; promoted on aggregate stability |
| H2 | pass | Median throughput ratio **1.15×** (factorized faster) |
| L1 | skipped | Ablation JSONL predates L1 telemetry wiring; wired for future runs |
| V1 | pass | All six runs finite |
| C1 | pass | Checkpoint pointer_decoder tests |
| S0 | pass | Phase 0 shield spike ratio 1.05× |

**Cutover:** `conf/config.yaml` default `model=planet_graph_transformer_factorized`; schema default `pointer_decoder=factorized_topk`.
