# M2 Planet Self-Attention Ablation Results

**Pinned commit:** `a012e90b0d27d99739ada956f2b45a34e17e0f50`  
**Feature schema:** v4 / E=18  
**Profile:** `training=ablation_m2`, `format=mix_2p_4p_16env`, `curriculum=latest_only`, `telemetry=ablation`  
**Status:** Phase 3 ablation **complete** (2026-05-25)

## Runs

| Arm | Model | Seed | Updates | Avg ep reward (450–500) | Env steps/s | Win rate | Log |
|-----|-------|------|---------|-------------------------|-------------|----------|-----|
| gnn_baseline | `gnn_pointer` | 101 | 500 | -0.543 | 973 | 0.199 | `outputs/.../gnn_pointer-mix2p4p-selfplay-u500-env32-s101-20260525T155506Z_jax.jsonl` |
| gnn_baseline | `gnn_pointer` | 202 | 500 | -0.602 | 927 | 0.130 | `outputs/.../gnn_pointer-mix2p4p-selfplay-u500-env32-s202-20260525T161045Z_jax.jsonl` |
| gnn_baseline | `gnn_pointer` | 303 | 500 | -0.315 | 905 | 0.195 | `outputs/.../gnn_pointer-mix2p4p-selfplay-u500-env32-s303-20260525T162226Z_jax.jsonl` |
| transformer_m2 | `planet_graph_transformer` | 101 | 500 | -0.371 | 1585 | 0.226 | `outputs/.../planet_graph_transformer-mix2p4p-selfplay-u500-env32-s101-20260525T163153Z_jax.jsonl` |
| transformer_m2 | `planet_graph_transformer` | 202 | 500 | -0.357 | 1663 | 0.233 | `outputs/.../planet_graph_transformer-mix2p4p-selfplay-u500-env32-s202-20260525T164001Z_jax.jsonl` |
| transformer_m2 | `planet_graph_transformer` | 303 | 500 | -0.241 | 1627 | 0.222 | `outputs/.../planet_graph_transformer-mix2p4p-selfplay-u500-env32-s303-20260525T164635Z_jax.jsonl` |

Per-run JSON: `artifacts/m2/metrics_{arm}_s{seed}.json`

## Gate evaluation

| ID | Gate | Result | Evidence |
|----|------|--------|----------|
| **W1** | Reward lift vs GNN (450–500) | **PASS** | Paired lifts +31.8%, +40.7%, +23.4% on `average_episode_reward` (median +31.8%; threshold +2%). `episode_reward_mean` not emitted under lean telemetry. |
| **H2** | Throughput ≥ 0.90× GNN | **PASS** | Median `rollout_env_steps_per_sec`: GNN 927, transformer 1627 (**1.76×**). Per-format log medians: 2p **1.59×**, 4p **1.69×**. |
| **H3** | Training stability | **PASS** | All 6 runs finished 500 updates; finite entropy, no NaN/inf. |
| **H1** | Submission validator | **Deferred** | Not run in this sweep. Run before Phase 4 cutover. |
| **S1** | Shield legal rate ±5pp | **N/A** | Shield diagnostics omitted by `telemetry=ablation` / `lean_rollout_metrics`. |

Machine-readable summary: `artifacts/m2/gate_evaluation.json`

## Phase 4 cutover (2026-05-25)

**Decision:** Promote `planet_graph_transformer` as the default encoder preset (`conf/config.yaml` → `model: planet_graph_transformer`). GNN preset `gnn_pointer` remains available for one release as a fallback.

| Gate | Cutover status |
|------|----------------|
| W1 | PASS (+31.8% median lift) |
| H2 | PASS (1.76× throughput) |
| H3 | PASS |
| H1 | Deferred — run submission validator before production checkpoints |
| S1 | N/A under lean telemetry |

**Unblocks:** M1 Phase 4 factored-pointer ablation (encoder held at transformer for both arms).
