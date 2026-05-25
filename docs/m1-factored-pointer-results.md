# M1 Factored Pointer Ablation Results

**Pinned commit:** `a34c2a587ced6117da2df4fc99a6fd27c39fb196`  
**Encoder:** `planet_graph_transformer` (both arms — M2 Phase 4 cutover)  
**Feature schema:** v4 / E=18  
**Profile:** `training=ablation_m2`, `telemetry=ablation_m1`, `format=mix_2p_4p_16env`, `curriculum=latest_only`  
**Status:** Phase 4 ablation **pending**

## Arms

| Arm | Hydra preset | `pointer_decoder` |
|-----|--------------|-------------------|
| A (baseline) | `planet_graph_transformer` | `joint_flat` |
| B (treatment) | `planet_graph_transformer_factorized` | `factorized_topk` |

## Run

```bash
# In-process (recommended — one JAX compile per decoder path)
uv run python scripts/run_m1_ablation.py --skip-existing

# Evaluate gates after all 6 runs complete
uv run python scripts/evaluate_m1_gates.py
```

Per-run JSON: `artifacts/m1/metrics_{arm}_s{seed}.json`  
Pin: `artifacts/m1/baseline_pin.json`

## Runs

| Arm | Decoder | Seed | Updates | Reward (450–500) | Env steps/s | Stop util | Log |
|-----|---------|------|---------|------------------|-------------|-----------|-----|
| _pending_ | | | | | | | |

## Gate evaluation

| ID | Gate | Threshold | Result |
|----|------|-----------|--------|
| **R1** | Episode reward | Factorized ≥ joint −2% per seed | _pending_ |
| **H2** | Throughput | ≥ 0.85× joint flat per seed | _pending_ |
| **L1** | Stop utilization | > 0.5 (`mean_active / max_moves_k`) | _pending_ |
| **S0** | Shield spike | ≤ 1.25× (Phase 0) | _see Phase 0_ |
| **S1** | Shield diagnostic | ±5pp | _optional_ |
| **H1** | Submission validity | Zero illegal actions | _not run_ |
| **V1** | Stability | No NaN/inf | _pending_ |
| **C1** | Checkpoint rejection | Wrong decoder fails load | _tests_ |

Machine-readable summary: `artifacts/m1/gate_evaluation.json` (after `evaluate_m1_gates.py`)

## Cutover recommendation

_Pending ablation completion. If R1/H2/L1 pass, promote `pointer_decoder=factorized_topk` on the transformer preset._
