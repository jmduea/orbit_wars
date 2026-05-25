# Feature Encoding v2 — Phase 0 Results

**Date:** 2026-05-25  
**Plan:** `.omg/plans/ralplan-feature-encoding-v2.md`  
**Spike script:** `scripts/spike_feature_encoding_v2_phase0.py`

Phase 0 is contract-only (no v2 encoder code). This document records locked schema dims, spike outcomes, v1 baseline capture, and dependency gates.

---

## Schema lock (final)

| Tensor | Dim | Notes |
|--------|-----|-------|
| **P** (planet) | **13** | Layer B sun-polar + Layer C local state |
| **E** (edge) | **12** | Layer C geometry + target-at-edge snapshot |
| **G** (global) | **46** | v1 global (45) + `angular_velocity` |
| **K** | `max(0, candidate_count - 1)` | default **3** when `candidate_count=4` |
| **Pointer logits** | `MAX_PLANETS * K + 1` | default **181** (+ NO_OP) |

### Float budget (default H=1, K=3)

| Component | Floats |
|-----------|-------:|
| `planet_features` (60×13) | 780 |
| `edge_features` (60×3×12) | 2160 |
| `global_features` | 46 |
| **Total encoder payload / env-step** | **2986** |

Encoder-only rollout memory (float32, 16 envs × 64 steps): **~11.7 MB** (policy/transition overhead excluded).

**Verdict:** ACCEPT for Phase 1. Static payload exceeds a single v1 owned-source row (171 floats) but is comparable to ~17 owned sources; joint pointer softmax (181) replaces v1's per-source 4-way slot decoder. The ralplan “≤200 floats/decision” target applies to v1 row semantics, not the full board tensor.

Run: `uv run python scripts/spike_feature_encoding_v2_phase0.py budget`

---

## Symmetry / equivariance spike

| Check | Result |
|-------|--------|
| Canonical angle roundtrip (decode contract) | PASS |
| Synthetic 4-planet 90° CCW + owner relabel | PASS (sorted pair delta &lt; 1e-15) |
| Synthetic 4-planet 180° + owner relabel | PASS |
| JAX 2p reset seed 11 + 90° transform | Loose check only (ownership not fully symmetric mid-label); max pair delta ≈ 0.29 |

**Verdict:** PASS for ADR-004 frame math on known symmetric transforms. Real early-game boards are not fully ownership-invariant under 90° without Layer D sort (deferred).

Run: `uv run python scripts/spike_feature_encoding_v2_phase0.py symmetry`

---

## jax-ppo-split dependency gate

| Item | Status |
|------|--------|
| `jax-ppo-split` manifest | **complete** (2026-05-25) |
| Evidence | `src/jax/ppo.py` deleted; modules at `rollout/`, `ppo_update.py`, `train_state.py`, `opponents/jax_actions/` |
| Gate for Phase 3+ | **CLEARED** |

---

## v1 baseline (encoding v1, `model=gnn_pointer`)

Ablation anchor config (matches `conf/sweeps/wandb/gnn_pointer_reward_validate.yaml` throughput knobs):

```
model=gnn_pointer
format=mix_2p_4p_8env
training.rollout_steps=64
training.minibatch_size=256
training.rollout_microbatch_envs=8
task.candidate_count=4
task.feature_history_steps=1
task.trajectory_shield_enabled=false
```

### Throughput (benchmark, steady-state after warmup)

Command:

```bash
uv run python scripts/benchmark_jax_rl.py \
  --overrides model=gnn_pointer format=mix_2p_4p_8env \
  training.rollout_steps=64 training.minibatch_size=256 \
  training.rollout_microbatch_envs=8 \
  --updates 5 --warmup 1
```

| Format override | env_steps_per_sec | notes |
|-----------------|------------------:|-------|
| `mix_2p_4p_8env` | **1298** | 32 envs (2p+4p groups), 2026-05-25 |
| `2p_16env` | **1283** | isolated 2p smoke |
| `4p_16env` | **1298** | isolated 4p smoke |

### Short training smoke (seed 101, 25 updates)

Command:

```bash
uv run python -m src.train model=gnn_pointer format=mix_2p_4p_8env \
  training.total_updates=25 training.rollout_steps=64 \
  training.minibatch_size=256 training.rollout_microbatch_envs=8 \
  seed=101 telemetry.wandb.enabled=false \
  artifacts.artifact_pipeline.enabled=false artifacts.replay.enabled=false
```

Log: `outputs/campaigns/default/runs/20260525T065628Z-s101-b7c145c5/logs/…_jax.jsonl`

| Metric (update 25, steady-state) | Value |
|----------------------------------|------:|
| `overall_win_rate` | 1.00 |
| `win_rate_2p` | 1.00 |
| `first_place_rate_4p` | 0.00 (no 4p episodes in this short smoke) |
| `rollout_env_steps_per_sec` | 3627 |
| `rollout_env_steps_per_sec_2p` | 3854 |
| `rollout_env_steps_per_sec_4p` | 0 (no 4p group in logged updates) |
| `trajectory_shield_legal_non_noop_rate` | 0.0 (shield disabled) |
| `completed_episodes` | 33 |

**Caveats:** 25 updates is a Phase 0 smoke only — not a policy-quality claim. Full ablation baseline should use ≥3 seeds and 500+ updates per ralplan Phase 4. Re-baseline with `trajectory_shield_enabled=true` before Phase 3 shield parity gates.

### Reference (promoted attention baseline, v1 encoding)

From `docs/baseline_sweep_results.md` (different model; listed for throughput context):

| Metric | Median (3 seeds, 25 updates) |
|--------|-----------------------------:|
| `env_steps_per_sec` | 489 |
| `overall_win_rate` | 0.00 |

---

## Phase 0 exit checklist

- [x] ADR-001 action space finalized
- [x] ADR-002 edge layout (top-K) finalized
- [x] ADR-003 ship feature scale
- [x] ADR-004 symmetry frame
- [x] P/E/G dims locked
- [x] Float budget spike
- [x] Equivariance spike (known transforms)
- [x] v1 baseline captured (commands + smoke metrics)
- [x] jax-ppo-split gate cleared
- [x] Config sketch (`conf/model/gnn_pointer_v2.yaml`)
- [x] Submission audit (see `docs/feature-encoding-v2.md`)
- [ ] Layer D planet sort — **deferred** (user lock)

**Phase 0 status: COMPLETE** — ready for Phase 1 encoder implementation.
