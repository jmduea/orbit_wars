---
title: "feat: Preflight training profile registry (per-model PPO)"
type: feat
status: completed
date: 2026-06-02
origin: user / prior debug session (floating base.yaml PPO caused 0% win rate)
---

# feat: Preflight Training Profile Registry

## Summary

Pin **model-specific PPO hyperparameters** for preflight calibration and learn-proof gates. Gate envelopes (2p_16, rollout_steps=128, opponents, update counts) stay in code; `lr`, `clip_coef`, `ent_coef`, `epochs`, `vf_coef`, `max_grad_norm`, `update_chunk_rows`, and `reseed_every_updates` come from `docs/benchmarks/preflight-profiles.json`.

Merge `feat/planet-flow-policy` into the implementation branch so post-hygiene factorized and Planet Flow share one gate path (Planet Flow keeps `training=planet_flow` + smoke overrides via CLI).

## Problem

`ow benchmark learn-proof` and `calibrate` compose `training=2p_16` but inherit drifting PPO defaults from `conf/training/base.yaml` (`lr` 3e-4 → 6e-5). Gates read JSONL correctly; factorized runs showed real 0% `overall_win_rate` under the new defaults.

## Key decisions

**KTD1 — Registry file is source of truth for PPO pins.** `docs/benchmarks/preflight-profiles.json` maps `model` → `ppo_overrides[]` + `source` provenance string.

**KTD2 — `transformer_factorized_small` ships calibration-era PPO** (May 2026 preflight calibrate run + `conf/training/README.md` workstation knobs) until a promoted `ppo_stability_kl` W&B winner is recorded in the same file.

**KTD3 — CLI `--train-overrides` appends after registry** (Planet Flow smoke winner pattern). `--profile-path` overrides JSON path.

**KTD4 — Merge planet-flow-policy** for unified preflight/planet-flow CLI; comparison scope is post-hygiene factorized vs Planet Flow only.

## Implementation units

### U1. `src/jax/preflight_profiles.py` + JSON registry

- Load/validate profiles; `ppo_overrides_for_model(model, path?)`.
- Seed `transformer_factorized_small` with promoted PPO overrides.
- Optional stub entry for `planet_flow_target_heatmap` (PPO from smoke JSON, not registry default).

### U2. Wire `preflight.py` + `preflight_calibration.py`

- Append registry PPO overrides in `_gate_specs` (factorized path) and `calibration_train_overrides`.
- Port `extra_train_overrides` through `run_preflight_gate` / ladder from planet-flow branch.

### U3. CLI `learn-proof --train-overrides` + `--profile-path`

- Match planet-flow worktree `benchmark.py` behavior.

### U4. Tests + docs

- `tests/test_preflight_profiles.py`: registry loads; `_gate_specs` includes `training.lr=0.0003` for small model.
- Update `docs/benchmarks/preflight-calibration.md` with profile refresh steps.

### U5. Operator verification (sequential GPU)

```bash
uv run ow benchmark calibrate --model transformer_factorized_small --out docs/benchmarks/preflight-calibration.json
uv run ow benchmark learn-proof --model transformer_factorized_small --through beat_random \
  --out outputs/preflight/factorized_post_hygiene_learn_proof.json
```

## Out of scope

- Auto-import W&B sweep winners (manual JSON update when promoted).
- Gate 5 tournament re-run.

## Acceptance

- AE1: `_gate_specs("transformer_factorized_small")["beat_noop"].train_overrides` contains `training.lr=0.0003` (from registry).
- AE2: Fresh learn-proof JSONL shows nonzero `overall_win_rate` on some updates (not all-zero artifact).
- AE3: `make test-fast` passes including new profile tests.
