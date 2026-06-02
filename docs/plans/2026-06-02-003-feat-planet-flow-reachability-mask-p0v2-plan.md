---
title: "feat: Planet Flow reachability-masked PPO contract (P0 v2)"
type: feat
status: active
date: 2026-06-02
origin: docs/brainstorms/2026-06-02-planet-flow-reachability-contract-requirements.md
---

# feat: Planet Flow Reachability-Masked PPO Contract (P0 v2)

## Summary

Implement P0 v2 from the reachability contract: build a shared **catalog-reachability mask**, apply it at **logit level** for Planet Flow sampling and PPO replay, wire inference parity, and add **calibrated post-mask unreachable gates** to preflight/sweep. Keep the existing per-source argmax compiler unchanged.

## Problem Frame

Planet Flow P0 trains on demand over all active planets while the compiler only executes top-K catalog edges. u150 replay (~65% unreachable demand) showed PPO credit on impossible targets while the compiler fired at legal neutrals. `planet_flow_target_mask` today mirrors `planet_mask` (activity-only), not catalog reachability.

## Requirements Traceability

| Req | Plan unit |
|-----|-----------|
| R0–R3 | U1 |
| R4–R6 | U2, U3 |
| R7–R9 | U4 (compiler tests only; no compiler rewrite) |
| R10, R12 | U5 |
| R13–R14 | U6 |
| R15 | U6 (extend existing coverage audit hook) |
| R16 | U3 (inference path) |

## Key Technical Decisions

**KTD1 — Shared reachability builder.** New `catalog_target_reachability(batch, game)` returns `(num_planets,)` bool: for each target row, ∃ owned source with `edge_mask[src, slot]` and `edge_tgt_ids[src, slot] == planets.id[tgt_row]`. Uses `edge_mask` only (no demand). Lives in `src/jax/planet_flow.py` next to compiler.

**KTD2 — Sampling mask = active ∧ reachable.** `target_mask = batch.planet_mask & catalog_reachability_by_row`. Never sample on inactive planets; never sample on catalog-unreachable planets.

**KTD3 — Logit-level masking.** Refactor `planet_flow_action_log_prob_entropy`, `planet_flow_categorical_kl`, and `sample_planet_flow_pressure_action` to apply `_safe_masked_logits` per planet row (mask shape matches bucket dim broadcast from planet reachability). Do not rely on post-sample zeroing alone.

**KTD4 — No demand renormalization.** Masking excludes unreachable logits; do not redistribute mass onto reachable planets in the sampler (R9). Compiler continues to consume sampled `target_pressure` as today.

**KTD5 — Post-mask unreachable gate.** After masking ships, compiler `unreachable_demand_rate` should be ~0. Add calibrated ceiling to `preflight-calibration.json` via measurement campaign; wire `preflight.py` + `sweep_score.py` eligibility. June 1 learn-proof / entropy / held-demand gates remain unchanged.

**KTD6 — Checkpoint metadata deferred.** R16 (June 1 R12 schema) stays P1; P0 v2 uses config profile `model=planet_flow_target_heatmap` + reachability code path uniformly.

---

## Implementation Units

### U1. Catalog reachability mask builder

**Goal:** Single source of truth for R1–R3 used by rollout, PPO, eval, compiler diagnostics split.

**Requirements:** R0–R3

**Files:**
- `src/jax/planet_flow.py` — add `catalog_target_reachability(game_row, batch_row) -> bool[num_planets]`
- `tests/test_planet_flow_reachability.py` (new)

**Approach:**
- For each target planet id, OR over owned source rows where any `edge_mask` slot targets that id.
- Match target id via `game_row.planets.id` (same as `_edge_target_pressure` fix).
- Export batched vmap wrapper for rollout.

**Test scenarios:**
- Single owned source, enemy off-catalog → enemy row False, neutral in catalog True.
- Capture adds edge → enemy becomes True next turn (F3).
- No owned ships → all False (or only rows with edges from empty ownership — document expected hold behavior per Outstanding Question).

**Verification:** `uv run pytest tests/test_planet_flow_reachability.py -q`

---

### U2. Logit-level Planet Flow action codec

**Goal:** R4–R5 — sample/logprob/entropy/KL respect reachability at softmax input.

**Requirements:** R4–R6

**Files:**
- `src/jax/action_codec.py`
- `tests/test_planet_flow_action_codec.py` (new or extend if exists)

**Approach:**
- In `sample_planet_flow_pressure_action`: build `bucket_mask = target_mask[..., None]` (broadcast over bucket axis); run categorical/argmax on `_safe_masked_logits(logits, bucket_mask)`.
- In `planet_flow_action_log_prob_entropy`: pass reachability as `bucket_mask` into `_safe_categorical_*` with `active=target_mask`.
- In `planet_flow_categorical_kl`: mask both old/new logits with reachability before softmax (mirror factorized pattern).

**Test scenarios:**
- Raw logits favor unreachable planet; mask excludes it → sample never picks unreachable; logprob/entropy zero contribution from that row.
- KL between old/new policies ignores unreachable rows.
- `reachable_count=0` → zero logprob/entropy for step (document in test; align with Outstanding Question).

**Verification:** `uv run pytest tests/test_planet_flow_action_codec.py -q`

---

### U3. Wire mask through rollout, PPO replay, inference

**Goal:** Replace activity-only mask with catalog reachability everywhere learner acts.

**Requirements:** R2, R4, R16

**Files:**
- `src/jax/rollout/collect.py` — compute reachability mask; pass to `sample_planet_flow_pressure_action`
- `src/jax/ppo_update.py` — replay uses stored mask (must be reachability mask from rollout)
- `src/jax/action_sampling.py` — inference/submission path
- `tests/test_jax_ppo.py` or targeted planet-flow replay test if marker exists

**Approach:**
- After `_policy_turn_batch`, compute `reachability = catalog_target_reachability(state.game, batch)` batched.
- `target_mask = policy_batch.planet_mask & reachability`.
- Store `planet_flow_target_mask` as this combined mask in rollout tensors (no schema change).

**Test scenarios:**
- Rollout stored mask excludes unreachable enemy on synthetic board.
- PPO replay logprob matches resampled mask from trajectory.
- `build_checkpoint_agent` inference uses same mask builder (grep submission path).

**Verification:** targeted pytest on planet flow PPO replay tests (jax tier if applicable)

---

### U4. Compiler regression alignment (R7–R9)

**Goal:** Lock hold semantics; no compiler rewrite.

**Requirements:** R7–R9, AE3

**Files:**
- `tests/test_planet_flow_compiler.py` — extend AE for all-unreachable masked demand → no launch

**Approach:**
- Add test: sampled pressure only on unreachable planets (simulated pre-mask) vs masked pressure all zero → no valid launches.
- Existing neutral-spill test remains valid P0 behavior.

**Verification:** `uv run pytest tests/test_planet_flow_compiler.py -q`

---

### U5. Diagnostics and post-mask metrics

**Goal:** R10, R12 — masked entropy reporting; unreachable vs held distinction preserved.

**Requirements:** R10, R12

**Files:**
- `src/jax/train/metrics.py` (if masked entropy needs explicit naming)
- `src/jax/rollout/collect.py` (optional debug: pre-mask unreachable from raw logits when `metric_groups.debug`)

**Approach:**
- Confirm `planet_flow_unreachable_demand_rate` from compiler uses masked `target_pressure` post-U3 → expect ~0 in integration smokes.
- Optional debug metric: mass on unreachable from raw argmax logits before mask (Outstanding Question — implement only if cheap).

**Verification:** short train smoke + inspect `logs/*_jax.jsonl` for unreachable rate

---

### U6. Preflight, sweep gates, calibration

**Goal:** R13–R15 — fail runs with post-mask unreachable above calibrated ceiling; inherit June 1 gates.

**Requirements:** R13–R15

**Files:**
- `src/jax/preflight.py`
- `src/jax/preflight_calibration.py`
- `src/jax/train/sweep_score.py`
- `docs/benchmarks/preflight-calibration.json` (after `make preflight-calibrate` campaign)
- `tests/test_preflight_calibration.py`, `tests/test_planet_flow_sweep_score.py`

**Approach:**
- Add `planet_flow_post_mask_unreachable_demand_rate_max` (or reuse existing rate with documented post-mask semantics) to calibration schema.
- Planet Flow preflight gate: fail if rate > calibrated ceiling after sufficient demanded mass.
- Sweep eligibility: reject when post-mask unreachable rate exceeds ceiling (alongside existing entropy/launch floors).
- Document R15 as extending existing coverage audit command/hook — no new audit product.

**Test scenarios:**
- Preflight config test asserts new threshold key present for planet flow profile.
- Sweep score test: high post-mask unreachable → ineligible.

**Verification:** `make test-domain-config` + targeted preflight/sweep tests

---

### U7. Pipeline relaunch (follow-on, after U1–U6)

**Goal:** Re-run sweep/calibration/learn-proof on `training=2p4p_16_split` with P0 v2 contract.

**Requirements:** Success Criteria (P0 v2 proof)

**Depends on:** U1–U6 merged

**Commands:**
```bash
uv run ow make wandb_sweep=planet_flow_ppo_signal
uv run wandb sweep outputs/_meta/sweeps/planet_flow_ppo_signal.yaml
# agent on new sweep id
make preflight-calibrate  # after measurement runs
uv run ow benchmark learn-proof --through beat_random ...
```

**Verification:** sweep YAML has `2p4p_16_split`; agent log shows masked training; post-mask unreachable ≈ 0

---

## Sequencing

```
U1 → U2 → U3 → U4
         ↘ U5 (parallel after U3)
U1–U5 → U6 → U7
```

U2 blocks U3. U6 needs U3 metrics path. U7 is operator/GPU work after code lands.

## Scope Boundaries

**In scope:** Reachability mask, logit masking, rollout/PPO/inference wiring, gates, tests.

**Out of scope:** Compiler rewrite (Option B), catalog threat slot (C-lite), checkpoint schema migration (June 1 R12), reachable-attack-pressure new metric (deferred).

## Risks & Dependencies

- **Calibration:** R13 thresholds must come from measurement — run short Planet Flow smokes before tightening gates.
- **GPU contention:** U7 sweep/calibration one job at a time.
- **Legacy checkpoints:** Global-heatmap runs remain valid under old code path until retrained; eval must use matching code version.
- **Zero reachable set:** Define hold behavior before U6 gates to avoid false preflight failures.

## Test Plan (default loop)

```bash
make test-fast  # after each unit
uv run pytest tests/test_planet_flow_reachability.py tests/test_planet_flow_action_codec.py tests/test_planet_flow_compiler.py -q
make test-domain-config  # after U6
```

JAX-compile smokes only when user requests or pre-merge.
