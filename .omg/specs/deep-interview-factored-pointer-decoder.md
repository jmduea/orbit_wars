# Deep Interview Spec: M1 Factored Pointer Decoder

**Slug:** `factored-pointer-decoder`  
**Status:** approved (ralplan iter-2 consensus)  
**Workflow:** ralplan → omg-autopilot  
**Related:** `planet-self-attention-encoder` (M2, executing — cutover blocked until M2 Phase 3 ablation)

## Goal

Replace the joint flat edge pointer (ADR-001) with a **factorized top-K pointer**: per launch step, sample **source planet → target slot (within top-K) → ship bucket**, plus a **learned stop head** that gates padding within a fixed `max_moves_k` loop. **Preserve ADR-002** top-K edge layout and feature schema.

## Non-Goals

- Dense P×P target space (full planet visibility without top-K)
- `TurnBatch` / feature schema changes (E=18 / schema v4 unchanged)
- Encoder backbone swap (M2)
- Observation normalization wiring
- MCTS / AlphaZero loop (M3 — consumes M1 output)
- Default cutover before M1 Phase 4 ablation gates pass
- Default cutover before M2 Phase 3 ablation completes
- Shield thinning refactor (deferred `thin-trajectory-shield` spec)
- Removing joint-flat decoder path (M1.1 cleanup after cutover)

## Locked Decisions (ralplan iter-2 — restored after user adjustment)

1. **Decode semantics:** **`factorized_topk`** — source ~ P (owned+ships mask) → target slot ~ K (per chosen source row) → bucket; stop head gates padding. **Not** dense P×P. **Preserve ADR-002**; amend ADR-001 only.
2. **Launch bound:** Fixed `range(max_moves_k)` JIT loop + `step_active_mask` from stop head. **No** separate `max_launches_per_turn=8` default.
3. **Features / edges:** Keep `TurnBatch` edge fields as canonical top-K target candidates.
4. **Config plane:** `model.pointer_decoder: joint_flat | factorized_topk` on existing encoder presets. Optional `conf/model/gnn_pointer_factorized.yaml`.
5. **Checkpoint plane:** `schema_version` stays **4**; add `pointer_decoder` + `action_layout_version` (`1`=joint flat, `2`=factorized top-K). Load-time rejection mirrors `encoder_backbone`.
6. **Module layout:** `src/jax/decoders/`, `src/jax/action_codec.py`. Break shield→policy import cycle in Phase 0.
7. **Shield:** Extract `evaluate_edge_pair(src_row, slot)`; precompute `(P×K×buckets)` legality — same O(P×K) cost class as today.
8. **Sequencing:** Phases 0–3 may proceed in parallel with M2 encoder work. **Phase 4 cutover blocked until M2 Phase 3 ablation completes.**

## Success Gates

| ID | Gate | Threshold |
|----|------|-----------|
| S0 | Shield spike | Factored shield+sample ≤ **1.25×** joint flat median (`mix_2p_4p_8env`) |
| S1 | Shield diagnostic | `trajectory_shield_legal_non_noop_rate` within **±5pp** of joint-flat baseline |
| H1 | Submission validity | Zero illegal actions (≥100 eps/format) |
| H2 | Throughput | `env_steps_per_sec` ≥ **0.85×** joint flat at matched env count |
| R1 | Episode reward | Mean reward ≥ joint flat − **2%** (final 50-update window, 3 seeds, 2p+4p) |
| L1 | Stop utilization | Mean active steps / `max_moves_k` > **0.5** (stop head not collapsed) |
| V1 | Training stability | No NaN/inf; entropy doesn't collapse at step 0 |
| C1 | Checkpoint rejection | Wrong `pointer_decoder` / `action_layout_version` fails with actionable error |

## Open Questions (resolved at ralplan)

- Target storage in rollout: **`source_index` + `target_slot` (0..K-1)** per step, resolved at action-build via `edge_tgt_ids` (locked)
- Escape hatch if S0 fails: keep `max_moves_k=3`, defer incremental per-step shield refresh to M1.1 (locked)
- Phase 4 baseline encoder: **GNN first**; transformer stratification is Phase 4b (optional, not blocker)
