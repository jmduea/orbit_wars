# Deep Interview Spec: M3 AlphaZero-Style MCTS Planning

**Slug:** `alphazero-mcts-planning`  
**Status:** approved (ralplan iter-2 consensus)  
**Workflow:** ralplan → omg-autopilot  
**Related:** `factored-pointer-decoder` (M1, executing), `planet-self-attention-encoder` (M2, complete)

## Goal

Add **AlphaZero-style Monte Carlo tree search** that uses the factorized top-K pointer policy as a **prior** and the existing value head for **leaf evaluation**, with **exact JAX env expansion** (no learned dynamics). Train via **policy-iteration loss** against MCTS visit counts plus value regression against search-backed returns.

## Non-Goals

- MuZero / latent dynamics model
- Dense P×P action expansion (ADR-002 violation)
- MCTS on `joint_flat` pointer (M1.1 cleanup first)
- Feature schema / `TurnBatch` changes
- Encoder architecture changes (M2)
- 4p MCTS in Phase 0–2 (deferred to Phase 3)
- Curriculum / snapshot pool / mixed-format in Phase 0–2
- Kaggle submission MCTS in v1 (train-only; inference flag optional Phase 3)
- Default cutover from PPO without Phase 4 ablation gates

## Locked Decisions (ralplan iter-2)

1. **Node granularity:** Per-launch-step nodes under a turn-boundary macro action (ADR-006).
2. **Draft state:** `TurnDraftState` — frozen turn-start `TurnBatch`, mutable `remaining_ships`, partial factorized sequences. Encoder ship features stay turn-root; legality uses draft ships (M1 parity, ADR-documented).
3. **4p backup (Phase 3+):** Learner-root search; opponents sampled at env boundary only (no opponent launch-step trees).
4. **Training integration:** `training.algorithm: ppo | alphazero` with separate collect/update modules; no MCTS branches inside `collect_rollout_jax`.
5. **Implementation:** Python MCTS tree + jitted expand/eval kernels in `src/jax/mcts/`; not monolithic full-tree JIT.
6. **Pointer gate:** `pointer_decoder=factorized_topk` only; reject at config validation.
7. **Preconditions:** M1 Phase 2 E2E + factored smoke; Phase 3 checkpoint plane; pinned `planet_graph_transformer_factorized` baseline — **not** M1 Phase 4 cutover outcome.

## Success Gates

| ID | Gate | Threshold |
|----|------|-----------|
| P0 | Spike throughput | `mcts_simulations_per_sec` ≥ **0.5×** PPO collect step (2p, fixed sims) **or** user-approved budget revision |
| P1 | Expansion parity | Deterministic MCTS root (sim=0 greedy) matches factored greedy sampler on ≥50 seeded states |
| P2 | Legality | **0** illegal commits in ≥1000 MCTS commit steps |
| P3 | Tree integrity | Unit tests: parent visits = sum children; finite Q |
| H1 | AZ collect throughput | `rollout_env_steps_per_sec` ≥ **0.70×** PPO same preset (Phase 2) |
| W1 | Learning | Episode reward ≥ PPO − **5%** at 500 updates, 3 seeds, 2p self-play, matched env-steps |
| V1 | Stability | No NaN/inf; visit entropy not collapsed at step 0 |
| C1 | Checkpoint | `training.algorithm` metadata round-trip; PPO checkpoint load into AZ config fails clearly |

## Open Questions (user lock at ralplan approval)

| # | Question | Default |
|---|----------|---------|
| Q1 | Sim budget | 800 sims/move with Phase 0 escape hatch |
| Q2 | Value target | Network leaf bootstrap + terminal outcome |
| Q3 | Start Phase 0 | After M1 Phase 2 smoke **or** parallel spike only |
| Q4 | Submission MCTS | Train-only v1 |

## References

- Plan: `.omg/plans/ralplan-alphazero-mcts-planning.md`
- M1 consumer: `.omg/plans/ralplan-factored-pointer-decoder.md`
- Architecture (to create): `docs/architecture/jax-mcts-alphazero.md`
