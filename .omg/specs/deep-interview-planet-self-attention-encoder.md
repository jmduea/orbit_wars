# Deep Interview Spec: M2 Planet Self-Attention Encoder

**Slug:** `planet-self-attention-encoder`  
**Status:** approved (ralplan iter-2 consensus)  
**Workflow:** ralplan → omg-autopilot  
**Related:** `intercept-edge-features` (M4, executing — must complete before M2 ablation)

## Goal

Replace the k-NN `PlanetEdgeBackboneEncoder` message-passing stack with a masked planet self-attention encoder (Graph Transformer) while preserving `PlanetEdgeEncoderOutput`, ADR-001/002 action space, and feature schema.

## Non-Goals

- Feature schema / `TurnBatch` changes (owned by M4 intercept edges)
- Action-space / decoder / shield rewrites (M1 follow-up)
- Observation normalization wiring
- Multi-relation attention bias (M2.1 follow-up)
- MCTS / planning (M3)
- Default architecture cutover before ablation gates pass

## Locked Decisions (ralplan iter-2)

1. **Sequencing:** M2 implementation may land on `main` pre-M4; **ablation (Phase 3) starts only after M4 merge** with E=18 / schema v4 pinned on both arms.
2. **Architecture slug:** `planet_graph_transformer` (new Hydra preset); `gnn_pointer` remains legacy GNN encoder until cutover gate.
3. **Encoder depth:** 2 transformer layers default; Phase 0 JIT spike may downgrade to 1 layer before lock.
4. **Attention bias:** spatial coordinate bias only (orbit-derived pairwise distance / delta).
5. **Edge fusion:** tgt-aware fusion (`src + tgt + edge`) applied **symmetrically to both** GNN and transformer encoders before ablation.
6. **Checkpoint plane:** `encoder_backbone: planet_gnn | planet_self_attention` metadata; load-time rejection on mismatch (separate from feature `schema_version`).
7. **Padding:** fixed `MAX_PLANETS=60` masked attention; no variable-length bucketing.

## Success Gates

| ID | Gate | Threshold |
|----|------|-----------|
| W1 | `episode_reward_mean` lift vs paired GNN | ≥2% final window (updates 450–500), 3 paired seeds, 2p + 4p |
| H1 | Submission validity | Zero illegal actions (≥100 eps/format) |
| H2 | Throughput | `median(env_steps_per_sec)_transformer ≥ 0.90 × GNN` per format |
| H3 | Training stability | No NaN/inf, no collapse |
| S1 | Shield diagnostic | `trajectory_shield_legal_non_noop_rate` within ±5pp of GNN |

## Open Questions (resolved at ralplan)

- Architecture naming: **`planet_graph_transformer`** (locked)
- `hidden_size=224`, `attention_heads=7` → head_dim=32 (locked; validate divisibility at build)
- M2 before M4 merge: **implementation OK; ablation blocked until M4 lands**
