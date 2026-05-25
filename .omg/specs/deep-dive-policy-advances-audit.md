# Deep Dive Spec: Policy Advances Audit

## Goal

Verify whether five policy/feature advance items already exist in the codebase, and define implementation milestones for gaps.

## Scope (5 features)

| # | Feature | Audit question |
|---|---------|----------------|
| F1 | Variable-length attention via `planet_mask` | Does attention ignore inactive planets without relying on unmasked padded slots? |
| F2 | Categorical value head over returns | Is `SharedValueHead` replaced with distributional (categorical) return prediction? |
| F3 | Decoder recurrent state across turns | Does decoder GRU state carry across game turns (not just within K-step decode)? |
| F4 | Continuous ship buckets | Are ship counts predicted continuously instead of discrete bucket categoricals? |
| F5 | Lookahead / intercept plan features | Are intercept lookahead features fully implemented per the M4/M5 plan? |

## Audit Findings (2026-05-25)

### F1 — Variable-length attention (`planet_mask`)

**Status: PARTIAL — masked fixed-`MAX_PLANETS` attention is implemented; true variable-length is explicitly out of scope.**

Evidence:
- `TurnBatch.planet_mask` is populated from `planets.active` in `src/jax/features.py:81`.
- `planet_self_attention_mask` and `planet_attention_mask_with_bias` in `src/jax/encoders/planet_encoder_common.py:72-96` build additive masks from `planet_mask`.
- `PlanetGraphTransformerEncoder` (`src/jax/encoders/planet_graph_transformer.py:122-126`) and `PlanetEdgeBackboneEncoder` (`src/jax/policy.py:367-369`) apply masks in attention / adjacency.
- `masked_mean` pools only active planets (`planet_encoder_common.py:37-43`).
- Tensors remain fixed shape `(batch, MAX_PLANETS, …)` for JIT stability.
- M2 ralplan locked **fixed P=60 + boolean/additive mask** and rejected variable-length bucketing (`ralplan-planet-self-attention-encoder.md` Q3).

**Conclusion:** Feature (a) — mask-aware attention over padded planets — **exists**. Feature (b) — dynamic-length sequences without `MAX_PLANETS` padding — **not implemented and closed by M2 ADR**.

### F2 — Categorical distributions over returns

**Status: NOT IMPLEMENTED.**

Evidence:
- `SharedValueHead` emits scalar via `Dense(1)` (`src/jax/policy.py:187-202`).
- `FormatRoutedValueHead` also scalar (`policy.py:205-243`).
- `build_value_head` supports only `shared` and `format_routed` (`schema.py:47`, `policy.py:256-266`).
- PPO value loss uses scalar MSE against Monte Carlo / GAE returns (`src/jax/ppo_update.py:55-67`).
- Composable multi-value-head milestone added format routing only; no distributional critic.

**Conclusion:** Requires new milestone **M5-value** (categorical / distributional critic).

### F3 — Decoder recurrent state across turns

**Status: PARTIAL — within-turn GRU only.**

Evidence:
- `AutoregressivePointerDecoder` and `FactorizedTopKPointerDecoder` use `GRUCell` over K steps within one turn (`policy.py:121-143`, `factorized_topk_pointer.py:45-84`).
- `init_decoder_state` is derived fresh from `encoder_out.context_query` each forward pass; not stored in env or rollout carry.
- `collect_rollout_jax` scan carry is `(key, state, batch, opp_batch_cache)` — no decoder hidden state (`src/jax/rollout/collect.py:70-71`).
- `JaxTransitionBatch` / `JaxEnvState` have no decoder-state field.

**Conclusion:** Requires new milestone **M6-decoder-carry** (cross-turn decoder memory).

### F4 — Continuous ship buckets

**Status: NOT IMPLEMENTED.**

Evidence:
- `TaskConfig.ship_bucket_count: int = 8` — discrete buckets (`schema.py:21`).
- Decoders output `Dense(ship_bucket_count)` categorical logits (`policy.py:88`, `factorized_topk_pointer.py:55`).
- `ship_count_for_bucket_jax` maps bucket index → fraction of available ships (`trajectory_shield.py:102-113`).
- Trajectory shield, action builders, and PPO all assume discrete bucket indices.

**Conclusion:** Requires new milestone **M6-ships** (continuous ship fraction head + shield integration).

### F5 — Lookahead / intercept plan features

**Status: PARTIAL — M4 anchor intercept complete; M5 lookahead deferred.**

Evidence (implemented — M4 complete):
- `intercept_anchors: tuple[float, float] = (1.0, 6.0)` in `TaskConfig` and `conf/task/default.yaml`.
- `orbital_position_at_step_jax` in `src/jax/feature_primitives.py`.
- Per-anchor intercept block in `src/jax/features.py:236-299`; catalog E=18 in `src/features/catalog/edge.py`.
- Schema v4 + `intercept_anchors` metadata in `checkpoint_compat.py`.
- Tests: `tests/test_intercept_edge_features.py`, updated goldens.

Evidence (not implemented — deferred M5):
- Per-planet position lookahead at fixed τ (spec: `deep-interview-intercept-edge-features.md` non-goal).
- Intercept-sorted top-K re-rank (TODO at lexsort in `features.py`).
- Forward-projected `target_ships` per anchor (`edge.py:63` TODO(M5)).
- Phase 4 M4 ablation / reward gate deferred per manifest.

**Conclusion:** M4 **done**. Remaining intercept/lookahead work → **M5-intercept-lookahead**.

## Acceptance Criteria (audit pass)

- [x] Each of F1–F5 has evidence-backed status (exists / partial / not implemented).
- [x] Gaps mapped to named implementation milestones with dependencies.
- [x] F1 closed decision documented (masked fixed-P vs true variable-length).
- [x] F5 split into M4-complete vs M5-remaining scope.

## Execution Bridge

Proceed to `ralplan-policy-advances-audit` for milestone plans, ADR, and execution ordering.
