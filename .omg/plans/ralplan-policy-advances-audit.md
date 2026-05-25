# Ralplan iter-1 (consensus): Policy Advances Audit & Milestone Plan

**Source spec:** `.omg/specs/deep-dive-policy-advances-audit.md`  
**Slug:** `policy-advances-audit`  
**Workflow:** ralplan  
**Status:** planned (consensus iter-1)  
**Date:** 2026-05-25

---

## Executive Summary

| # | Feature | Status | Action |
|---|---------|--------|--------|
| F1 | Variable-length attention via `planet_mask` | **PARTIAL (closed)** | Masked fixed-`MAX_PLANETS` **done**; true variable-length **rejected** (M2 ADR). No new work unless ADR reopened. |
| F2 | Categorical value head over returns | **NOT IMPLEMENTED** | Milestone **M5-value** |
| F3 | Decoder state across turns | **PARTIAL** | Milestone **M6-decoder-carry** |
| F4 | Continuous ship buckets | **NOT IMPLEMENTED** | Milestone **M6-ships** |
| F5 | Intercept / lookahead features | **PARTIAL** | M4 **complete**; remainder → **M5-intercept-lookahead** |

**Recommended execution order:** `M5-intercept-lookahead` → `M5-value` → (`M6-decoder-carry` ∥ `M6-ships` after M1 factorized decoder stabilizes)

---

## RALPLAN-DR Summary

### Principles

1. **Evidence-first audit** — distinguish masked fixed-P (implemented) from true variable-length (closed).
2. **One milestone per gap** — no bundling unrelated critic/decoder/ship changes.
3. **Preserve JIT contract** — `TurnBatch` stays fixed-shape; cross-turn state lives in rollout carry, not dynamic tensors.
4. **Shield remains legality oracle** — continuous ships and intercept features inform the policy; shield validates launches.
5. **Checkpoint schema bumps are explicit** — distributional critic and continuous ships invalidate old value/action heads.

### Decision Drivers

1. **F1 is already satisfied for training** — both GNN and transformer encoders mask inactive planets; reopening variable-length adds JIT complexity with no current ablation signal.
2. **Scalar critic limits tail-risk modeling** — categorical returns (C51-style) may help long-horizon Orbit Wars credit assignment.
3. **Within-turn GRU resets each turn** — cross-turn memory could capture multi-turn fleet plans but requires env/rollout schema changes.
4. **8 discrete buckets quantize ship fractions** — continuous head could reduce bucket quantization error but complicates shield and log-prob math.
5. **M4 intercept anchors landed** — remaining lookahead (top-K re-rank, per-planet τ features, projected target ships) is the highest-value feature follow-up.

---

## Feature Audit Detail

### F1 — Variable-length attention

**What exists today**

```72:77:src/jax/encoders/planet_encoder_common.py
def planet_self_attention_mask(planet_mask: jax.Array) -> jax.Array:
    """Return a boolean mask for planet self-attention over padded rows."""

    mask = planet_mask[:, :, None] & planet_mask[:, None, :]
    has_valid_key = mask.any(axis=-1, keepdims=True)
    return jnp.where(has_valid_key, mask, jnp.ones_like(mask))
```

- `PlanetGraphTransformerEncoder` applies this mask in every transformer block.
- `PlanetEdgeBackboneEncoder` masks k-NN adjacency with `planet_mask`.
- Pooling uses `masked_mean(..., planet_mask)`.

**What does not exist**

- Dynamic sequence lengths, planet-count bucketing, or sparse attention that skips padded slots in compute.
- M2 ADR locked: *"Reject variable-length bucketing"* (`ralplan-planet-self-attention-encoder.md` Q3).

**Verdict:** **No implementation work** unless product reopens the M2 padding ADR. Optional hardening: add a fast-tier test asserting inactive planets receive ~zero attention weight (regression guard only).

---

### F2 — Categorical value head

**What exists:** scalar `SharedValueHead` / `FormatRoutedValueHead` + scalar PPO value loss.

**Gap:** distributional critic — predict `N` return bins; train with cross-entropy on projected returns; infer value as `Σ p_i · z_i`.

---

### F3 — Decoder recurrent state across turns

**What exists:** GRU over K moves **within** a turn; state re-initialized from `context_query` each `encode_turn`.

**Gap:** carry `decoder_hidden` in rollout scan across env steps; reset on episode termination; thread through policy `apply` for training consistency.

---

### F4 — Continuous ship buckets

**What exists:** discrete `ship_bucket_count` categoricals; `ship_count_for_bucket_jax` maps bucket → ceil-fraction of available ships.

**Gap:** predict continuous fraction `f ∈ (0,1]` (e.g. Beta or sigmoid-scaled scalar); round/clamp at action build; extend shield to validate continuous count.

---

### F5 — Intercept / lookahead

**M4 complete (evidence):**

- `edge_feature_dim = 18`, schema v4, two anchor speeds, `orbital_position_at_step_jax`, goldens in `test_intercept_edge_features.py`.

**M5 remaining:**

| Item | Location | Notes |
|------|----------|-------|
| Intercept-sorted top-K re-rank | `src/jax/features.py` lexsort | TODO from M4; ADR-002 amendment |
| Forward-projected `target_ships` per anchor | `edge.py:63` | TODO(M5) |
| Per-planet τ lookahead features | deferred spec | Fixed-τ planet position channels beyond edge intercept block |
| M4 Phase 4 ablation | deferred | Reward gate for intercept anchors |

---

## Viable Options (implementation gaps)

### F2 — Categorical value head

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A — Recommended** | C51-style: `N=51` fixed bins, `-V_max…V_max`, CE loss on two-hot projected returns | Well-studied; composable with existing PPO shell | Hyperparams: bin count, support range |
| B | Quantile regression (N quantiles) | Direct tail modeling | Different loss plumbing; no standard "value" for GAE without extra work |
| C | Keep scalar + auxiliary distributional head | Backward compatible checkpoints | Two heads to tune; unclear default for PPO bootstrap |

### F3 — Cross-turn decoder carry

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A — Recommended** | Store `dec_hidden` in rollout carry; reset on `done`; optional `model.decoder_carry=true` flag | Clean separation; JIT-friendly fixed shape | Submission runtime must mirror carry; checkpoint param shape unchanged |
| B | Encode prior-turn actions into `global_features` | No rollout schema change | Feature engineering burden; delayed credit |
| C | Transformer over last-T turn embeddings | Rich memory | Much larger scope; conflicts with fixed-turn encoding |

### F4 — Continuous ships

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A | Beta distribution head + sampled/clamped integer ships | Proper generative model | Complex log-prob; shield must iterate or approximate |
| **B — Recommended (phased)** | Scalar fraction head + deterministic `ceil(f * available)` at action build; keep bucket path behind flag | Minimal action-space change; easier shield port | Not truly continuous policy gradient on count |
| C | Finer discrete buckets (e.g. 32) | Trivial | Doesn't meet "continuous" intent |

### F5 — M5 intercept lookahead

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A — Recommended (sequenced)** | (1) intercept top-K re-rank, (2) projected target_ships per anchor, (3) optional planet τ channels | Incremental schema bumps; each step testable | Multiple small schema versions |
| B | All-at-once M5 bundle | One training reset | Harder to attribute ablation lift |
| C | Defer all M5 until M1 factorized decoder complete | Reduces concurrent risk | Delays feature value |

---

## Implementation Milestones

### M5-intercept-lookahead (feature encoding)

**Scope:** Remaining intercept/lookahead items from M4 deferral list.

**Phases:**
1. Intercept-sorted top-K re-rank behind `task.edge_rank_mode=snapshot|intercept_min` (default `snapshot` for golden stability).
2. Forward-projected `target_ships` per anchor in edge catalog (+ schema v5 bump if dims change).
3. Optional planet-level τ lookahead channels (separate from edge block) — requires interview if dims affect checkpoint broadly.
4. Run deferred M4 Phase 4 ablation subset (paired seeds, non-regression + directional reward read).

**Files:** `src/jax/features.py`, `src/features/catalog/`, `checkpoint_compat.py`, goldens, docs.

**Tests:** `make test-domain-features`; `test_intercept_edge_features.py` extensions.

**Dependency:** M4 complete ✓

---

### M5-value (categorical / distributional critic)

**Scope:** Replace or augment scalar `SharedValueHead` with categorical return distribution.

**Phases:**
1. Add `ModelConfig.value_head: distributional` with `value_bins: int = 51`, `value_max: float`.
2. Implement `CategoricalValueHead` — logits `(batch, N)`; value = expected bin center.
3. PPO: project returns to two-hot targets; value loss = CE; optionally keep scalar bootstrap from expected value for GAE.
4. Extend `build_value_head`; checkpoint metadata `value_head_kind`, `value_bins`.
5. Tests: shape, expected-value consistency, PPO smoke with distributional head.

**Files:** `src/jax/policy.py`, `src/jax/ppo_update.py`, `schema.py`, `conf/model/*.yaml`, `tests/test_jax_policy*.py`, `tests/test_ppo_update.py`.

**Dependency:** composable value-head baseline ✓; recommend after M5-intercept-lookahead so ablations aren't confounded.

---

### M6-decoder-carry (cross-turn decoder memory)

**Scope:** Persist decoder GRU hidden state across game turns within an episode.

**Phases:**
1. Add `decoder_hidden: jax.Array | None` to policy forward; initialize from `context_query` when `None`.
2. Extend rollout scan carry + `JaxTransitionBatch` with `decoder_hidden` (reset on terminal).
3. PPO re-eval: pass stored hidden or recompute consistently (document choice: stored preferred for on-policy correctness).
4. Submission runtime: maintain hidden across turns in `submission_runtime.py`.
5. Config flag `model.decoder_carry: bool = false` (default off for checkpoint compat).
6. Tests: carry reset on done; shape smoke; optional slow-tier rollout test.

**Files:** `src/jax/policy.py`, `src/jax/decoders/`, `src/jax/rollout/collect.py`, `src/jax/rollout/types.py`, `src/jax/submission_runtime.py`.

**Dependency:** M1 factorized decoder stable (executing); avoid parallel decoder API churn.

---

### M6-ships (continuous ship fraction)

**Scope:** Replace discrete bucket categorical with continuous fraction prediction.

**Phases:**
1. Add `task.ship_action_mode: buckets|continuous_fraction` (default `buckets`).
2. Decoder: single sigmoid output for fraction per step (or Beta for stochastic policy).
3. Action build: `ships = clamp(ceil(f * available), 1, available)`.
4. Shield: validate continuous count via existing trajectory checks (no bucket enumeration).
5. PPO: Gaussian/Beta log-prob or treat as discrete after quantization (document gradient choice).
6. Schema/checkpoint bump for action head shape change.
7. Tests: fraction→count mapping, shield legality, PPO shape smoke.

**Files:** `src/jax/policy.py`, `src/jax/decoders/`, `src/game/trajectory_shield.py`, `src/opponents/jax_actions/builders.py`, `src/jax/ppo_update.py`, `schema.py`.

**Dependency:** M1 decoder complete; strongly couples to shield — do not parallel with `thin-trajectory-shield` until scoped.

---

## ADR

**Decision:** Treat F1 masked attention as **complete**; schedule four implementation milestones for remaining gaps with dependency order M5I → M5V → M6B/M6S.

**Drivers:**
- M2 already decided fixed-P masked attention for JIT stability.
- M4 intercept anchors shipped; remaining lookahead is incremental.
- Distributional critic and continuous ships both invalidate checkpoints — sequence them for clean ablations.
- Cross-turn decoder carry is independent but touches rollout/submission — flag-gated default off.

**Alternatives rejected:**
- Reopen variable-length attention (M2) — no evidence of ROI vs JIT cost.
- Bundle all five into one milestone — critic flagged untestable confounds.
- Implement continuous ships before M1 decoder lands — decoder API still in flux.

**Consequences:**
- Four new manifest entries recommended when execution starts.
- F1 optional: one regression test only.
- Default configs unchanged until milestones execute and ablate.

---

## Test Strategy

| Milestone | Fast tier | Slow tier (user approval) |
|-----------|-----------|---------------------------|
| F1 guard | Optional encoder mask test | — |
| M5-intercept | `make test-domain-features` | — |
| M5-value | `test_jax_policy*.py`, `test_ppo_update.py` | rollout smoke |
| M6-decoder-carry | carry reset unit test | `test_jax_rollout.py` |
| M6-ships | builders + shield tests | PPO smoke |

---

## Consensus Log (iter-1)

| Agent | Verdict |
|-------|---------|
| Planner | F1 closed; four milestones; order M5I → M5V → M6* |
| Architect | Approved — keep fixed-P; flag-gate decoder carry; no bundle |
| Critic | Approved-with-changes — require config flags for carry + continuous ships; separate schema bumps; do not start M6 until M1 decoder executing → complete |

---

## Open Questions for Execution

1. **F1:** Accept M2 closed decision, or reopen variable-length ADR?
2. **M5-value:** Default bin count / support range for Orbit Wars returns?
3. **M6-ships:** Beta vs sigmoid fraction (exploration vs simplicity)?
4. **M5-intercept:** Run deferred M4 ablation before or after top-K re-rank?
