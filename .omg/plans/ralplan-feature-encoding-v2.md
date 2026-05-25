# Ralplan: Feature Encoding v2 Greenfield

**Source spec:** `.omg/specs/deep-dive-feature-encoding.md`  
**Phase 1 interview:** `.omg/specs/deep-interview-feature-encoding-v2-phase1.md`  
**Trace:** `.omg/specs/deep-dive-trace-feature-encoding.md`  
**Design:** `docs/feature-encoding-v2-design.md`, `docs/feature-encoding-v2-pointer.md`  
**North star:** Orthogonal JAX-only feature stack with joint pointer actions; v1 remains until v2 wins ablation.

**Plan iteration:** 3 (Phase 1 refresh after deep-interview, 2026-05-25)

## RALPLAN-DR Summary

### Principles

1. **Contract before code** — freeze `JaxTurnBatchV2`, edge representation, action index space, and checkpoint metadata in Phase 0.
2. **Side-by-side, not in-place** — v1 path untouched until Phase 5; version dispatch at env/policy/checkpoint boundaries only.
3. **One policy in v2 v1** — ship `gnn_pointer_v2` first; defer `transformer_v2` until after ablation or cutover.
4. **Measure cutover** — numeric win-rate, throughput, and shield gates before v1 deprecation.
5. **Respect active splits** — do not land v2 rollout/PPO/shield changes until `jax-ppo-split` is complete or merge-frozen.

### Decision Drivers

1. **v1 has ~17 duplicated dims and dual Python/JAX drift** — greenfield cheaper long-term than incremental dedup.
2. **Joint pointer rewrites action/shield/rollout** — not an encoder-only project; Phase 0 ADR is mandatory.
3. **Dense `(P,P,E)` at P=60 is ~12× v1 candidate payload** — edge representation must be budgeted in Phase 0.

### Viable Options (Edge + Action)

| Option | Edge representation | Action space | Pros | Cons |
|--------|---------------------|--------------|------|------|
| **A — Recommended** | Top-K edges per owned source (K≈`candidate_count`) | Joint `(src,tgt)` over valid edges + NO_OP | Near v1 memory; true joint pointer; ablation-fair | K tuning; edge ordering contract |
| **B** | Dense `(P,P,E)` static | Joint flat `P²` logits | Simple JIT; full visibility | Memory/compile risk; slow ablation |
| **C** | Planet+edge encoder, **slot action space** | Keep v1 candidate slots | Lower action rewrite | Does not deliver joint pointer goal |
| **D** | Env-loop source + target pointer | Two-stage (not joint) | Smaller action change | Contradicts interview lock on joint pointer |

**Planner recommendation:** **Option A** (top-K hybrid edges + joint pointer over edge list).

**User interview lock:** joint pointer + hybrid edges + planet tensor. Option A satisfies intent with controlled memory.

---

## Phase 0 — Contract & Baseline (NO encoder code)

**Exit criteria (all required):**

- [x] **ADR-001 Action space:** joint pointer index = flat edge list per env `(src_idx, tgt_idx)`; NO_OP token; legality mask rules (active, owned source, ≠self, sun-cross, shield); file touch list.
- [x] **ADR-003 Ship feature scale:** rename `max_ships` → `ship_feature_scale`; document vs fleet-speed `1000` literal and `MAX_FLEET_SPEED`.
- [x] **ADR-004 Symmetry frame:** `θ_ref` = sun → unweighted centroid of owned planets (2p/4p); see `docs/feature-encoding-v2-symmetry.md`.
- [x] **Schema lock:** P=13, E=12, G=46; encoding formulas; float budget spike → `docs/feature-encoding-v2-phase0-results.md`.
- [x] **Edge representation ADR:** top-K per source; K default = `task.candidate_count - 1`.
- [x] **v1 baseline table:** win rate, `rollout_env_steps_per_sec` (2p/4p), shield rates — smoke capture in phase0 results doc.
- [x] **Submission audit:** `scripts/validate_kaggle_docker_submission.py` + packager paths; confirm game API accepts planet-id targets.
- [x] **Ablation numeric gates** (defaults, user may tune):
  - Win rate: v2 ≥ v1 − **2%** at matched updates/seeds
  - Throughput: v2 rollout env steps/sec ≥ **85%** of v1 on 4p smoke
  - Shield: `trajectory_shield_legal_non_noop_rate` within **±5pp** of v1
  - Evidence: **≥3 seeds**, 2p and 4p stages, **500+ updates** smoke / 2000+ for cutover recommendation
- [x] **Config sketch:** `task.encoding_version=v1|v2`, `task.ship_feature_scale`, `conf/model/gnn_pointer_v2.yaml`
- [x] **Checkpoint metadata v2:** `schema_version: 2`, `planet_feature_dim`, `edge_feature_dim`, `global_feature_dim`, `ship_feature_scale`, `edge_layout`
- [x] **Dependency gate:** `jax-ppo-split` status = complete (2026-05-25)

**Deliverables:** `docs/feature-encoding-v2.md` (schema draft), ADR in plan appendix.

**Tests:** none (docs/ADR only).

---

## Phase 1 — JAX Encoder v2 (encoder-only; NEXT)

**Interview lock (2026-05-25):** JAX v2 is the **canonical new implementation**; v1 stays the **runtime default** (`encoding_version=v1`) until Phase 5 cutover. Phase 1 does **not** wire env, rollout, policy, or submission.

### Scope (in)

| Deliverable | Path | Notes |
|-------------|------|-------|
| Schema registry | `src/features/registry_v2.py` | P=13, E=12, G=46; slice helpers mirror v1 registry |
| v2 encoder | `src/jax/features_v2.py` | `encode_turn_v2`, `JaxTurnBatchV2`, `JaxFeatureHistoryV2` |
| Config fields | `src/config/schema.py`, `src/conf_schema.py`, `conf/task/` | `encoding_version`, `ship_feature_scale` (default 1000) |
| Golden tests | `tests/test_feature_encoding_v2_golden.py` | 2p, 4p, sun-cross, H>1 |
| Constants (optional) | `src/game/constants.py` | `BASE_PLANET/EDGE/GLOBAL` v2 dim exports if needed |

### Scope (explicitly out)

- `src/jax/env.py` encode dispatch (`encode_turn` vs `encode_turn_v2`) → **Phase 3**
- `src/jax/policy.py`, rollout, PPO, shield, action builders → **Phases 2–3**
- Python `encode_turn` removal / submission migration → **Phase 5**
- Layer D invariant planet sort → **deferred**
- Python↔JAX v2 value parity harness → **optional follow-up**

### `JaxTurnBatchV2` contract (frozen Phase 0)

Static shapes at default `candidate_count=4` → K=3:

```
planet_features:  (MAX_PLANETS, P)       P=13
planet_mask:      (MAX_PLANETS,)         active planets
edge_features:    (MAX_PLANETS, K, E)    E=12; zeroed when edge_mask=False
edge_mask:        (MAX_PLANETS, K)       valid (src→tgt) pairs; sun-cross masked
edge_src_ids:     (MAX_PLANETS,)         planet id per row (decode side channel)
edge_tgt_ids:     (MAX_PLANETS, K)
global_features:  (G,) or (H * G,)       G=46; H = feature_history_steps
theta_ref:        scalar per env         learner frame reference (ADR-004)
```

Batched leading dim: `(num_envs, …)` after vmap — same convention as v1 `JaxTurnBatch`.

### Encoding semantics

1. **Learner frame:** `θ_ref` = sun → unweighted centroid of learner-owned active planets; planets use sun-polar `(r, θ)`; edges use learner-frame `(Δx, Δy, distance)`.
2. **Top-K edges:** Per active source row, rank targets by distance ascending; deprioritize sun-crossing (JAX v1 spirit). K = `max(0, candidate_count - 1)`.
3. **History:** Planet `ship_delta` from prior step; **global-only** stack when H>1 (no edge history); edges recomputed each step.
4. **Normalization:** `ship_feature_scale` (not `max_ships`) for all ship/fleet fractions in v2 paths.

### Implementation order (recommended)

1. `registry_v2.py` + dim validation tests  
2. Frame helpers (`θ_ref`, canonical polar, sun-cross) — reuse math from `scripts/spike_feature_encoding_v2_phase0.py`  
3. Planet + global encoders (port logic from v1 global block where applicable)  
4. Edge top-K builder + edge feature rows  
5. `JaxFeatureHistoryV2` + global history stack  
6. Golden fixtures + JIT `vmap` smoke  

### Exit criteria

- [x] `registry_v2` validates P=13, E=12, G=46 at import
- [x] `encode_turn_v2` JIT-compiles; `vmap` smoke at default + H=10 config
- [x] Golden vectors: 2p reset, 4p reset, sun-cross fixture, H>1 history reorder
- [x] `encoding_version` + `ship_feature_scale` in TaskConfig/Hydra; default `v1` unchanged behavior
- [x] `make test-domain-features` green; **all v1 tests unchanged**

### Verify

```bash
make test-domain-features
uv run --group dev pytest tests/test_feature_encoding_v2_golden.py -m "not slow and not jax"
uv run --group dev pytest tests/test_feature_encoding_v2_golden.py -m "jax and not slow"  # if marked jax
```

---

## Phase 2 — Policy v2 (GNN only) — COMPLETE (2026-05-25)

**Prerequisite:** Phase 1 exit criteria met.

**Scope:**
- `PlanetEdgeBackboneEncoder` + `gnn_pointer_v2` in `src/jax/policy_v2.py`
- `EncoderOutputV2` contract
- `build_jax_policy` version branch
- `train_state.py` dummy init + shape validation for v2 weights

**Exit:**
- [x] Forward + sample on synthetic `JaxTurnBatchV2`
- [x] `make test-domain-policy` + relevant `test-jax` green
- [x] 10-update training smoke with `encoding_version=v2` (`test_v2_ten_update_training_smoke`)

**Deferred:** `transformer_v2` (post-ablation).

---

## Phase 3 — Joint Pointer + Shield + Rollout — IN PROGRESS

**Prerequisite:** Phase 2 forward/sample smoke green.

**Scope:**
- Joint edge pointer decoder (adapted slot decoder over flat edge list + NO_OP)
- `trajectory_shield` edge-batch variant
- `opponents/jax_actions/builders_v2.py` — `build_action_from_edge_batch`
- `rollout/types.py`, `collect_v2.py`, `ppo_update_v2.py` — v2 transitions + flatten
- `env.py` — encode dispatch + history type (`encode_dispatch.py`)

**Exit:**
- [x] 2p random-opponent rollout + PPO smoke (`test_v2_rollout_and_ppo_update_smoke`)
- [ ] End-to-end rollout smoke 2p + 4p (500 updates)
- [ ] Shield diagnostics within ±5pp of v1 baseline
- [x] `make test-domain-jax-env` green

**Gate:** `jax-ppo-split` complete.

---

## Phase 4 — Ablation & Documentation

**Scope:**
- Hydra ablation runbook (`feature_encoder=v1|v2`, matched hyperparams)
- W&B metric extraction template
- Complete `docs/feature-encoding-v2.md` (field tables, v1→v2 mapping, removed fields)
- Optional: `scripts/ablate_feature_encoder.py`

**Exit:**
- [ ] Evidence table in plan appendix (≥3 seeds, win rate + throughput + shield)
- [ ] Recommendation: cutover / iterate / abort

---

## Phase 5 — Submission + Cutover (only if Phase 4 passes gates)

**Scope:**
- Migrate `validate_kaggle_docker_submission.py` + packager to JAX v2
- Deprecate v1 behind config default flip
- Extend `checkpoint_compat.py` rejection for v1 loads into v2 runs
- Remove Python `encode_turn` from submission path (keep test fixtures until v1 deleted)

**Exit:**
- [ ] Docker validation green on v2
- [ ] Manifest `feature-encoding` → complete with evidence
- [ ] v1 encoder marked deprecated (deletion follow-up)

---

## ADR

**Decision:** Option A — top-K edge list + joint `(source,target)` pointer; GNN policy v2 v1; phased side-by-side.

**Drivers:** Interview-locked greenfield; memory/JIT budget; critic/architect joint-pointer + edge-tensor risks.

**Alternatives rejected:**
- **Dense P² edges (B):** compile/memory regression without proven win-rate upside
- **Slot-preserving encoder (C):** fails joint pointer goal
- **Two-stage pointer (D):** contradicts interview decision

**Consequences:**
- New checkpoint schema; v1 checkpoints incompatible with v2 training
- `candidate_count` retained as K cap for edge list (config continuity)
- Large touch surface: shield, rollout, submission — sequenced after contract phase

---

## Architect Review (Iteration 3 — Phase 1 refresh)

**Approved** with notes:

1. **Phase boundary is correct.** Keeping `encode_turn` as the only env entry until Phase 3 avoids half-wired training paths. Phase 2 can consume `JaxTurnBatchV2` via unit tests and synthetic batches without touching rollout.
2. **`JaxFeatureHistoryV2` should be global-only** — do not port v1's self/candidate history buffers. Store `(H-1)` global frames + prior planet ownership/ships for deltas only.
3. **Edge row indexing:** Use planet **row index** (0..MAX_PLANETS-1) in tensors; `edge_src_ids`/`edge_tgt_ids` carry game planet ids for Phase 3 decode. Do not conflate row index with planet id in masks.
4. **Reuse v1 primitives** where semantics match: `owner_relative_summary`, fleet pressure, sun-cross geometry from `src/jax/features.py` — copy into v2 module or extract shared `jax/feature_geom.py` only if duplication exceeds ~40 lines (prefer copy in Phase 1 to minimize blast radius).
5. **`conf_schema.py` + Hydra:** Add fields to both dataclass layers; validate `encoding_version in {"v1","v2"}` at compose time. Do not rename/remove `max_ships` in Phase 1.
6. **Risk:** Edge top-K + learner frame is the highest complexity slice — implement planet/global first, land golden tests, then edges.

**Phase 2–3 reminder:** Version dispatch stays in `env.py`, `build_jax_policy`, `checkpoint_compat`, submission — not game rules.

---

## Critic Checklist (Iteration 3 — Phase 1 refresh)

- [x] Phase 0 complete with evidence doc
- [x] Phase 1 scope bounded (encoder-only) with explicit out-of-scope list
- [x] `JaxTurnBatchV2` field table matches Phase 0 ADR shapes
- [x] H=1 and H>1 both in Phase 1 exit criteria (interview lock)
- [x] Default `encoding_version=v1` preserves existing training
- [x] Implementation order + verify commands documented
- [x] Each phase has pytest/Makefile target
- [x] v1 tests must not regress
- [x] No env/rollout/policy wiring in Phase 1
- [x] Phase 1 implemented (2026-05-25)

**Consensus:** **Approved** for Phase 1 execution.

**User selections:** Option A (top-K edges); Phase 0 complete; Phase 1 interview locks side-by-side + H>1.

**Status:** Phase 0 **COMPLETE** · Phase 1 **COMPLETE** · Phase 2 **COMPLETE** (2026-05-25) · Phase 3 **IN PROGRESS** (2p random rollout wired).

---

## Phase Boundary Matrix

| Concern | Phase 1 | Phase 2 | Phase 3 | Phase 5 |
|---------|---------|---------|---------|---------|
| `encode_turn_v2` | ✅ build | consume in tests | wire env dispatch | submission |
| `encoding_version` config | ✅ add (default v1) | policy branch | rollout branch | default flip |
| `JaxTurnBatchV2` | ✅ define | policy input | rollout transitions | — |
| Joint pointer / shield | — | stub decoder | ✅ full | — |
| Python encoder | unchanged | unchanged | unchanged | deprecate |
| Golden tests | ✅ v2 only | + policy smoke | + e2e rollout | docker |

---

## Test Matrix

| Phase | Command |
|-------|---------|
| 1 | `make test-domain-features` |
| 2 | `make test-domain-policy` |
| 3 | `make test-domain-jax-env` + `make test-jax` |
| 4 | ablation smoke commands in runbook |
| 5 | `scripts/validate_kaggle_docker_submission.py` |

---

## Appendix: File Touch List (v2)

| Area | Files |
|------|-------|
| Schema | `src/features/registry_v2.py`, `src/config/schema.py`, `conf/task/` |
| Encoder | `src/jax/features_v2.py` |
| Policy | `src/jax/policy.py`, `src/jax/train_state.py` |
| Rollout/PPO | `src/jax/rollout/*`, `src/jax/ppo_update.py` |
| Shield/Actions | `src/game/trajectory_shield.py`, `src/opponents/jax_actions/*` |
| Env | `src/jax/env.py` |
| Checkpoint | `src/artifacts/checkpoint_compat.py` |
| Submission | `scripts/validate_kaggle_docker_submission.py` |
| Docs | `docs/feature-encoding-v2.md` |
| Tests | `tests/test_feature_encoding_v2_golden.py`, policy/rollout v2 tests |
