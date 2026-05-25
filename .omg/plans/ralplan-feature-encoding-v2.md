# Ralplan: Feature Encoding v2 Greenfield

**Source spec:** `.omg/specs/deep-dive-feature-encoding.md`  
**Trace:** `.omg/specs/deep-dive-trace-feature-encoding.md`  
**Design:** `docs/feature-encoding-v2-design.md`, `docs/feature-encoding-v2-pointer.md`  
**North star:** Orthogonal JAX-only feature stack with joint pointer actions; v1 remains until v2 wins ablation.

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

- [ ] **ADR-001 Action space:** joint pointer index = flat edge list per env `(src_idx, tgt_idx)`; NO_OP token; legality mask rules (active, owned source, ≠self, sun-cross, shield); file touch list.
- [ ] **Schema lock:** P, E, G field tables with dims, normalization, and total float budget (target: ≤200 floats/decision at default H=1, K=4).
- [ ] **Edge representation ADR:** top-K per source vs dense; K default = `task.candidate_count - 1` for ablation continuity.
- [ ] **v1 baseline table:** win rate, `rollout_env_steps_per_sec` (2p/4p), shield rates at promoted config (≥1 seed, document command).
- [ ] **Submission audit:** `scripts/validate_kaggle_docker_submission.py` + packager paths; confirm game API accepts planet-id targets.
- [ ] **Ablation numeric gates** (defaults, user may tune):
  - Win rate: v2 ≥ v1 − **2%** at matched updates/seeds
  - Throughput: v2 rollout env steps/sec ≥ **85%** of v1 on 4p smoke
  - Shield: `trajectory_shield_legal_non_noop_rate` within **±5pp** of v1
  - Evidence: **≥3 seeds**, 2p and 4p stages, **500+ updates** smoke / 2000+ for cutover recommendation
- [ ] **Config sketch:** `task.encoding_version=v1|v2`, `conf/model/gnn_pointer_v2.yaml`
- [ ] **Checkpoint metadata v2:** `schema_version: 2`, `planet_feature_dim`, `edge_feature_dim`, `global_feature_dim`, `edge_layout`
- [ ] **Dependency gate:** `jax-ppo-split` status = complete OR explicit merge freeze on `rollout/`, `ppo_update.py`, `opponents/jax_actions/`

**Deliverables:** `docs/feature-encoding-v2.md` (schema draft), ADR in plan appendix.

**Tests:** none (docs/ADR only).

---

## Phase 1 — JAX Encoder v2

**Scope:**
- `src/features/registry_v2.py` — planet/edge/global schemas
- `src/jax/features_v2.py` — `encode_turn_v2` → `JaxTurnBatchV2`
- `src/config/schema.py` + `conf/task/` — `encoding_version`
- Planet deltas; global-only history when H>1

**Exit:**
- [ ] JIT smoke: `vmap(encode_turn_v2)` at default config
- [ ] `tests/test_feature_encoding_v2_golden.py` — golden vectors (2p, 4p, sun-cross, history)
- [ ] `make test-domain-features` green; **v1 tests unchanged**

---

## Phase 2 — Policy v2 (GNN only)

**Scope:**
- `PlanetEdgeBackboneEncoder` + `gnn_pointer_v2` in `src/jax/policy.py` (or `policy_v2.py`)
- `EncoderOutputV2` contract
- `build_jax_policy` version branch
- `train_state.py` dummy init + shape validation for v2 weights

**Exit:**
- [ ] Forward + sample on synthetic `JaxTurnBatchV2`
- [ ] `make test-domain-policy` + relevant `test-jax` green
- [ ] 10-update training smoke with `encoding_version=v2`

**Deferred:** `transformer_v2` (post-ablation).

---

## Phase 3 — Joint Pointer + Shield + Rollout

**Scope:**
- Joint edge pointer decoder (not adapted slot decoder)
- `trajectory_shield` edge-batch variant
- `opponents/jax_actions/builders.py` — `build_action_from_edge_batch`
- `rollout/types.py`, `collect.py`, `ppo_update.py` — v2 transitions + flatten
- `env.py` — encode dispatch + history type

**Exit:**
- [ ] End-to-end rollout smoke 2p + 4p (500 updates)
- [ ] Shield diagnostics within ±5pp of v1 baseline
- [ ] `make test-domain-jax-env` green

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

## Architect Notes

- Version dispatch only in: `env.py`, `build_jax_policy`, `checkpoint_compat.feature_metadata`, submission script.
- Do not scatter `if v2` in game rules.
- Rollout may store compact edge indices + re-encode in update if memory tight (spike in Phase 0 budget).
- Metrics: version-gated schema slices in `rollout/metrics.py`.

---

## Critic Checklist (Iteration 2)

- [x] Ralplan file exists with phased deliverables
- [x] Spec matches v2 greenfield direction
- [x] Design docs referenced (see `docs/`)
- [x] Phase 0 ADR/metrics before Phase 1 code
- [x] Each phase has pytest/Makefile target
- [x] Numeric cutover criteria (defaults locked, user tunable)
- [x] Edge memory budget addressed (Option A top-K)
- [x] Submission phase before v1 removal talk
- [x] Single v2 policy target for v2 v1 (GNN)
- [x] jax-ppo-split dependency gate
- [x] v1 functional until Phase 5
- [x] Ablation runbook in Phase 4

**User selections:** Option A (top-K edges); execute Phase 0 via team.

**Phase 0 status:** In progress — see `docs/feature-encoding-v2.md` for ADR/schema draft. Remaining: dim lock spike, v1 baseline capture, jax-ppo-split gate.

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
