# Deep Interview Spec: Feature Encoding v2 — Phases 3–5

**Status:** Draft (pending user approval)  
**Parent spec:** `.omg/specs/deep-dive-feature-encoding.md`  
**Plan:** `.omg/plans/ralplan-feature-encoding-v2.md`  
**Prior interview:** `.omg/specs/deep-interview-feature-encoding-v2-phase1.md`  
**Ambiguity at close:** ~5%

## Goal

Complete the remaining feature-encoding v2 rollout through **Phase 3 curriculum end-to-end**, **Phase 4 ablation documentation (non-blocking)**, and **Phase 5 hard cutover** — flipping the project to v2 as the sole production path regardless of v1/v2 metric comparison.

## Constraints

| Constraint | Source |
|------------|--------|
| Phase 3 exit requires **full `self_play_staged`** (all 3 stages) run **separately** for **2p-only** and **4p-only** before mixed-format training | Interview R3 |
| Shield diagnostics still collected in Phase 3; not a Phase 5 blocker | Ralplan + interview |
| **Ablation gates overridden** — proceed to Phase 5 even if v2 underperforms v1 on win rate, throughput, or shield | Interview R4–R5 (contrarian) |
| Phase 5 = **hard cutover** in one phase: default `encoding_version=v2`, reject v1 checkpoints in v2 runs, remove Python encoder from submission path | Interview R6 |
| v1 path may remain in codebase for tests/rollback hooks until explicit deletion follow-up; production default and submission use v2 only | Derived from hard cutover |
| Side-by-side dispatch code stays until v1 deletion; behavior default is v2 after Phase 5 | Derived |
| Do not re-open Phase 0–2 contracts (P=13, E=12, G=46, top-K edges, joint pointer, `gnn_pointer_v2`) | Ralplan lock |
| `jax-ppo-split` gate remains satisfied (complete 2026-05-25) | Manifest |

## Non-Goals

- Re-running Phase 0–2 encoder/policy contract work
- `transformer_v2` (deferred post-cutover)
- Layer D invariant planet sort (deferred)
- Python↔JAX v2 value parity harness (optional follow-up)
- Indefinite dual-default support — hard cutover replaces ablation-gated north star
- Mixed 2p/4p format validation **before** separate 2p-only and 4p-only staged runs complete

## Acceptance Criteria

### Phase 3 — Curriculum end-to-end

1. **2p-only run:** `curriculum=self_play_staged`, 2p format, `encoding_version=v2`, `model=gnn_pointer_v2` — promotes through `bootstrap_random` → `mixed_exploiters` → `self_play_pressure` with curriculum events logged.
2. **4p-only run:** Same curriculum profile and stage progression on 4p-only format.
3. **Shield diagnostics:** Capture `trajectory_shield_legal_non_noop_rate` vs v1 baseline table (informational; not blocking).
4. **Rollout/PPO path:** v2 collect + PPO + joint pointer + shield wired for staged opponents (beyond current 2p random smoke).
5. **Tests:** `make test-domain-jax-env`, `make test-jax`, and new/extended v2 curriculum integration tests green.
6. **Mixed format** training is **out of Phase 3 scope** until both single-format staged runs pass.

### Phase 4 — Ablation documentation (non-blocking)

1. Hydra runbook: matched v1 vs v2 hyperparams, metric extraction template (W&B or local logs).
2. Evidence table in plan appendix: ≥3 seeds recommended but **does not gate Phase 5**.
3. Document observed win rate, throughput, shield deltas for future tuning — recommendation field may say "cutover per interview override" regardless of numbers.
4. Complete field tables in `docs/feature-encoding-v2.md` (v1→v2 mapping, removed fields).

### Phase 5 — Hard cutover

1. Flip Hydra default: `conf/task/default.yaml` → `encoding_version: v2`.
2. `checkpoint_compat.py`: v2 metadata on save; **reject v1 checkpoints** loaded into v2 training runs.
3. Migrate `scripts/validate_kaggle_docker_submission.py` + packager to JAX v2 encode/decode path; **remove Python `encode_turn` from submission path**.
4. `make test` domain targets green; docker validation green on v2.
5. Manifest entries `feature-encoding` + `feature-encoding-v2-plan` → `complete` with evidence paths.

## Assumptions Exposed & Resolved

| Assumption | Resolution |
|------------|------------|
| Phase 3 = 500-update random smoke sufficient? | **No** — full staged curriculum, 2p and 4p separately (R2) |
| Ablation gates block cutover? | **No** — user override; metrics inform docs only (R4–R5) |
| Soft/default-flip cutover? | **No** — hard cutover including submission (R6) |
| Mixed 2p/4p before single-format staged? | **No** — 2p-only and 4p-only staged runs first (R3) |
| Original north star "v1 until v2 wins ablation"? | **Superseded** by interview override for Phases 4–5 |

## Ontology (Key Entities)

| Entity | Role | Stability |
|--------|------|-----------|
| `JaxTurnBatchV2` | Encoder output contract | Frozen (Phase 0) |
| `collect_v2` / `ppo_update_v2` | v2 rollout + learning loop | In progress → complete Phase 3 |
| `self_play_staged` | Curriculum profile for Phase 3 exit | Locked |
| 2p-only / 4p-only format configs | Pre-mixed validation runs | New Phase 3 artifact |
| Ablation evidence table | Informational Phase 4 output | Non-gating |
| `encoding_version` default | Production switch | v1 → v2 Phase 5 |
| Submission packager | Kaggle docker path | JAX v2 only after Phase 5 |

## Interview Transcript

| Round | Dimension | Question (summary) | Answer |
|-------|-----------|-------------------|--------|
| 1 | Goal | Primary session outcome? | Interview all remaining phases 3→5 end-to-end |
| 2 | Success | Phase 3 → 4 boundary? | Full curriculum/staged rollout before ablation |
| 3 | Constraint | Which curriculum for Phase 3 done? | Full `self_play_staged`, separately 2p-only and 4p-only before mixed |
| 4 | Success | Phase 4 cutover bar? | "Cutover no matter what" (freeform) |
| 5 | Challenge | Override ablation north star? | **Yes** — proceed to Phase 5 even if v2 underperforms |
| 6 | Constraint | Phase 5 deprecation aggressiveness? | **Hard cutover** — default v2, reject v1 ckpts, drop Python submission encoder |

## Ambiguity Progress

| Dimension | Initial | Final |
|-----------|---------|-------|
| Goal Clarity (35%) | 55% | 2% |
| Constraint Clarity (25%) | 40% | 5% |
| Success Criteria (25%) | 50% | 8% |
| Context Clarity (15%) | 20% | 5% |
| **Weighted total** | **100%** | **~5%** |
