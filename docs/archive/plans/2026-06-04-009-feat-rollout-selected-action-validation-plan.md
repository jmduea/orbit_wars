---
title: "feat: Rollout selected-action validation (encode / K-scan / hygiene)"
type: feat
status: active
date: 2026-06-04
origin: docs/plans/2026-06-01-launch-hygiene-rollout-throughput-design.md
related_plan: docs/solutions/developer-experience/production-training-throughput-profiling.md
---

# Plan: Rollout selected-action validation on factorized K-scan

## Summary

Replace per–sub-step **full cheap trajectory-shield lattice** work in learner rollout sampling with **unshielded legality masks + post-sample validation** of only the chosen `(source, slot, bucket)`, preserving launch-hygiene carry semantics and PPO replay parity. Builds on shipped wins (incremental `factorized_decode_step`, gated shield diagnostic merge) and closes GitHub #189 / design-doc Phase B without further shield micro-opts.

## Problem Frame

ce-optimize on `optimize/multitask-smoke-throughput` and smoke `--detailed-timing` attribution (2026-06-04) show:

- Rollout collect is **~99.7%** of per-update time on smoke; PPO is negligible.
- `task=shield_off` ≈ `cheap` shield on smoke — **shield lattice is not the bottleneck** at 2 envs, but **primary preset** rollout remains dominated by `_sample_shielded_factored_sequence_with_params` (see `docs/plans/2026-06-01-launch-hygiene-rollout-throughput-design.md`).
- Failed experiments: forbidden-carry reshaping, all-inactive fast path, stop-first-only, shield dispatch/hoist — **do not** repeat.
- **Kept:** O(K) `factorized_decode_step` in `lax.scan`; skip diagnostic merge when `trajectory_shield_debug` is false.

Design decision (origin): sample from cheaper masks; validate only the selected launch; reject invalid launches as stop/no-op; keep hygiene carry updates on `launch_valid` after tiered/selected rejects.

## Requirements

| ID | Requirement |
|----|-------------|
| R1 | Add `task.rollout_factorized_sampling`: `lattice` (default) \| `selected_validate` |
| R2 | `selected_validate`: use unshielded bucket masks (edge + ships, no per-K sun stack) for sampling |
| R3 | After sample, pointwise cheap-shield check on selected launch; reject → stop/no-op (same as tiered reject) |
| R4 | Launch-hygiene `cumulative_forbidden` carry unchanged; update only on `launch_valid` after rejects |
| R5 | `tiered` + `trajectory_shield_final_validate_selected`: keep exact selected-launch check after R3 |
| R6 | PPO replay: store shield-only masks in `bucket_mask_stack`; replay hygiene prefix unchanged (R9) |
| R7 | `tests/test_factored_sequence_scan.py` parity + new selected-validate correctness tests |
| R8 | Smoke benchmark path: optional override `task.rollout_factorized_sampling=selected_validate` + `--detailed-timing` |
| R9 | Default production configs remain `lattice` until tier-2 e2e proves `selected_validate` |

## Key Technical Decisions

**KTD1 — Opt-in via task flag.** Default `lattice` preserves current behavior; benchmarks/smoke opt into `selected_validate`. Rationale: semantics change needs golden/replay proof before global default.

**KTD2 — Sampling mask = `_unshielded_factorized_topk_result`.** Reuse existing helper (edge_mask × ship buckets, no sun_cross stack). Rationale: design-doc “cheaper policy masks”; already used for `shield_off`.

**KTD3 — Post-sample cheap check = new pointwise helper.** `selected_factored_launch_passes_cheap_shield_jax` mirrors one cell of `apply_cheap_trajectory_shield_factorized_topk` sun/bucket logic. Rationale: avoid full `(P,K,buckets)` lattice per sub-step.

**KTD4 — No change to encode path in this plan.** `forward_factorized_encode` once per env-step stays; encode fusion (#007) is separate. This plan only changes shield/sampling inside K-scan.

**KTD5 — Verification order.** Unit tests → `make test-fast` → smoke `ow benchmark training --detailed-timing` (3 repeats) → tier-1 `make test-launch-hygiene-throughput` → tier-2 only if smoke shows ≥5% rollout improvement.

## Implementation Units

### U1. Task config surface

**Files:** `src/config/schema.py`, `conf/task/rollout_lattice.yaml`, `conf/task/rollout_selected_validate.yaml`, `tests/test_config_consolidation.py`

**Verification:** `uv run ow train print_resolved_config=true task=rollout_selected_validate` shows field.

### U2. Shield helpers

**Files:** `src/jax/shield/trajectory.py`, `src/jax/shield/__init__.py`, `src/game/shield_config.py` (optional reader)

**Verification:** Unit test: pointwise cheap check matches lattice mask for random single cell.

### U3. Rollout sampling path

**Files:** `src/jax/action_sampling.py`

**Verification:** `tests/test_rollout_selected_action_validation.py` (new): reject sun-blocked launch; hygiene carry matches oracle on valid sequence.

### U4. Replay contract

**Files:** `src/jax/factored_sequence_scan.py` (only if replay log-prob drift); else document no change

**Verification:** `test_rollout_replay_logprob_parity_tiered_shield` and new parity test under `selected_validate`.

### U5. Smoke measurement hook

**Files:** `scripts/ce_optimize/multitask_smoke_measure.py` (override only, immutable per ce-optimize — add override string in plan; implement via conf group not script if harness immutable)

Use Hydra override in benchmark command docs, not harness edit: `task=rollout_selected_validate`.

**Verification:** Dual-run median `env_steps_per_sec` vs lattice on same commit.

## Risks

- **Semantic drift:** Unshielded sampling may admit launches cheap shield would block → must reject post-sample; replay must use stored masks.
- **Tiered interaction:** Order: sample → cheap point check → tiered exact → hygiene carry.
- **Compile cache:** Compare benchmarks with `ORBIT_WARS_PYTEST_JAX_CACHE=0`.

## Deferred

- Global default flip to `selected_validate` on `conf/task/base.yaml`
- Encode-once fusion / 4p opponent cache (#007 Phase B)
- Inactive-env sub-step skip (failed experiment; do not retry without new evidence)
