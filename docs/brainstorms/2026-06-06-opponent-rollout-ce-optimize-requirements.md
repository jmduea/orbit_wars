---
title: "Requirements: Opponent rollout ce-optimize (Approach A)"
type: requirements
status: active
date: 2026-06-06
approach: ablation-ladder-ce-optimize
related:
  - docs/solutions/developer-experience/offline-rollout-phase-profile-decoupled-from-jit-collect.md
  - docs/solutions/developer-experience/production-training-throughput-profiling.md
  - docs/benchmarks/cherry-pick-manifest.json
  - docs/benchmarks/launch-hygiene-e2e-baseline-learning-first.json
---

# Requirements: Opponent rollout ce-optimize (Approach A)

## Summary

Define a **`ce-optimize` experiment loop** that reduces **opponent rollout cost** (opponent sampling plus opponent `encode_turn` cache refresh) under **admission-realistic mixed 2p/4p geometry**. Experiments rank on **quick offline phase-profile geometry**; winners require **full admission geometry confirmation** before merge. A one-time **opponent ablation ladder baseline** (pre-loop) pins the worst-case scoring rung before hypotheses run.

## Problem Frame

Post-hygiene training is rollout-dominated. Coarse `--detailed-timing` splits rollout from PPO but not opponent from learner inside collect. Offline phase profiling on **admission-shaped quick geometry** with explicit **`task=map_pool`** measured **~68% opponent**, **~17% policy**, **~10% env_step** — opponent is the largest collect slice in that profile. The admission preset still uses `opponents=noop_only`; the opponent phase bucket includes noop-path collect work (sampling branch + `encode_turn` refresh), not production_mix opponents.

Prior `ce-optimize` work (`optimize/multitask-smoke-throughput`) used **noop opponents** and optimized the **learner** path. Admission is **operator-locked to `opponents=noop_only`**, so opponent-path wins do not move today's admission gate until production training uses richer opponents. This loop is **forward-looking**: make non-noop training affordable while preserving a true noop JAX path as ladder floor.

Inline `telemetry=rollout_phase_timing` on production `ow train` is **out of bounds** — host per-step sync stalls 20+ minutes at 32×256. Measurement must use **offline** `ow benchmark rollout-phase-profile` (integration implementation; main delegates `--repo-root`).

## Key Decisions

**Ablation ladder with worst-case primary score.** Score experiments against the ladder rung with the **highest median `rollout_phase_opponent_fraction`** from the baseline capture — not against noop alone. Rationale: optimize what hurts production training most; noop is calibration floor only.

**Quick rank, full confirm.** Iteration ranking uses quick geometry (2 envs × 8 steps, admission model/format mix). A kept experiment must pass full admission geometry (32 envs × 256 steps, mixed 2p/4p) confirmation before merge. Rationale: minutes per experiment vs. trustworthy admission-shaped proof.

**Offline phase profile as measurement spine.** Primary metric is opponent phase **fraction** from profile JSON, not raw `env_steps_per_sec` at quick geometry. Full-geometry confirmation adds throughput against an opponent-heavy baseline captured during pre-loop setup.

**Pre-loop ladder baseline before ce-optimize hypotheses.** Run the full ladder once and record per-rung opponent fractions **and** throughput at `worst_rung`. Rationale: avoids guessing which rung is worst; seeds hypothesis backlog with measured deltas.

**Per-rung override bundles, not profile labels alone.** JAX rollout samples from `CurriculumController.stage_view.family_probs`, not raw `opponents.mix` weights when curriculum is off. Each ladder rung must ship an explicit Hydra override bundle (see R6). Rationale: `opponents=noop_only` with `curriculum=off` still routes to `latest` neural sampling — verified against composed config.

**Do not change admission's locked noop recipe in this loop.** Ladder rungs use measurement-harness overrides only. Admission gate recipe in `docs/benchmarks/cherry-pick-manifest.json` stays `opponents=noop_only` until a separate operator decision.

**ce-optimize phase mapping.** This doc's **Pre-loop** steps run before `/ce-optimize` Phase 0 (Setup). Ladder capture → ce-optimize Phase 0 spec + harness; worst-rung baseline measurement → Phase 1 (Measurement Scaffolding); hypothesis loop → Phases 2–3.

## Requirements

### Measurement harness

- R1. The loop uses a dedicated `ce-optimize` spec named `opponent-rollout-throughput` with `metric.primary.type: hard`, direction **minimize**, name **`rollout_phase_opponent_fraction_worst_rung`** (median opponent fraction at the pinned worst-case ladder rung, quick geometry).
- R2. **Degenerate gates** (all must pass per experiment): `tests/test_rollout_noop_opponent.py`; config composition test proving each ladder rung's override bundle resolves; benchmark command exits zero; finite phase metrics; GPU backend present when measuring on GPU. Plan may add scripted-opponent smoke tests once `scripted_heavy` bundle exists.
- R3. **Diagnostics** (logged, not gated): per-rung opponent/policy/env/reset/post_step fractions; `rollout_phase_measured_total_seconds`; ladder rung id; `env_steps_per_sec` when throughput leg runs; optional `compile_seconds`.
- R4. Measurement command invokes offline phase profile with `--preset admission`, default quick geometry, and **`--repo-root` equal to the active ce-optimize worktree** (integration when main delegates without a worktree). Later `--train-overrides` win over preset defaults. Cold-cache discipline: unset `JAX_COMPILATION_CACHE_DIR`, set `ORBIT_WARS_PYTEST_JAX_CACHE=0`. Harness must parse `rollout_phase_opponent_fraction` from profile/breakdown JSON — not `multitask_smoke_measure.py` training outputs alone.
- R5. Shared admission-shaped overrides across ladder rungs: mixed 2p/4p (`training=2p4p_32_split` at full geometry; quick mode uses `training=smoke`), `model=transformer_factorized_small`, **`task=map_pool`** (all rungs), `task.candidate_count=3`, `model.max_moves_k=2`, W&B and artifact pipeline off for timing stability. **Only** opponent/curriculum fields vary per rung (R6).

### Opponent ablation ladder

- R6. **Ladder rungs** (fixed order). Each rung adds a documented `overrides[]` bundle stored in `ladder-baseline.json`:

  | Rung | Label | Override intent |
  |------|-------|-----------------|
  | 0 | `noop` | `opponents.mode.opponent=noop` — true JAX noop fast path (not merely `opponents=noop_only` profile label under `curriculum=off`) |
  | 1 | `scripted_heavy` | `curriculum.enabled=true` + staged `opponent_families` for scripted families only (random, nearest_sniper, turtle, opportunistic); **no** latest/historical neural |
  | 2 | `self_play` | `curriculum.enabled=true` + stage with `latest: 1.0` (neural factorized opponents) |
  | 3 | `production_mix` | `opponents=default` + `curriculum=default` (latest + historical weights); **requires** snapshot seeding in offline profiler before measured updates (see Dependencies) |

- R7. **Pre-loop baseline capture:** for each rung, run quick profile (≥3 measured updates after ≥2 warmup); record median opponent fraction; pin **`worst_rung`** = argmax fraction (R8 tie-break); persist `ladder-baseline.json` with per-rung `label`, `overrides[]`, and median fractions. **After pinning `worst_rung`:** run `ow benchmark training` at that rung with admission overrides + `--detailed-timing`, ≥3 repeats; record median `env_steps_per_sec` on the same rung in `ladder-baseline.json` for R14.
- R8. If two rungs tie within measurement noise (`noise_threshold` ≤ 0.03 fraction points), pin the **higher opponent-family complexity** rung (production_mix > self_play > scripted_heavy > noop).
- R9. Hypothesis experiments are always scored at **`worst_rung`** opponent config unless a hypothesis explicitly targets a lower rung (document exception in experiment log).

### ce-optimize execution

- R10. **Mutable scope:** opponent rollout sampling and caching (`src/opponents/jax_actions/`, opponent branches in `src/jax/rollout/collect.py`, opponent-related rollout helpers). No learner K-scan, launch-hygiene carry, or PPO replay changes unless a hypothesis proves opponent-only isolation is impossible.
- R11. **Immutable scope:** integration-repo offline phase profile stack (`integration:src/jax/rollout/collect_timed.py`, `integration:src/jax/rollout_phase_profile.py`, profile rollout group init), measurement harness script once written, opponent parity/validity tests referenced by gates. All experiments run on **integration worktrees**; `--repo-root` must equal the active worktree path under test.
- R12. First run defaults: `execution.mode: serial`, `max_concurrent: 1`, `stopping.max_iterations: 4`, `stopping.max_hours: 1`, stability `repeat_count: 3`, `aggregation: median`, `noise_threshold: 0.03` on opponent fraction.
- R13. **Keep rule:** primary metric improves beyond noise threshold **and** all degenerate gates pass **and** full-geometry confirmation passes (R14).
- R14. **Full-geometry confirmation** (run only for keep candidates): `ow benchmark rollout-phase-profile --full-geometry` at `worst_rung` — opponent fraction must not regress vs. pre-experiment baseline at worst rung; **and** `ow benchmark training` with admission overrides + `worst_rung` opponents, `--detailed-timing`, ≥3 repeats — median `env_steps_per_sec` must improve vs. `ladder-baseline.json` throughput on the same rung, or stay within 5% if opponent fraction improved ≥10% relative.
- R15. Rejected hypothesis categories from prior learner ce-optimize (forbidden-carry reshaping, inactive-env sub-step skip, shield lattice micro-opts without opponent isolation proof) are **out of scope** unless new phase evidence implicates opponent path.

### Hypothesis seed themes (non-exhaustive)

- R16. Initial backlog **should seed from** Pre-loop ladder deltas and profiling gaps; examples only: 4p per-step multi-player `encode_turn` reduction; 2p `opp_batch_cache` refresh elision for families that do not read edge features; historical snapshot pool sampling amortized below per-step vmapped policy calls; scripted-opponent shield path cheaper than neural K-step; family-mixture dispatch that avoids computing unused branches.

## Key Flows

- F1. **Pre-loop — Ladder baseline**
  - **Trigger:** Operator starts `/ce-optimize` with this spec for the first time (or requests fresh baseline).
  - **Steps:** For each ladder rung → quick phase profile → median opponent fraction → pin `worst_rung` → throughput benchmark at `worst_rung` → write `ladder-baseline.json`.
  - **Outcome:** Documented worst-case opponent config, fraction, and throughput baseline; ce-optimize primary metric defined.
  - **Maps to:** ce-optimize Phase 0 (Setup) after this completes.

- F2. **ce-optimize Phase 1 — Baseline measurement**
  - **Trigger:** Pre-loop complete; clean git tree for mutable scope.
  - **Steps:** Run measurement harness at `worst_rung` → record gates, primary metric, diagnostics → user approves proceeding.
  - **Outcome:** `experiment-log.yaml` baseline entry on disk.

- F3. **ce-optimize Phases 2–3 — Experiment loop**
  - **Trigger:** User approves baseline.
  - **Steps:** Generate hypothesis backlog (CP-2) → select hypothesis → worktree experiment → quick profile at `worst_rung` → evaluate gates → append log immediately → if best, run full-geometry confirmation → keep or revert.
  - **Outcome:** Cumulative diff on `optimize/opponent-rollout-throughput` branch; strategy digest updated per batch.

## Success Criteria

- SC1. Pre-loop baseline exists with all four ladder rungs measured, `worst_rung` pinned, opponent fractions **and** `worst_rung` throughput recorded.
- SC2. At least one kept experiment reduces median quick `rollout_phase_opponent_fraction` at `worst_rung` by ≥10% relative vs. loop baseline, with full-geometry confirmation passed.
- SC3. No kept experiment regresses degenerate gates or opponent validity tests.
- SC4. Experiment log and ladder baseline are durable on disk under `.context/compound-engineering/ce-optimize/opponent-rollout-throughput/` for resume after session crash.

## Scope Boundaries

**Deferred for later**

- Learner K-scan / launch-hygiene / `rollout_factorized_sampling=selected_validate` optimization track.
- Changing admission manifest to non-noop opponents.
- Tier-2 `make test-launch-hygiene-e2e-throughput` recalibration for opponent-heavy recipe (run after first kept win, not per experiment).
- LLM-as-judge quality scoring for opponent behavior — hard metrics only.

**Outside this loop's identity**

- Env parity / map-pool bake / compile-cliff work (separate admission picks).
- Throughput wins that only improve noop rung while `worst_rung` fraction is unchanged.

## Dependencies / Assumptions

- Offline rollout phase profiling is landed on **integration** (`ow benchmark rollout-phase-profile`); main repo delegates via `--repo-root`.
- **Pre-loop gate:** `scripted_heavy` override bundle and ladder composition test must exist before Pre-loop starts (ce-plan delivers).
- **Pre-loop gate:** `production_mix` rung requires offline profiler snapshot seeding (extend `run_rollout_phase_profile` or defer rung 3 until seeding exists — ce-plan chooses).
- **Assumption:** quick opponent fractions at `task=map_pool` rank hypotheses that also win at full geometry — validated by R14; if repeated false positives, escalate noise threshold or require full geometry every iteration.
- **Canonical repo:** `optimize/opponent-rollout-throughput` branch and ce-optimize worktrees are created on **`orbit_wars-integration`** (opponent hot path + offline profiler live there). Main repo only dispatches measurement with `--repo-root` pointing at the active integration worktree — never a fixed integration checkout that omits the experiment diff.
- GPU available for measurement; single GPU job at a time per operator convention.
- Prior evidence (~68% opponent at quick map_pool) used as motivation only; Pre-loop re-measures on current commit.

## Outstanding Questions

**Resolve before planning**

- **production_mix snapshot seeding:** extend offline profiler vs. defer rung 3 from Pre-loop (ce-plan must decide).

**Deferred to planning**

- Exact family weights for `scripted_heavy` staged `opponent_families`.
- Harness implementation: new `scripts/ce_optimize/opponent_rollout_measure.py` (recommended over extending `multitask_smoke_measure.py` given different metric spine).

## Sources / Research

- `docs/solutions/developer-experience/offline-rollout-phase-profile-decoupled-from-jit-collect.md` — phase definitions, quick vs full geometry, verified map_pool fractions.
- `docs/solutions/developer-experience/production-training-throughput-profiling.md` — rollout vs PPO coarse split.
- `docs/benchmarks/cherry-pick-manifest.json` — admission locked recipe (`opponents=noop_only`).
- `docs/plans/2026-06-04-009-feat-rollout-selected-action-validation-plan.md` — learner-path ce-optimize; explicit deferral of 4p opponent cache.
- Prior session: [map pool + phase profile](e5ddf466-ecd5-4963-894f-216c10614b49) — offline profiling ship, admission `task=map_pool` gate work.
