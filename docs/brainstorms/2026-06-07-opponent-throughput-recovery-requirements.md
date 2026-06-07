---
date: 2026-06-07
topic: opponent-throughput-recovery
---

# Requirements: Opponent Throughput Recovery

## Summary

Recover a viable training loop by reducing the rollout cost of opponent sampling, even if v1 uses weaker rollout opponents than ideal full self-play. Historical self-play is optional or out of the hot path; latest self-play remains available as a pulse, calibration, or gated check so stronger policy behavior is not abandoned.

## Problem Frame

Recent profiling separated opponent rollout cost into sampling/shield work versus encoding work. The bottleneck is the sampler side: opponent sampling can consume roughly 70% of rollout time, while opponent encoding is much smaller. Small sampler tweaks have not made the training loop viable enough.

The immediate product problem is not perfect opponent strength. The immediate problem is that training against the full opponent stack is too slow to be the default path. Scripted opponents are weak relative to leaderboard competition, but they are currently the only practical way to keep training moving. The system needs a throughput recovery mode that accepts weaker rollout opponents while preserving honest evaluation and a path back toward stronger self-play.

## Key Decisions

- **Throughput recovery over semantic purity.** v1 may change rollout opponent composition if that restores usable training throughput. The evaluation and tournament paths remain the source of truth for competitive strength.
- **Historical self-play is demoted.** Historical opponents do not need to stay in normal rollout collection. They may be disabled, moved behind a separate check, or replaced by cheaper diversity mechanisms.
- **Latest self-play remains strategically important.** Full/latest self-play should stay available as a pulse, calibration lane, or gated check. The feature must not make cheap scripted opponents the only path the learner ever sees.
- **Environment complexity can carry more diversity.** Map, task, or curriculum variation may replace some historical-policy diversity as long as mechanical parity remains intact.
- **Measured rollout-phase improvement is the proof.** The first slice is valuable if it knocks opponent sampling share down materially, with an example target of roughly 70% to 60% on a matched rollout-phase profile.

## Actors

- A1. **Training operator** - Chooses the opponent-throughput mode for live experiments and decides whether the measured speedup is worth the weaker rollout opponents.
- A2. **Coding agent** - Plans and implements the recovery path without preserving historical-pool semantics as a default constraint.
- A3. **Learner policy** - Trains mostly against cheap or simplified rollout opponents in v1, with stronger latest-self-play exposure kept alive by pulse/check mechanics.
- A4. **Evaluation and gate tooling** - Verifies that faster training does not create a false sense of competitive progress.

## Requirements

### Throughput recovery

- R1. v1 must provide a training path where historical self-play is removed, bypassed, or no longer sampled in the normal rollout hot path.
- R2. v1 must allow cheap rollout opponents to dominate normal updates, including scripted, random, noop, or environment-complexity-driven variants.
- R3. The recovery path must keep latest self-play available through a bounded pulse, calibration lane, gate, or similar non-default-hot-path mechanism.
- R4. The primary success bar is a matched rollout-phase profile showing a material reduction in opponent sampling share, with roughly 70% to 60% accepted as a meaningful first win.
- R5. The recovery path must preserve JAX environment mechanical validity and player/action legality across 2p and 4p formats.

### Training and evaluation integrity

- R6. Weaker rollout opponents must not weaken evaluation, tournament, package validation, or submit-valid proof paths.
- R7. Training reports must make the active opponent mode visible enough that a reader can distinguish cheap-opponent progress from stronger self-play progress.
- R8. Any latest-self-play pulse or check must expose whether the learner still handles stronger live-policy behavior instead of only exploiting cheap scripted opponents.
- R9. Environment complexity changes used for diversity must preserve mechanical parity expectations and must not depend on Python callbacks or non-JAX hot-path fallbacks.

### Scope and migration

- R10. Existing historical-pool state may remain loadable for checkpoints, but v1 does not need to preserve historical rollout behavior as the default training path.
- R11. The default or recommended recovery profile must be easy for operators and agents to select without rebuilding the broader training pipeline.
- R12. The requirements for future full self-play are preserved: the system should not foreclose a later all-seat/latest-self-play optimization.

## Key Flows

- F1. **Cheap-majority training update**
  - **Trigger:** Operator starts a throughput-focused training run.
  - **Steps:** The run selects a recovery opponent mode; normal rollout updates avoid historical sampling; cheap opponents and environment diversity supply most training pressure; latest self-play is absent or bounded according to the configured pulse/check cadence.
  - **Outcome:** Training proceeds with a lower opponent-sampling share and visible telemetry showing the cheaper opponent composition.

- F2. **Latest self-play pulse or check**
  - **Trigger:** The configured cadence, gate, or operator action requests stronger opponent exposure.
  - **Steps:** The learner faces latest self-play in a bounded context; results are reported separately from cheap-opponent progress.
  - **Outcome:** The system retains a signal for whether cheap-majority training is overfitting or losing contact with stronger policy behavior.

- F3. **Honest evaluation**
  - **Trigger:** A checkpoint is evaluated for progress, promotion, or submit-valid proof.
  - **Steps:** Evaluation uses the existing held-out/tournament validation posture rather than the cheap rollout opponent mix as proof of competitive strength.
  - **Outcome:** Faster rollout training cannot masquerade as leaderboard readiness without passing stronger external checks.

## Acceptance Examples

- AE1. **Covers R1, R4**
  - **Given:** The baseline rollout-phase profile shows opponent sampling consuming about 70% of rollout time.
  - **When:** The recovery opponent mode is profiled on matched geometry.
  - **Then:** Opponent sampling share falls materially, with about 60% accepted as a useful first improvement.

- AE2. **Covers R3, R8**
  - **Given:** Normal updates are dominated by cheap opponents.
  - **When:** A latest-self-play pulse or check runs.
  - **Then:** Its result is reported separately so operators can see whether the learner still handles stronger live policy behavior.

- AE3. **Covers R6**
  - **Given:** A checkpoint was trained under the cheap-majority recovery path.
  - **When:** The checkpoint enters evaluation or tournament proof.
  - **Then:** The proof path uses the same honest evaluation standard as other checkpoints, not the cheap rollout mix.

- AE4. **Covers R10**
  - **Given:** A resumed run contains historical-pool checkpoint state.
  - **When:** The recovery opponent mode is active.
  - **Then:** The run may load or ignore the state safely without sampling historical opponents in normal rollout collection.

## Success Criteria

- SC1. A matched rollout-phase profile shows a meaningful drop in opponent sampling share, with roughly 10 percentage points accepted as sufficient for the first slice.
- SC2. A training run can use the recovery path without invalid JAX states, illegal player actions, or broken 2p/4p rollout semantics.
- SC3. Telemetry or run metadata clearly identifies the cheap-majority opponent mode and any latest-self-play pulse/check results.
- SC4. Evaluation and submit-valid flows remain unchanged in strength and meaning.
- SC5. The requirements preserve a later path to optimize full/latest self-play rather than declaring scripted opponents the long-term endpoint.

## Scope Boundaries

**In scope:**
- Cheap-majority rollout opponent modes for throughput recovery.
- Removing, bypassing, or bounding historical opponents in the rollout hot path.
- Latest-self-play pulse/check exposure.
- Environment-complexity variation as a source of diversity when it preserves mechanical parity.
- Rollout-phase profiling as the primary proof of speedup.

**Deferred for later:**
- A full all-seat/latest-self-play sampler redesign.
- A distilled opponent trained to approximate self-play.
- A new long-term opponent ranking or bracket-training system.
- Proving leaderboard-strength training opponents in the first slice.

**Outside this effort's identity:**
- Weakening evaluation, tournament, Docker/package validation, or submit-valid proof.
- Treating cheap scripted opponents as the final competitive-training strategy.
- Preserving historical-pool sampling semantics as a hard constraint for v1.

## Dependencies / Assumptions

- The current bottleneck diagnosis is accepted for this requirements pass: opponent sampling dominates the profiled rollout share more than opponent encoding.
- Existing scripted opponents are weak, but they are acceptable as v1 throughput scaffolding.
- Environment complexity can vary cheaply while preserving mechanical parity.
- Latest self-play is likely needed later, so the recovery path should not delete or conceptually deprecate it.
- Historical self-play is expensive enough and flexible enough to rethink completely.

## Outstanding Questions

**Deferred to planning:**
- Whether the first implementation should be a config-only opponent composition profile, a separate rollout group/lane, or a new sampler path.
- Which cadence or gate should trigger latest-self-play pulses.
- Which rollout-phase profile should be the canonical matched comparison for SC1.
- Which telemetry fields should represent cheap-opponent progress versus latest-self-play checks.

## Sources / Research

- Current opponent sampler path: `src/opponents/jax_actions/sampling.py`
- Rollout collection path: `src/jax/rollout/collect.py`, `src/jax/rollout/collect_kernel.py`
- Historical snapshot lifecycle: `src/jax/train/snapshots.py`, `src/jax/train/loop.py`
- Opponent family sampling: `src/opponents/pool.py`, `src/opponents/constants.py`
- Rollout profiling guidance: `docs/solutions/developer-experience/offline-rollout-phase-profile-decoupled-from-jit-collect.md`
- Related throughput context: `docs/brainstorms/2026-06-01-launch-hygiene-e2e-throughput-requirements.md`, `docs/plans/2026-06-03-007-feat-jax-encoding-throughput-plan.md`
