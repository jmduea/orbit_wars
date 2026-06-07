---
date: 2026-06-01
topic: launch-hygiene-e2e-throughput
origin: docs/brainstorms/2026-06-01-launch-hygiene-bundle-requirements.md
related_plan: docs/plans/2026-06-01-002-fix-launch-hygiene-throughput-plan.md
---

# Requirements: Launch Hygiene End-to-End Throughput Recovery

## Summary

Restore full training-loop throughput (env steps/sec and samples/sec) to within the derived pass band of a recorded pre-hygiene baseline while keeping launch hygiene always on. Replace microbenchmark-only verification with a short `ow benchmark` full-pipeline gate and baseline artifact first; optimize hot paths only when R1 shows the measured gap exceeds that band.

## Problem Frame

Launch hygiene shipped with correctness and sampler-level performance work. The incremental-carry fix closed a +56% regression on `scripts/benchmark_factorized_sampler.py` at default K, but the blast radius on end-to-end training was underestimated. Microbenchmark green-light did not guarantee full-loop recovery: operators perceive slower training runs, but no paired before/after e2e measurements exist yet — R1 baseline capture validates or falsifies that gap before recovery work proceeds.

The gap is both **verification** (wrong benchmark tier gated merge) and **potential residual cost** (hygiene and replay work outside the microbenchmark's narrow scope). Without a recorded baseline and production-aligned short benchmark, optimization targets stay vague and regressions can recur silently.

## Key Decisions

- **E2e gate over microbenchmark-only.** The factorized sampler microbenchmark remains a fast regression signal but is not sufficient to declare training throughput healthy. The authoritative gate is a short full training loop under `ow benchmark`.

- **Baseline must be captured, not assumed.** Pre-hygiene reference is not yet defined. Recovery work starts by recording exact commit SHA, merge topology, profile overrides, device/backend notes, repeat runs, and key metrics — thresholds follow measured calibration per project policy, not invented round numbers beyond the stated ~10% recovery target used to derive the pass band.

- **Hygiene stays on in production paths.** Prefer always-on hygiene with in-place optimization. A Hydra toggle for dev-only A/B is acceptable second choice. Semantic rule changes are a last resort if measured hotspots cannot be closed within tolerance.

- **Measure-first, then optimize conditionally.** Gate definition and baseline capture precede hot-path fixes. Profiling on the e2e benchmark identifies whether remaining cost lives in rollout sampling, replay/PPO, lookup build, carry memory, or other pipeline stages deferred by the sampler throughput plan. If R1 shows the primary profile is already within the derived pass band, Phase B recovery (R7) is out of scope.

- **Multiple benchmark profiles.** Primary profile matches default daily training (`task=shield_cheap`, current model YAML). At least one secondary profile (e.g. workstation validation overrides already used in training benchmarks) supports cross-config record-keeping; secondary gating is P1, not P0.

- **Pre-hygiene baseline commit.** Baseline capture starts from the **parent of the PR #163 / launch-hygiene merge commit on `main`**, but the baseline artifact must pin the **exact SHA**, document squash vs merge topology, and note any co-landing commits. When attribution is unclear, record a hygiene-on vs hygiene-off A/B on post-merge vs parent.

- **Canonical baseline artifact.** Stored at `docs/benchmarks/launch-hygiene-e2e-baseline.json` (or a dated sibling under `docs/benchmarks/`), referenced by R5 gate and R10 CLI comparison.

## Requirements

### Verification and baseline

R1. Record a **pre-hygiene baseline artifact** at `docs/benchmarks/launch-hygiene-e2e-baseline.json` containing: exact git SHA (with merge-topology notes and co-landing commit list), benchmark profile overrides, device/backend identity, repeat count (N≥3 runs), run date, per-run and aggregate (`mean`, `stddev` or CI band) values for `env_steps_per_sec`, `samples_per_sec`, and `seconds_per_update_mean` on the primary profile, plus an operator-facing wall-clock example (e.g. baseline 42s/update → pass floor 46s/update when the derived band is ~10%).

R2. **Extend** `ow benchmark training` (not a parallel greenfield harness) with throughput gate flags and baseline comparison. Add a sibling subcommand only if extension proves inadequate. Runs warmup + measured updates on the production-aligned path: rollout collection, PPO update, env stepping — not sampler-only isolation.

R3. The e2e benchmark reports at minimum: `env_steps_per_sec`, `samples_per_sec` (must be added to `training_benchmark_payload` if absent), `seconds_per_update_mean`, and compile-to-update-3 timing when available.

R4. Define a **primary profile** (default daily train: `task=shield_cheap`, `model=transformer_factorized`, and other daily-train overrides documented in the baseline artifact) and at least **one secondary profile** for cross-config record-keeping. The primary profile must include `task=shield_cheap` explicitly — `DEFAULT_BENCHMARK_OVERRIDES` in `src/jax/training_benchmark.py` is not equivalent until wrapped or extended. Profile overrides are documented in the baseline artifact and benchmark CLI help.

R5. Post-recovery gate: primary-profile full-loop throughput within the **derived pass band** from the R1 baseline artifact on the **same machine** as baseline capture. Comparisons across device classes are out of scope unless a new baseline is recorded. The band is computed from baseline repeat-run statistics (not a hardcoded constant divorced from measurement noise).

R6. Retain `scripts/benchmark_factorized_sampler.py` (or equivalent) as a **fast tier-1 signal** with its existing tolerance, but document explicitly that passing tier-1 does not satisfy R5.

### Performance recovery (behavior-preserving)

R7. When R1 shows the measured gap exceeds the derived pass band, reduce full-loop overhead while preserving origin bundle launch hygiene semantics (R1–R10 in `docs/brainstorms/2026-06-01-launch-hygiene-bundle-requirements.md`), including builder merge backstop (R7–R8) and factorized opponent path boundaries (R10), plus rollout↔replay parity (origin bundle R9).

R8. Hot-path work must cover **all factorized decoder paths** implicated by profiling — rollout sampling, PPO replay scan, builder/opponent entrypoints, and any code path still invoking prefix-recompute hygiene when incremental carry is available.

R9. No production Hydra toggle to disable hygiene by default. Dev-only toggle for A/B profiling is acceptable if documented and excluded from promotion paths.

### Regression prevention

R10. **(P0)** Add an automated e2e throughput gate via `ow benchmark training` (or documented extension): ~20 total updates with compile-to-update-3 reported separately; pass/fail uses mean `env_steps_per_sec`, `samples_per_sec`, and `seconds_per_update_mean` averaged over the **measured** update window only, compared against `docs/benchmarks/launch-hygiene-e2e-baseline.json` using `--assert-within-pct` or equivalent CLI flag. The gate must run via CLI/subprocess — not pytest wall-time assertions inside the test process. CI tier placement on shared GPU runners follows project cost policy and may remain P1 until a variance budget exists.

R11. **(P0)** Document in `AGENTS.md` and `ow benchmark` help that throughput verification for hygiene-related changes requires the e2e benchmark gate (R10), not the factorized sampler microbenchmark alone.

## Key Flows

F1. **Baseline capture**
- **Trigger:** Recovery effort starts; no authoritative pre-hygiene numbers exist.
- **Steps:** Check out pinned pre-hygiene SHA; run primary profile N≥3 times with warm-up; record metrics, device identity, merge/co-landing notes to `docs/benchmarks/launch-hygiene-e2e-baseline.json`.
- **Outcome:** Baseline artifact referenced by R5 gate and R10 CLI.

F2. **E2e regression detection**
- **Trigger:** Hygiene or adjacent hot-path change lands on a branch.
- **Steps:** Run e2e benchmark on changed branch vs baseline artifact on the same machine; compare `env_steps_per_sec`, `samples_per_sec`, and `seconds_per_update_mean` against the derived pass band.
- **Outcome:** Fail if outside band on primary profile; pass tier-1 microbenchmark alone is insufficient.

F3. **Hotspot-driven optimization (conditional)**
- **Trigger:** R1 shows measured gap exceeds the derived pass band, or post-fix R10 gate fails while microbenchmark passes.
- **Steps:** Profile full benchmark loop; prioritize stages by env-step and update time share; apply behavior-preserving fixes (carry parity completion, lookup cost, replay scan, memory bandwidth); re-run R5/R10 gate.
- **Outcome:** Full loop within band; hygiene behavioral tests remain green. **Early exit:** if R1 gap is already within band, skip Phase B and close on verification-only deliverables (R1–R4, R6, R10, R11).

## Acceptance Examples

AE1. **Covers R5, R6**
- **Given:** Baseline artifact records primary-profile `env_steps_per_sec = X` with derived pass floor `F` (e.g. `F = 0.9 × X` when the band is ~10% of baseline mean).
- **When:** Post-fix branch runs the same profile on the same machine class.
- **Then:** Measured `env_steps_per_sec`, `samples_per_sec`, and `seconds_per_update_mean` are all ≥ their respective pass floors from the baseline artifact.

AE2. **Covers R2, R3**
- **Given:** Operator runs `ow benchmark training` with documented primary overrides.
- **When:** Warmup completes and measured updates finish.
- **Then:** JSON/text summary includes `env_steps_per_sec`, `samples_per_sec`, `seconds_per_update_mean`, and compile-to-update-3 without requiring a separate sampler script.

AE3. **Covers R7, R8**
- **Given:** R1 shows gap exceeds pass band; profiling shows PPO replay hygiene path dominates update time while rollout sampling is within band.
- **When:** Fix migrates replay to shared incremental carry (or equivalent measured optimization).
- **Then:** R10 gate passes; existing hygiene parity tests and factored replay tests pass.

## Success Criteria

**P0 Phase A — verification (always required):** Baseline artifact captured (R1); e2e benchmark extended under `ow benchmark` with primary profile (R2–R4); tier-1 microbenchmark documented as non-authoritative (R6); automated CLI e2e gate operational (R10); operator docs updated (R11); origin bundle hygiene semantics (R1–R10) preserved and origin bundle R11 behavioral tests remain passing.

**P0 Phase B — recovery (conditional):** Primary profile within derived pass band (R5) **only when R1 shows gap exceeds band**; hot-path optimizations (R7–R8) applied until R10 passes.

**P1 (follow-up):** Secondary profile recorded with documented tolerance or known delta; R10 promoted to shared-GPU CI when variance budget exists; extended operator/runbook docs beyond the P0 paragraph.

**Non-goals for this effort:** Tournament win-rate proof; hygiene metric rate targets (R12 from origin bundle); semantic rule reduction unless Phase B optimization paths exhaust; replacing launch hygiene with env-only dedup.

## Scope Boundaries

**In scope:**
- Baseline definition and capture
- `ow benchmark training` extension with throughput gate and documentation
- Behavior-preserving hot-path optimizations identified by e2e profiling (Phase B only)
- Tier-1 microbenchmark retained as supplementary signal

**Deferred for later:**
- Long campaign wall-clock proof (multi-hour trains)
- Kaggle Docker submission path throughput unless primary e2e gate shows replay/env parity gap there
- Telemetry for hygiene block rates
- R10 on shared GPU CI runners until variance budget is calibrated

**Outside this effort's identity:**
- Disabling hygiene in production for throughput
- Changing hygiene semantics without a new behavioral requirements pass
- Gating on learning-signal or tournament metrics

## Dependencies / Assumptions

- Launch hygiene bundle semantics remain as specified in `docs/brainstorms/2026-06-01-launch-hygiene-bundle-requirements.md`.
- Incremental carry work from `docs/plans/2026-06-01-002-fix-launch-hygiene-throughput-plan.md` is landed or partially landed; perceived slowdown is a hypothesis until R1 quantifies the gap.
- `src/jax/training_benchmark.py` is the natural anchor for R2 — production-aligned short runs already compute `env_steps_per_sec`; `samples_per_sec` must be added to the payload for R3/R10.
- Baseline comparisons assume the same machine as R1 capture; cross-machine thresholds require a new baseline artifact.
- Subjective slowdown may be falsified by R1; Phase B is skipped when the primary profile is already within the derived pass band.

## Outstanding Questions

**Deferred to planning:**
- Which secondary profile(s) beyond workstation validation overrides to record first in P1.
- Profiling tooling choice (manual timing splits vs JAX profiler integration) for Phase B only.
- Exact warmup/measured update split within the ~20-update R10 run (defaults acceptable if documented in CLI help).

## Sources / Research

- Origin feature requirements: `docs/brainstorms/2026-06-01-launch-hygiene-bundle-requirements.md`
- Sampler throughput plan (completed, microbenchmark scope): `docs/plans/2026-06-01-002-fix-launch-hygiene-throughput-plan.md`
- Sampler microbenchmark: `scripts/benchmark_factorized_sampler.py`
- Production-aligned short benchmark: `src/jax/training_benchmark.py`, `src/cli/benchmark.py`
- Hygiene hot path: `src/jax/launch_hygiene.py`, `src/jax/action_sampling.py`, `src/jax/factored_sequence_scan.py`
- Project calibration policy: `docs/benchmarks/preflight-calibration.json`, `AGENTS.md` verification thresholds guidance
