---
date: 2026-06-04
topic: action-sampling-scan-scaffolding
status: completed
type: refactor
origin: GitHub #197 — plan 011 U7
---

# refactor: action_sampling sequence scan scaffolding (#197)

## Summary

Extract shared shield-diagnostics helpers used by factorized and flat-edge `jax.lax.scan` paths in `src/jax/action_sampling.py`. Behavior-preserving dedupe from plan 011 U7; does not touch rollout decode/shield hot-path logic (#189) or PPO replay (#200).

## Problem Frame

`_sample_shielded_factored_sequence_with_params` and the flat-edge branch in `_sample_shielded_sequence_with_params` duplicate shield diagnostic init, per-step accumulation, and post-scan rate finalization. Duplication increases merge conflict risk while perf work (#189, #192) edits the same file.

## Requirements

| ID | Requirement |
|----|-------------|
| R1 | Shared helpers for empty diagnostics, per-step merge, and post-scan `legal_non_noop_rate` |
| R2 | No change to sampling outputs, shield modes, or `ShieldedSequenceSample` fields |
| R3 | `make test-fast` and `make test-launch-hygiene-throughput` green |

## Scope Boundaries

**In scope:** `src/jax/action_sampling.py` helper extraction only.

**Deferred:** Generic scan wrapper with per-step kernel callback (follow-up if still needed after helper dedupe).

**Outside scope:** Shield dispatch dedupe (#195), PPO minibatch (#196), perf optimizations.

## Implementation Units

### U1. Shield diagnostics helpers

**Goal:** Single implementation of init / accumulate / finalize for `ShieldDiagnostics`.

**Requirements:** R1, R2

**Files:** `src/jax/action_sampling.py`

**Approach:** Add `_shield_diagnostic_zeros`, `_empty_shield_diagnostics`, `_merge_shield_step_diagnostics`, `_finalize_shield_diagnostics`; replace duplicated blocks in both scan paths.

**Test scenarios:**

| Scenario | Expected |
|----------|----------|
| Launch hygiene factorized sampler microbench | Pass tier-1 throughput gate |
| Existing rollout/sampling tests via test-fast | Unchanged pass set |

**Verification:** `make test-fast`; `make test-launch-hygiene-throughput`

## Key Technical Decisions

**KTD1 — Helpers only, not generic scan.** Factorized carry includes `cumulative_forbidden` and stop/slot sequences; flat-edge carry differs. Shared scan wrapper is higher risk during perf sprint; helpers deliver most duplication removal safely.

## Sources

- GitHub [#197](https://github.com/jonduea/orbit_wars/issues/197)
- `docs/plans/2026-06-03-011-refactor-src-simplification-followup-plan.md` U7
