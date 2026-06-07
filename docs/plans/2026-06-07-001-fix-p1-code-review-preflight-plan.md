---
date: 2026-06-07
topic: p1-code-review-preflight-fixes
status: active
branch: prep/launch-hygiene-cherry-pick
origin: ce-code-review on commit 4ec161e
---

# Plan: Fix P1 code-review items on integration cherry-pick branch

## Summary

Align tests, task defaults, and tier-2 throughput gate with intentional preflight sweep recipe and learning-first admission baseline. Surgical config/test/Makefile changes only — no revert of committed preflight YAML.

## Problem Frame

Commit `4ec161e` introduced deliberate recipe changes (preflight sweep → `rollout_selected_validate` / `noop_only` / 50 updates; tier-2 Makefile → `2p4p_32_split`) but left drift in compose tests, `task/base.yaml`, and baseline comparison geometry.

## Requirements

| ID | Requirement |
|----|-------------|
| R1 | `tests/test_ssot_wandb_sweep_compose.py` reflects `conf/wandb_sweep/fixed/preflight.yaml` (action_decision=false, total_updates=50, opponents=noop_only) |
| R2 | Tier-2 `make test-launch-hygiene-e2e-throughput` compares apples-to-apples vs learning-first baseline; keeps 2p4p split geometry; runs without geometry crash |
| R3 | `conf/task/base.yaml` default `rollout_factorized_sampling: lattice`; `selected_validate` only in `rollout_selected_validate.yaml` |
| R4 | (P2) Fix stale header comment in `conf/wandb_sweep/fixed/preflight.yaml` |

## Key Technical Decisions

**KTD1 — Test follows recipe, not legacy expectations.** Preflight fixed YAML is authoritative; update compose test assertions only.

**KTD2 — Tier-2 gate uses admission preset + learning-first baseline.** Per `docs/benchmarks/cherry-pick-manifest.json` and `conf/benchmark/gates/admission.yaml`, replace `--preset primary` + old baseline with `--preset admission` + `launch-hygiene-e2e-baseline-learning-first.json` (synced from main harness). Admission preset resolves beat_noop + operator-locked overrides (`2p4p_32_split`, `rollout_steps=256`, noop opponents, shield_cheap).

**KTD3 — Lattice default in task base.** `rollout_selected_validate` task profile already sets `selected_validate`; preflight sweep uses `task=rollout_selected_validate` via fixed yaml.

## Implementation Units

### U1. Update preflight sweep compose test

**Goal:** Test matches intentional preflight recipe.

**Files:** `tests/test_ssot_wandb_sweep_compose.py`

**Approach:** Change assertions at lines 20–24 to `action_decision=false`, `total_updates=50`, `opponents=noop_only`.

**Test scenarios:**
- Happy path: `compose_sweep_gen(["wandb_sweep=preflight"])` returns expected metric groups, updates, opponents.

**Verification:** `uv run --group dev pytest tests/test_ssot_wandb_sweep_compose.py -q` passes.

### U2. Restore lattice default in task base

**Goal:** `selected_validate` does not bleed into shield_cheap / lattice default paths.

**Files:** `conf/task/base.yaml`

**Approach:** Set `rollout_factorized_sampling: lattice` (comment unchanged). Confirm `conf/task/rollout_selected_validate.yaml` still sets `selected_validate`.

**Test scenarios:**
- Test expectation: none — YAML-only; covered by existing task compose tests if any.

**Verification:** Resolved default task uses lattice; `task=rollout_selected_validate` resolves selected_validate.

### U3. Fix tier-2 Makefile throughput gate geometry

**Goal:** Semantically valid baseline compare with 2p4p split; no crash on geometry mismatch.

**Files:** `Makefile`, `docs/benchmarks/launch-hygiene-e2e-baseline-learning-first.json` (copy from main harness)

**Approach:**
- Switch `--preset primary` → `--preset admission`
- Switch baseline path to `launch-hygiene-e2e-baseline-learning-first.json`
- Update Makefile comment to reference learning-first admission recipe
- Copy baseline JSON from main repo (authoritative capture)

**Test scenarios:**
- Gate command composes without error (CPU dry-run not required; geometry validated by benchmark loader tests)

**Verification:** `make test-launch-hygiene-e2e-throughput` starts and reaches compare (may fail throughput on hygiene branch — acceptable).

### U4. Fix preflight YAML header comment (P2)

**Goal:** Comment matches noop_only / selected_validate / 50-update recipe.

**Files:** `conf/wandb_sweep/fixed/preflight.yaml`

**Approach:** Replace stale "mix2p4p self-play" wording with accurate description.

**Verification:** Visual review.

## Scope Boundaries

### In scope
- P1 items #1–#4 and trivial P2 #6

### Out of scope
- Recapturing throughput baselines on GPU
- Reverting preflight YAML recipe
- Full ce-simplify-code (follows LFG)

## Verification

1. `make test-fast` or targeted: `tests/test_ssot_wandb_sweep_compose.py`
2. Optional: `uv run ow train print_resolved_config=true task=default` shows lattice sampling
3. Tier-2 gate command syntax valid (GPU run optional for CI)
