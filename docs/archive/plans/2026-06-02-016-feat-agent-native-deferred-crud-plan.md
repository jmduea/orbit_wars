---
title: "Agent-native deferred CRUD + context (post-015)"
status: active
date: 2026-06-02
origin: docs/plans/2026-06-02-015-feat-agent-native-audit-gaps-plan.md
parent_pr: 184
---

# Agent-native deferred CRUD and context

## Summary

One PR-sized slice on `feat/seed-scheduler-calibration` (PR #184): safe `ow runs archive`, checkpoint delete primitive, W&B sweep summary in `make agent-context`, calibrate primitive-chain docs, and `ow benchmark factorized-sampler` wrapper (Makefile demotion). Seed-scheduler U1–U3 remain GPU-blocked until the in-flight `calibrate-seed-scheduler` sweep finishes.

## Problem frame

Plan 015 and PR #184 shipped agent-context extensions, `ow sweep cancel`, and learn-proof decomposition. The 2026-06-02 audit still lists run archive, checkpoint delete, sweep context, calibrate analyze/sweep split clarity, and factorized-sampler script-only path. Seed calibration plan 003 U1 is ~62% through self_play arms (GPU active); U2/U3 must not start until 500-line JSONL per arm.

## Requirements

| ID | Requirement |
|----|-------------|
| R1 | `ow runs archive --run <path>` moves run tree under `outputs/archived/`; blocks active queue; requires `--confirm`; supports `--dry-run` JSON |
| R2 | `ow runs checkpoint delete` removes named `.pkl` under run `checkpoints/`; blocks promoted incumbent path; `--dry-run` |
| R3 | `make agent-context` JSON includes `wandb_sweeps` summary via subprocess `ow sweep list` (no JAX import in `agent_context.py`) |
| R4 | `docs/AGENT_CAPABILITIES.md` documents calibrate **analyze-only** vs **sweep** primitive chain |
| R5 | `ow benchmark factorized-sampler` forwards to tier-1 microbench; Makefile target delegates to `ow` |
| R6 | Deprecation stderr on `scripts/validate_kaggle_docker_submission.py` → `ow eval package` |

## Key technical decisions

**KTD1 — Archive destination.** `outputs/archived/campaigns/<campaign>/runs/<run_id>` preserves campaign slug for discovery; never delete promoted manifest targets without explicit checkpoint delete.

**KTD2 — agent_context stays JAX-free.** W&B sweep state via bounded subprocess (`ow sweep list --limit 5`) with timeout; on failure emit `present: false` and `list_command`.

**KTD3 — Seed calibration deferred.** Do not launch competing GPU jobs; document U1 progress in plan Open Questions; U2 analyze-only only after all 15 arms have 500+ JSONL lines.

## Scope boundaries

### In scope

U1–U5 below.

### Deferred

- Capability map parity regression test (audit #8)
- `ow sweep create` local index file (optional future)
- Seed U2/U3 until U1 complete
- Pin `training.reseed_every_updates` in YAML (U3)

---

## Implementation units

### U1. ow runs archive

**Requirements:** R1  
**Files:** `src/cli/runs.py`, `tests/test_cli_runs.py`, `docs/AGENT_CAPABILITIES.md`  
**Test scenarios:** dry-run JSON; archive moves tree; rejects active queue without `--confirm`.

### U2. ow runs checkpoint delete

**Requirements:** R2  
**Files:** `src/cli/runs.py`, `tests/test_cli_runs.py`, `docs/AGENT_CAPABILITIES.md`  
**Test scenarios:** dry-run lists paths; delete removes file; blocks promoted checkpoint path.

### U3. W&B sweep state in agent-context

**Requirements:** R3  
**Files:** `scripts/agent_context.py`, `tests/test_agent_context.py`  
**Test scenarios:** mock subprocess returns JSON; field present with `list_command` fallback.

### U4. Calibrate primitive chain docs

**Requirements:** R4  
**Files:** `docs/AGENT_CAPABILITIES.md`  
**Verification:** doc-only.

### U5. factorized-sampler primitive + script deprecation

**Requirements:** R5, R6  
**Files:** `src/cli/benchmark.py`, `Makefile`, `scripts/validate_kaggle_docker_submission.py`, `tests/test_benchmark_cli.py`, `docs/AGENT_CAPABILITIES.md`  
**Test scenarios:** parser accepts subcommand; dry-run or `--help` exits 0.

---

## Open questions

| # | Item | Status |
|---|------|--------|
| 1 | Seed U1: self_play reseed25 (~24 lines) and reseed50 (~316) incomplete; reseed100 pending | **In progress** — GPU `calibrate-seed-scheduler` running |
| 2 | Seed U2: `--analyze-only --eval-existing` after U1 | **Deferred** |
| 3 | Seed U3: lock `reseed_every_updates` in `conf/training/base.yaml` | **Deferred** — needs U2 decision JSON |
| 4 | Capability map ⊆ `ow --help` regression test | **Deferred** |
