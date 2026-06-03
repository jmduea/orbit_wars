---
title: "feat: Seed scheduler U2/U3 + capability-map parity test"
status: active
date: 2026-06-02
origin: docs/plans/2026-06-01-003-feat-seed-scheduler-calibration-plan.md
parent_pr: 184
---

# feat: Seed scheduler U2/U3 + capability-map parity test

## Summary

Close seed-scheduler calibration plan 003 units U2–U3 after GPU U1 completes (self_play reseed 25/50/100 @ 500 updates), and ship the deferred agent-native **capability map ⊆ `ow --help`** regression test from audit 2026-06-02 / plan 016 Open Questions.

## Problem frame

U1 training arms are complete (507/507/506 JSONL lines). In-flight `calibrate-seed-scheduler` (terminal `124252.txt`, replaces missing `92.txt`) still runs post-sweep held-out tournament eval across discovered `seed_sched_cal_*` campaigns. U2 must not start until that process exits. Calibration JSON still has `chosen_interval: null` (9 partial runs). Capability map table is audit-recommended but not yet in `docs/AGENT_CAPABILITIES.md`.

## Requirements

| ID | Requirement |
|----|-------------|
| R1 | After U1 exit: run `ow benchmark calibrate-seed-scheduler --analyze-only --eval-existing` → update `docs/benchmarks/seed-scheduler-calibration.json` + `.md` |
| R2 | If `pick_reseed_interval` returns winner: lock `training.reseed_every_updates` per plan 003 U3; update schema comments / AGENTS if default changes |
| R3 | Add **Capability map** section to `docs/AGENT_CAPABILITIES.md` (operator action → `ow` / `make` command) |
| R4 | Regression test: every capability-map `ow` leaf path appears in CLI `--help` trees (audit #8) |
| R5 | `make test-fast` after code/doc edits |

## Key technical decisions

**KTD1 — U2 is analyze-only.** No competing GPU training; `--eval-existing` reuses checkpoints. Full 15-arm grid if all campaigns have completed runs.

**KTD2 — U3 pins only on measured winner.** If auto-scale `-1` wins at 500u (effective 50), document in JSON whether YAML stays `-1` or pins `50` (plan 003 open question #1).

**KTD3 — Capability map test is help-tree, not e2e.** Parse markdown table; verify `ow` path tokens against subprocess `--help` for `ow`, `eval`, `benchmark`, `runs`, `promote`, `sweep` (and nested `gate`, `results`, `jobs`).

## Scope boundaries

### In scope

U2, U3 (conditional), capability map docs + test.

### Out of scope

- `ow runs archive` / factorized-sampler (plan 016)
- Re-running U1 training

---

## Implementation units

### U1. Seed scheduler analyze-only (U2)

**Requirements:** R1  
**Dependencies:** U1 GPU process exit  
**Approach:** Documented command in plan 003.  
**Verification:** JSON `runs[*].eval_win_rates_by_seed` non-empty where eval ran; `decision` populated or explicit null reason.

### U2. Lock reseed default (U3)

**Requirements:** R2  
**Dependencies:** U1  
**Files:** `conf/training/base.yaml`, `src/config/schema.py`, `AGENTS.md`, calibration md  
**Verification:** `uv run ow train print_resolved_config=true` shows expected `reseed_every_updates`.

### U3. Capability map + parity test

**Requirements:** R3, R4, R5  
**Files:** `docs/AGENT_CAPABILITIES.md`, `tests/test_agent_capability_map.py` (and optional `src/cli/capability_map.py` if parsing reused)  
**Test scenarios:** Each table row `ow …` leaf registered in nested `--help`; `make test-fast` green.

---

## Open questions

| # | Item | Status |
|---|------|--------|
| 1 | U1 terminal completion | **Wait** — `124252.txt` active |
| 2 | Full 15-arm grid vs 12 campaigns on disk | Verify after U2 JSON `runs` length |
| 3 | Pin `-1` vs `50` if winner is auto-scale at 500u | Decide from U2 `decision` |
