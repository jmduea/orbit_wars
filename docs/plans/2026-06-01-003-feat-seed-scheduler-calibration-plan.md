---
title: "feat: Complete seed scheduler calibration sweep"
status: active
date: 2026-06-01
origin: docs/benchmarks/seed-scheduler-calibration.md
deepened: null
---

# feat: Complete seed scheduler calibration sweep

## Summary

Finish the partial GPU calibration sweep for periodic seed reseeding, run held-out tournament eval on all checkpoints, pick a measured default interval, and update calibration artifacts. Training-side work is already implemented (`ow benchmark calibrate-seed-scheduler`, env reset on reseed, auto-scale `-1` default).

## Problem Frame

Leaderboard games use unseen seeds. Single-seed training is brittle (validation sweep showed ~10% win-rate std across seeds). Nine of fifteen training arms completed (`noop_only` and `random_only` full grids; `self_play_only` missing reseed 25/50/100). No tournament eval has run yet, so defaults remain provisional auto-scale `-1`.

## Requirements

| ID | Requirement |
|----|-------------|
| R1 | Complete missing `self_play_only` training arms at reseed intervals 25, 50, 100 (500 updates each) |
| R2 | Run held-out tournament eval vs `noop` on eval seeds `{0,1,2,3,4,43,44,45,46}` excluding train seed 42 |
| R3 | Apply decision rule: smallest interval maximizing min eval win rate across all three opponent profiles with stability gates |
| R4 | Update `docs/benchmarks/seed-scheduler-calibration.json` and `.md` with measured results |
| R5 | If calibration picks a fixed interval, update `conf/training/base.yaml` and schema comments; otherwise document why auto-scale stays |

## Key Technical Decisions

1. **Decision metric is held-out tournament win rate**, not in-training `overall_win_rate` on seed 42.
2. **Stability gate:** run-mean `|approx_kl| ≤ 0.005`, finite losses (same as validation-seed-sweep).
3. **Do not relax tournament thresholds** to force a pass; derive interval from data only.
4. **Partial sweep analysis** is allowed for interim docs, but default change requires full 15-arm grid + eval.

## Scope Boundaries

### In scope

- GPU training for missing arms
- Tournament eval via existing CLI
- Calibration artifact and default YAML updates

### Deferred to Follow-Up Work

- Plateau-triggered reseed tuning
- Shuffled-pool training via `heldout_eval_seed_set`

### Out of scope

- Changing PPO hyperparameters or model architecture during calibration

---

## Implementation Units

### U1. Finish self_play training arms

**Goal:** Complete reseed 25, 50, 100 for `self_play_only` at 500 updates.

**Requirements:** R1

**Files:**

- `outputs/campaigns/seed_sched_cal_self_play_only_reseed*_u500/` (artifacts)
- `src/jax/seed_scheduler_calibration.py` (only if sweep discovery bug found)

**Approach:** Run `uv run ow benchmark calibrate-seed-scheduler --opponents self_play_only --reseed-intervals 25,50,100 --no-include-total-fifth --total-updates 500`. Remove or ignore failed partial run for reseed 25 if it has no JSONL.

**Test scenarios:** Test expectation: none — GPU artifact generation.

**Verification:** Each campaign has 500+ JSONL lines and `jax_ckpt_last.pkl`.

### U2. Held-out tournament eval

**Goal:** Evaluate all completed checkpoints on held-out seeds.

**Requirements:** R2

**Dependencies:** U1

**Files:**

- `outputs/campaigns/seed_sched_cal_*/evaluations/`
- `docs/benchmarks/seed-scheduler-calibration.json`

**Approach:** `uv run ow benchmark calibrate-seed-scheduler --analyze-only --eval-existing`

**Test scenarios:** Test expectation: none — long-running eval.

**Verification:** JSON runs have non-empty `eval_win_rates_by_seed` and computed min/mean/std.

### U3. Lock defaults and document decision

**Goal:** Apply calibration decision to config and docs.

**Requirements:** R3, R4, R5

**Dependencies:** U2

**Files:**

- `conf/training/base.yaml`
- `src/config/schema.py`
- `docs/benchmarks/seed-scheduler-calibration.md`
- `docs/ONBOARDING.md` (if default changes)
- `AGENTS.md` (if default changes)

**Approach:** If `pick_reseed_interval` returns a fixed value, set `reseed_every_updates` to that value or keep `-1` if auto-scale matches winner. Record decision rationale in calibration JSON.

**Test scenarios:**

- Covers existing `tests/test_seed_scheduler_calibration.py::test_pick_reseed_interval_prefers_higher_min_win_rate`
- Run `make test-fast` after any config/doc edits

**Verification:** Calibration JSON `decision.chosen_interval` is non-null with eval data; Hydra resolves expected reseed value.

---

## Risks and Dependencies

- **GPU time:** ~3 training runs + ~135 tournament matches (15 checkpoints × 9 eval seeds). Serial on one GPU.
- **Interrupted runs:** Incomplete campaigns must be excluded from analysis (already filtered by `record_count > 0`).

## Sources and Research

- `docs/benchmarks/seed-scheduler-calibration.md` — partial sweep status
- `docs/benchmarks/validation-seed-sweep.md` — seed variance evidence
- `src/jax/seed_scheduler_calibration.py` — calibration harness
