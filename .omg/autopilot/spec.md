# OMG Autopilot: Feature Encoding v2 Phases 3–5

**Source:** `.omg/specs/deep-interview-feature-encoding-v2-phase3-5.md` (approved)  
**Plan:** `.omg/plans/ralplan-feature-encoding-v2.md` (iteration 4)  
**Started:** 2026-05-25

## Goal

Complete Phase 3 staged curriculum integration, Phase 4 ablation documentation (non-blocking), and Phase 5 hard v2 cutover.

## Execution Summary

### Phase 3
- Added `tests/test_jax_curriculum_v2.py` — 2p-only and 4p-only `self_play_staged` training loop tests
- Fixed `collect_v2.py` missing `samples` metric for curriculum telemetry

### Phase 4
- Added `docs/feature-encoding-v2-ablation-runbook.md`
- Updated `docs/feature-encoding-v2.md` status
- Evidence table deferred (runbook provides commands)

### Phase 5
- `conf/task/default.yaml` → `encoding_version: v2`
- Extended `checkpoint_compat.py` for v2 metadata + v1 rejection
- Added `src/jax/submission_runtime.py` + packager v2 template path
- Tests: `test_checkpoint_compat.py`, packager v2 assertions

## Verification

- `make test-fast` — 104 passed
- `make test-domain-features` — 22 passed
- `make test-domain-policy` — 33 passed
- `tests/test_jax_curriculum_v2.py` — 2 passed (~3 min)
- `tests/test_checkpoint_compat.py` + `test_kaggle_submission_packager.py` — included in test-fast

## Deferred / Manual

- Full ablation evidence table (≥3 seeds, 2000+ updates)
- Kaggle Docker validation (`validate_kaggle_docker_submission.py` without `--skip-docker`)
- v1 encoder deletion (deprecated follow-up)
