---
title: "refactor: Share metric parse in handle_results and promotion commit helper"
status: active
date: 2026-06-07
type: refactor
origin: prior /simplify artifacts review (skipped performance items)
---

# refactor: Share metric parse in handle_results and promotion commit helper

## Summary

Reduce duplicate JSONL metric scans on the checkpoint hot path and deduplicate promotion manifest writes by introducing a shared metric cache per `handle_results` invocation and a single `commit_promotion` helper in `promotion_manifest.py`.

## Problem Frame

After each async checkpoint commit, `CheckpointHandler.handle_results` may call `prune_checkpoints` (which scans the full training log when `keep_best_k_by_metric > 0`) and `promote_if_better` (which scans the same log again for the promotion metric). Metric, tournament, and unified promotion paths each repeat the same three-step write sequence: `write_promoted_manifest` → `merge_campaign_manifest` → `append_promotion_index`.

## Requirements

| ID | Requirement |
|----|-------------|
| R1 | At most one full JSONL parse per metric name per `handle_results` committed result |
| R2 | `prune_checkpoints` and `promote_if_better` accept optional pre-parsed metric maps |
| R3 | Single `commit_promotion` helper owns manifest + campaign + index writes |
| R4 | Metric, tournament, and unified promotion paths use the helper without behavior change |
| R5 | Existing tests pass; add targeted tests for parse-once and commit helper |

## Key Technical Decisions

1. **Parse cache scope:** Per committed result inside `handle_results`, build `dict[str, dict[int, float]]` keyed by metric name. Only parse when retention or promotion needs that metric.
2. **Optional injection:** Add optional `metrics_by_update: dict[int, float] | None` to `prune_checkpoints` and `promote_if_better`. When `None`, fall back to `collect_metric_by_update` (preserves call-site compatibility).
3. **Commit helper location:** `commit_promotion` in `promotion_manifest.py` next to existing write helpers; returns promoted manifest path.
4. **No API break:** Public CLI and job payloads unchanged.

## Implementation Units

### U1. Add `commit_promotion` helper

**Goal:** Centralize promoted manifest, campaign manifest merge, and index append.

**Files:** `src/artifacts/promotion_manifest.py`, `tests/test_promotion.py`

**Approach:** Add `commit_promotion(context, *, promoted_payload, campaign_updates, index_record) -> Path` calling existing three helpers. Unit test verifies all three artifacts written.

**Test scenarios:**
- Happy path: helper writes promoted manifest, updates campaign manifest, appends index row
- Test expectation: none for call sites until U3/U4 wire through

### U2. Share metric parse in `handle_results`

**Goal:** Parse each needed metric name once per committed checkpoint result.

**Files:** `src/artifacts/checkpoint_retention.py`, `src/artifacts/promotion.py`, `src/jax/train/checkpoint.py`, `tests/test_artifact_pipeline.py` or `tests/test_promotion.py`

**Approach:**
- Add optional `metrics_by_update` param to `prune_checkpoints` and `promote_if_better`
- In `handle_results`, collect unique metric names from retention + promotion config; parse once each; pass maps through
- Add test that mocks `collect_metric_by_update` and asserts single call when retention and promotion share metric name

**Test scenarios:**
- Shared metric name: `collect_metric_by_update` called once when both retention best-k and promotion use same metric
- Different metric names: at most two parses
- `metrics_by_update=None` fallback still works (existing tests)

### U3. Wire metric promotion through `commit_promotion`

**Goal:** Replace inline triple-write in `promote_if_better`.

**Files:** `src/artifacts/promotion.py`, `tests/test_promotion.py`

**Dependencies:** U1

### U4. Wire tournament promotion through `commit_promotion`

**Goal:** Replace duplicate triple-write in tournament and unified ladder promotion.

**Files:** `src/artifacts/tournament/promotion.py`, `tests/test_tournament.py`

**Dependencies:** U1

## Scope Boundaries

**In scope:** U1–U4 above.

**Deferred to Follow-Up Work:**
- Caching metric maps across multiple `handle_results` invocations within a run
- `promotion_ops.demote_campaign` refactor to use `commit_promotion`
- Duplicate `load_active_optional_jobs` in `handle_results`

## Verification

- `make test-domain-artifacts`
- Targeted tests for U1/U2 new scenarios
