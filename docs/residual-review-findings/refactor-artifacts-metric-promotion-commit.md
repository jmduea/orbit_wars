# Residual Review Findings

Source: LFG `ce-code-review mode:autofix` for plan `docs/plans/2026-06-07-004-refactor-artifacts-metric-parse-promotion-commit-plan.md`

## Residual Review Findings

- **P2** — Add test: retention and promotion use different metric names → at most two `collect_metric_by_update` calls per `handle_results`.
- **P2** — Add test: `prune_checkpoints(..., metrics_by_update=...)` skips log scan when map is injected.
- **P2** — Pre-existing sync-checkpoint path promotes before update metric is logged (`checkpoint_pipeline=None`); async path unaffected.
- **P3** — Deferred: dedupe second `load_active_optional_jobs` in `CheckpointHandler.handle_results`.
- **P3** — Update docs still referencing deleted `src/artifacts/submit_valid_funnel.py`.
