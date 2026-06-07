"""JAX training loop and auxiliary checkpoint, telemetry, and rollout helpers."""

from __future__ import annotations

from src.artifacts.replay_schedule import checkpoint_replay_due
from src.jax.train.checkpoint import (
    CheckpointHandler,
    HistoricalSnapshotPool,
    append_jsonl,
    checkpoint_payload_builder,
    load_jax_checkpoint,
    restore_curriculum_artifacts,
    restore_historical_snapshot_pool,
    save_jax_checkpoint,
)
from src.jax.train.loop import run_jax_training
from src.jax.train.metrics import (
    finalize_cross_chunk_rate_metrics,
    merge_metric_dicts,
    sum_metric_dicts,
)
from src.jax.train.queue import (
    queue_optional_jobs_if_due,
    queue_tournament_job_if_eligible,
    start_artifact_worker_if_needed,
)
from src.jax.train.rollout_groups import (
    JaxRolloutGroup,
    active_group_indices,
    configured_rollout_groups,
    empty_per_format_rollout_stats,
    init_rollout_groups,
    replace_rollout_group_state,
)
from src.jax.train.snapshots import (
    add_historical_snapshot,
    init_historical_snapshot_pool,
    snapshot_due,
)
from src.jax.train.state import init_train_state, validate_policy_param_shapes
from src.jax.train.telemetry import (
    build_per_format_timing_metrics,
    build_update_record,
    historical_pool_snapshot_telemetry,
    rollout_metrics_for_update_record,
    split_debug_update_record,
    write_filtered_update_records,
)

__all__ = [
    "CheckpointHandler",
    "HistoricalSnapshotPool",
    "JaxRolloutGroup",
    "append_jsonl",
    "build_per_format_timing_metrics",
    "build_update_record",
    "checkpoint_payload_builder",
    "checkpoint_replay_due",
    "configured_rollout_groups",
    "historical_pool_snapshot_telemetry",
    "init_rollout_groups",
    "init_train_state",
    "load_jax_checkpoint",
    "queue_optional_jobs_if_due",
    "queue_tournament_job_if_eligible",
    "restore_curriculum_artifacts",
    "restore_historical_snapshot_pool",
    "rollout_metrics_for_update_record",
    "run_jax_training",
    "save_jax_checkpoint",
    "split_debug_update_record",
    "start_artifact_worker_if_needed",
    "validate_policy_param_shapes",
    "write_filtered_update_records",
]
