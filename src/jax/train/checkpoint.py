from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import jax
from src.artifacts.checkpoint_compat import (
    checkpoint_feature_metadata,
    feature_metadata,
    load_checkpoint_payload,
    validate_checkpoint_config_compatibility,
    validate_checkpoint_encoder_compatibility,
    validate_checkpoint_feature_compatibility,
    validate_checkpoint_pointer_decoder_compatibility,
)
from src.artifacts.checkpoint_retention import prune_checkpoints
from src.artifacts.pipeline import (
    AsyncArtifactPipeline,
    CheckpointJob,
    CheckpointResult,
    commit_checkpoint_payload,
    load_active_optional_jobs,
    protected_paths_from_jobs,
)
from src.artifacts.promotion import promote_if_better
from src.artifacts.replay import maybe_write_jax_checkpoint_replay
from src.artifacts.run_paths import RunContext, append_produced_artifact
from src.config import TrainConfig
from src.jax.train.queue import (
    queue_optional_jobs_if_due,
    queue_tournament_job_if_eligible,
    start_artifact_worker_if_needed,
)
from src.telemetry.metric_registry import filter_event_record
from src.training.curriculum import CurriculumController


@dataclass(slots=True)
class HistoricalSnapshotPool:
    params: dict
    snapshot_ids: jax.Array
    snapshot_updates: jax.Array
    valid_mask: jax.Array
    next_slot: int = 0
    next_id: int = 1


class _TelemetryLogger(Protocol):
    def log(self, record: dict[str, object], *, step: int) -> None: ...

    def log_promoted_checkpoint(
        self,
        checkpoint_path: Path | str,
        *,
        update: int,
        metric_name: str,
        metric_value: float,
    ) -> None: ...

    def log_artifact(
        self,
        path: Path,
        *,
        name: str,
        artifact_type: str,
    ) -> None: ...


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    """Append a JSON metrics record to ``path``, creating parents as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


def checkpoint_payload_builder(
    train_state: object,
    cfg: TrainConfig,
    *,
    key: jax.Array,
    update: int,
    total_env_steps: int,
    completed_episodes: int,
    curriculum: CurriculumController | None = None,
    historical_pool: HistoricalSnapshotPool | None = None,
) -> Callable[[], dict[str, object]]:
    params = train_state.params
    opt_state = train_state.opt_state
    rng_key = key
    cfg_snapshot = deepcopy(cfg)
    metadata = feature_metadata(
        cfg_snapshot.task,
        model_cfg=cfg_snapshot.model,
    )
    curriculum_state_snapshot = (
        deepcopy(curriculum.state_dict()) if curriculum is not None else None
    )
    historical_pool_snapshot = None
    if historical_pool is not None:
        historical_pool_snapshot = {
            "params": jax.device_get(historical_pool.params),
            "snapshot_ids": jax.device_get(historical_pool.snapshot_ids),
            "snapshot_updates": jax.device_get(historical_pool.snapshot_updates),
            "valid_mask": jax.device_get(historical_pool.valid_mask),
            "next_slot": historical_pool.next_slot,
            "next_id": historical_pool.next_id,
        }

    def build_payload() -> dict[str, object]:
        payload: dict[str, object] = {
            "update": update,
            "params": jax.device_get(params),
            "opt_state": jax.device_get(opt_state),
            "rng_key": jax.device_get(rng_key),
            "config": cfg_snapshot,
            "feature_metadata": metadata,
            "total_env_steps": total_env_steps,
            "completed_episodes": completed_episodes,
        }
        parent = cfg_snapshot.resume_checkpoint or cfg_snapshot.from_promoted
        if parent:
            payload["parent_checkpoint_path"] = str(parent)
        if curriculum_state_snapshot is not None:
            payload["curriculum_state"] = deepcopy(curriculum_state_snapshot)
        if historical_pool_snapshot is not None:
            payload["historical_snapshot_pool"] = historical_pool_snapshot
        return payload

    return build_payload


def restore_historical_snapshot_pool(
    payload: object, fallback: HistoricalSnapshotPool
) -> HistoricalSnapshotPool:
    if not isinstance(payload, dict):
        return fallback
    try:
        return HistoricalSnapshotPool(
            params=jax.device_put(payload["params"]),
            snapshot_ids=jax.device_put(payload["snapshot_ids"]),
            snapshot_updates=jax.device_put(payload["snapshot_updates"]),
            valid_mask=jax.device_put(payload["valid_mask"]),
            next_slot=int(payload.get("next_slot", fallback.next_slot)),
            next_id=int(payload.get("next_id", fallback.next_id)),
        )
    except KeyError:
        return fallback


def restore_curriculum_artifacts(
    checkpoint_path: str,
    curriculum: CurriculumController,
    historical_pool: HistoricalSnapshotPool,
) -> HistoricalSnapshotPool:
    checkpoint = load_checkpoint_payload(checkpoint_path)
    validate_checkpoint_config_compatibility(
        checkpoint, checkpoint_path=checkpoint_path
    )
    if not isinstance(checkpoint, dict):
        return historical_pool
    state = checkpoint.get("curriculum_state")
    if isinstance(state, dict):
        curriculum.load_state_dict(state)
    return restore_historical_snapshot_pool(
        checkpoint.get("historical_snapshot_pool"), historical_pool
    )


def load_jax_checkpoint(
    checkpoint_path: str, train_state: object, cfg: TrainConfig
) -> tuple[object, jax.Array, int, int, int]:
    """Load JAX training state and counters from a checkpoint payload."""

    checkpoint = load_checkpoint_payload(checkpoint_path)
    if not isinstance(checkpoint, dict) or "params" not in checkpoint:
        raise ValueError(
            f"JAX checkpoint must contain a parameter payload: {checkpoint_path}"
        )
    validate_checkpoint_config_compatibility(
        checkpoint, checkpoint_path=checkpoint_path
    )
    validate_checkpoint_feature_compatibility(
        checkpoint, cfg.task, checkpoint_path=checkpoint_path
    )
    stored_metadata = checkpoint_feature_metadata(checkpoint)
    validate_checkpoint_encoder_compatibility(
        stored_metadata,
        cfg,
        checkpoint_path=checkpoint_path,
    )
    validate_checkpoint_pointer_decoder_compatibility(
        stored_metadata,
        cfg,
        checkpoint_path=checkpoint_path,
    )
    params = jax.device_put(checkpoint["params"])
    opt_state = checkpoint.get("opt_state")
    if opt_state is None:
        opt_state = train_state.optimizer.init(params)
    else:
        opt_state = jax.device_put(opt_state)
    checkpoint_update = int(checkpoint.get("update", 0))
    key_payload = checkpoint.get("rng_key")
    key = (
        jax.device_put(key_payload)
        if key_payload is not None
        else jax.random.PRNGKey(cfg.seed + checkpoint_update)
    )
    total_env_steps = int(
        checkpoint.get(
            "total_env_steps",
            checkpoint_update * cfg.training.rollout_steps * cfg.training.num_envs,
        )
    )
    completed_episodes = int(checkpoint.get("completed_episodes", 0))
    return (
        train_state.replace(params=params, opt_state=opt_state),
        key,
        checkpoint_update + 1,
        total_env_steps,
        completed_episodes,
    )


def save_jax_checkpoint(
    run_dir: Path,
    update: int,
    train_state: object,
    cfg: TrainConfig,
    *,
    key: jax.Array,
    total_env_steps: int,
    completed_episodes: int,
    curriculum: CurriculumController | None = None,
    historical_pool: HistoricalSnapshotPool | None = None,
) -> Path:
    """Persist the latest and update-numbered JAX checkpoint payloads."""
    payload = checkpoint_payload_builder(
        train_state,
        cfg,
        key=key,
        update=update,
        total_env_steps=total_env_steps,
        completed_episodes=completed_episodes,
        curriculum=curriculum,
        historical_pool=historical_pool,
    )()
    return commit_checkpoint_payload(run_dir, update, payload)


@dataclass
class CheckpointHandler:
    """Checkpoint I/O, retention, promotion, and post-commit artifact hooks."""

    cfg: TrainConfig
    run_dir: Path
    log_path: Path
    run_context: RunContext
    telemetry: _TelemetryLogger
    artifact_queue_dir: Path
    checkpoint_pipeline: AsyncArtifactPipeline | None
    artifact_worker_state: dict[str, subprocess.Popen[object]] = field(
        default_factory=dict
    )
    checkpoint_failures: list[CheckpointResult] = field(default_factory=list)
    run_promotion_best: float | None = None

    @property
    def artifact_cfg(self):
        return self.cfg.artifacts.artifact_pipeline

    def protected_artifact_paths(self) -> set[Path]:
        paths = {self.run_dir / "jax_ckpt_last.pkl"}
        if self.checkpoint_pipeline is not None:
            paths.update(self.checkpoint_pipeline.protected_paths())
        paths.update(
            protected_paths_from_jobs(
                load_active_optional_jobs(self.artifact_queue_dir)
            )
        )
        return paths

    def handle_results(self, results: list[CheckpointResult]) -> None:
        for result in results:
            event_record = {
                "event": "checkpoint_result",
                "update": result.update,
                "checkpoint_status": result.status,
                "checkpoint_final": result.final,
                "checkpoint_reason": result.reason,
                "checkpoint_error": result.error,
            }
            filtered_event = filter_event_record(event_record, self.cfg)
            append_jsonl(self.log_path, filtered_event)
            self.telemetry.log(filtered_event, step=result.update)
            if result.status == "failed":
                self.checkpoint_failures.append(result)
                continue
            if not result.committed or result.numbered_path is None:
                continue

            append_produced_artifact(
                self.run_context.manifest_path,
                {
                    "kind": "checkpoint",
                    "update": result.update,
                    "path": str(result.numbered_path.resolve()),
                    "final": result.final,
                },
            )

            protected_paths = self.protected_artifact_paths()
            protected_paths.add(result.numbered_path)
            if result.latest_path is not None:
                protected_paths.add(result.latest_path)
            if result.final:
                protected_paths.add(result.numbered_path)
            if self.artifact_cfg.enabled and (
                self.artifact_cfg.replay_async
                or self.artifact_cfg.docker_validation_async
            ):
                job_paths = queue_optional_jobs_if_due(
                    self.cfg,
                    update=result.update,
                    checkpoint_path=result.numbered_path,
                    log_path=self.log_path,
                    queue_dir=self.artifact_queue_dir,
                    result_root=self.run_context.evaluations_dir,
                    queue_replay=self.artifact_cfg.replay_async,
                    queue_docker_validation=self.artifact_cfg.docker_validation_async,
                )
                if job_paths:
                    start_artifact_worker_if_needed(
                        self.cfg,
                        queue_dir=self.artifact_queue_dir,
                        result_root=self.run_context.evaluations_dir,
                        worker_state=self.artifact_worker_state,
                    )
                protected_paths.update(
                    protected_paths_from_jobs(
                        load_active_optional_jobs(self.artifact_queue_dir)
                    )
                )

            retention = self.cfg.artifacts.checkpoint_retention
            pruning = prune_checkpoints(
                self.run_dir,
                log_path=self.log_path,
                keep_last_n=retention.keep_last_n,
                keep_every_n_updates=retention.keep_every_n_updates,
                keep_best_k_by_metric=retention.keep_best_k_by_metric,
                best_metric_name=retention.best_metric_name,
                best_metric_mode=retention.best_metric_mode,
                min_update_for_pruning=retention.min_update_for_pruning,
                dry_run_pruning=retention.dry_run_pruning,
                protected_paths=protected_paths,
            )
            action_label = "would prune" if pruning.dry_run else "pruned"
            print(
                f"checkpoint retention: {action_label} {len(pruning.deleted)} files, "
                f"reclaimed_bytes={pruning.reclaimed_bytes}"
            )
            promotion_attempt, self.run_promotion_best = promote_if_better(
                self.cfg,
                self.run_context,
                checkpoint_path=result.numbered_path,
                update=result.update,
                log_path=self.log_path,
                run_best_value=self.run_promotion_best,
            )
            if promotion_attempt.promoted:
                print(
                    f"Promoted u{result.update} {promotion_attempt.metric_name}="
                    f"{promotion_attempt.metric_value} -> "
                    f"{promotion_attempt.promoted_manifest_path}",
                    flush=True,
                )
                append_jsonl(
                    self.log_path,
                    {
                        "event": "checkpoint_promoted",
                        "update": result.update,
                        "metric_name": promotion_attempt.metric_name,
                        "metric_value": promotion_attempt.metric_value,
                        "promoted_manifest_path": str(
                            promotion_attempt.promoted_manifest_path
                        ),
                    },
                )
                if self.cfg.telemetry.wandb.log_artifacts:
                    self.telemetry.log_promoted_checkpoint(
                        result.numbered_path,
                        update=result.update,
                        metric_name=promotion_attempt.metric_name,
                        metric_value=float(promotion_attempt.metric_value or 0.0),
                    )
            tournament_job = None
            if self.artifact_cfg.enabled:
                tournament_job = queue_tournament_job_if_eligible(
                    self.cfg,
                    update=result.update,
                    checkpoint_path=result.numbered_path,
                    queue_dir=self.artifact_queue_dir,
                    result_root=self.run_context.evaluations_dir,
                    promotion_attempt_reason=promotion_attempt.reason,
                )
            if tournament_job is not None:
                append_jsonl(
                    self.log_path,
                    {
                        "event": "tournament_job_queued",
                        "update": result.update,
                        "job_path": str(tournament_job),
                    },
                )
                start_artifact_worker_if_needed(
                    self.cfg,
                    queue_dir=self.artifact_queue_dir,
                    result_root=self.run_context.evaluations_dir,
                    worker_state=self.artifact_worker_state,
                )
            if self.artifact_cfg.enabled and not self.artifact_cfg.replay_async:
                replay_meta_path = maybe_write_jax_checkpoint_replay(
                    self.cfg,
                    update=result.update,
                    checkpoint_path=result.numbered_path,
                    log_path=self.log_path,
                )
                if replay_meta_path is not None:
                    self.telemetry.log_artifact(
                        replay_meta_path,
                        name=f"replay-meta-u{result.update}",
                        artifact_type="replay_metadata",
                    )

    def build_checkpoint_job(
        self,
        *,
        update: int,
        train_state: object,
        key: jax.Array,
        total_env_steps: int,
        completed_episodes: int,
        curriculum: CurriculumController | None,
        historical_pool: HistoricalSnapshotPool | None,
        final: bool,
    ) -> CheckpointJob:
        return CheckpointJob(
            update=update,
            run_dir=self.run_dir,
            build_payload=checkpoint_payload_builder(
                train_state,
                self.cfg,
                key=key,
                update=update,
                total_env_steps=total_env_steps,
                completed_episodes=completed_episodes,
                curriculum=curriculum,
                historical_pool=historical_pool,
            ),
            final=final,
        )

    def first_failure(self) -> CheckpointResult | None:
        return self.checkpoint_failures[0] if self.checkpoint_failures else None
