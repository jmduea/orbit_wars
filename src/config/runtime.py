from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.telemetry.metric_registry import (
    CURRICULUM_PROMOTION_METRIC_NAMES,
    validate_scalar_update_metric_name,
)

from .schema import (
    ArtifactsConfig,
    RewardConfig,
    TaskConfig,
    TrainConfig,
    TrainingConfig,
    register_config_schemas,
)

_CURRICULUM_FAMILIES = {
    "latest",
    "historical",
    "random",
    "noop",
    "nearest_sniper",
    "turtle",
    "opportunistic",
}

_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SAFE_RELATIVE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-/]*$")
_RUN_ID_CACHE: dict[int, str] = {}


def register_runtime_resolvers() -> None:
    """Register OmegaConf resolvers needed before Hydra computes output dirs."""

    if not OmegaConf.has_resolver("orbit_run_id"):
        OmegaConf.register_new_resolver("orbit_run_id", _orbit_run_id, use_cache=True)
    if not OmegaConf.has_resolver("orbit_slug"):
        OmegaConf.register_new_resolver("orbit_slug", _orbit_slug, use_cache=False)
    if not OmegaConf.has_resolver("orbit_safe_rel"):
        OmegaConf.register_new_resolver("orbit_safe_rel", _orbit_safe_rel, use_cache=False)


def _orbit_run_id(seed: int = 42) -> str:
    import uuid
    from datetime import datetime, timezone

    seed_int = int(seed)
    if seed_int in _RUN_ID_CACHE:
        return _RUN_ID_CACHE[seed_int]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    run_id = f"{timestamp}-s{seed_int}-{suffix}"
    _RUN_ID_CACHE[seed_int] = run_id
    return run_id


def _orbit_slug(value: object) -> str:
    raw = str(value).strip()
    if not _SLUG_RE.match(raw):
        raise ValueError(f"unsafe output slug: {raw!r}")
    return raw


def _orbit_safe_rel(value: object) -> str:
    raw = str(value).strip()
    path = Path(raw)
    if (
        not raw
        or path.is_absolute()
        or ".." in path.parts
        or not _SAFE_RELATIVE_RE.match(raw)
    ):
        raise ValueError(f"unsafe relative output path: {raw!r}")
    return raw


__all__ = [
    "ArtifactsConfig",
    "RewardConfig",
    "TaskConfig",
    "TrainingConfig",
    "TrainConfig",
    "compose_hydra_train_config",
    "register_runtime_resolvers",
    "train_config_from_omegaconf",
]


def compose_hydra_train_config(overrides: list[str] | None = None) -> TrainConfig:
    """Compose the repository root Hydra config with optional overrides."""

    config_dir = Path(__file__).resolve().parents[2] / "conf"
    override_list = overrides or []
    register_runtime_resolvers()
    register_config_schemas()
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        composed = compose(config_name="config", overrides=override_list)
    return train_config_from_omegaconf(composed, overrides=override_list)


def config_from_plain(data: dict[str, Any]) -> TrainConfig:
    """Build a TrainConfig from a plain nested mapping."""

    return train_config_from_omegaconf(OmegaConf.create(data))


def train_config_from_omegaconf(
    cfg_raw: Any, overrides: list[str] | None = None
) -> TrainConfig:
    """Convert a Hydra/OmegaConf object into a validated ``TrainConfig``."""

    register_runtime_resolvers()
    merged = OmegaConf.merge(OmegaConf.structured(TrainConfig), cfg_raw)
    cfg: TrainConfig = OmegaConf.to_object(merged)
    cfg.heldout_eval_seed_set = _parse_seed_set(cfg.heldout_eval_seed_set)
    _validate_train_config(cfg)
    return cfg


def _validate_train_config(cfg: TrainConfig) -> None:
    _validate_registered_update_metric_name(
        cfg.artifacts.checkpoint_retention.best_metric_name,
        field_name="artifacts.checkpoint_retention.best_metric_name",
    )
    _validate_registered_update_metric_name(
        cfg.training.plateau_metric,
        field_name="training.plateau_metric",
    )

    task = cfg.task
    if int(task.feature_history_steps) <= 0:
        raise ValueError("task.feature_history_steps must be a positive integer.")
    if task.trajectory_shield_hit_mode not in {"selected_target", "non_friendly"}:
        raise ValueError(
            "task.trajectory_shield_hit_mode must be 'selected_target' or 'non_friendly'."
        )
    if int(task.trajectory_shield_horizon) <= 0:
        raise ValueError("task.trajectory_shield_horizon must be a positive integer.")
    if float(task.trajectory_shield_epsilon) < 0.0:
        raise ValueError("task.trajectory_shield_epsilon must be non-negative.")
    if float(task.ship_feature_scale) <= 0.0:
        raise ValueError("task.ship_feature_scale must be positive.")

    value_head = cfg.model.value_head.strip().lower()
    if value_head not in {"shared", "format_routed"}:
        raise ValueError("model.value_head must be 'shared' or 'format_routed'.")

    training = cfg.training
    if int(training.update_chunk_rows_min) <= 0:
        raise ValueError("training.update_chunk_rows_min must be a positive integer.")
    if training.update_chunk_rows_max is not None and int(training.update_chunk_rows_max) <= 0:
        raise ValueError("training.update_chunk_rows_max must be a positive integer when set.")
    if (
        training.update_chunk_rows_max is not None
        and int(training.update_chunk_rows_max) < int(training.update_chunk_rows_min)
    ):
        raise ValueError(
            "training.update_chunk_rows_max must be >= training.update_chunk_rows_min when both are set."
        )
    if training.rollout_microbatch_envs is not None and int(training.rollout_microbatch_envs) <= 0:
        raise ValueError("training.rollout_microbatch_envs must be a positive integer when set.")
    gae_lambda = float(training.gae_lambda)
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("training.gae_lambda must be in [0, 1].")

    artifact_pipeline = cfg.artifacts.artifact_pipeline
    if int(artifact_pipeline.checkpoint_queue_size) <= 0:
        raise ValueError("artifacts.artifact_pipeline.checkpoint_queue_size must be positive.")
    for field_name in (
        "checkpoint_timeout_seconds",
        "final_flush_timeout_seconds",
        "interrupt_flush_timeout_seconds",
        "exception_flush_timeout_seconds",
        "docker_timeout_seconds",
        "worker_poll_seconds",
        "worker_idle_exit_seconds",
    ):
        if float(getattr(artifact_pipeline, field_name)) <= 0.0:
            raise ValueError(f"artifacts.artifact_pipeline.{field_name} must be positive.")
    if artifact_pipeline.replay_backend not in {"docker", "local"}:
        raise ValueError("artifacts.artifact_pipeline.replay_backend must be 'docker' or 'local'.")
    if artifact_pipeline.docker_player_count not in {"2", "4", "both"}:
        raise ValueError("artifacts.artifact_pipeline.docker_player_count must be '2', '4', or 'both'.")
    if int(artifact_pipeline.latest_lag_warning_updates) < 0:
        raise ValueError("artifacts.artifact_pipeline.latest_lag_warning_updates must be non-negative.")
    if not str(artifact_pipeline.queue_dir).strip():
        raise ValueError("artifacts.artifact_pipeline.queue_dir must be a non-empty relative path.")
    if Path(artifact_pipeline.queue_dir).is_absolute():
        raise ValueError("artifacts.artifact_pipeline.queue_dir must be relative to the run directory.")
    _validate_relative_path_fragment(
        str(artifact_pipeline.queue_dir),
        field_name="artifacts.artifact_pipeline.queue_dir",
    )

    if not str(artifact_pipeline.result_dir).strip():
        raise ValueError(
            "artifacts.artifact_pipeline.result_dir must be a non-empty relative path."
        )
    if Path(artifact_pipeline.result_dir).is_absolute():
        raise ValueError(
            "artifacts.artifact_pipeline.result_dir must be relative to the run directory."
        )
    _validate_relative_path_fragment(
        str(artifact_pipeline.result_dir),
        field_name="artifacts.artifact_pipeline.result_dir",
    )

    _validate_output_config(cfg)

    _validate_curriculum_config(cfg)

    opponents = cfg.opponents
    if not opponents.self_play.enabled:
        if opponents.snapshot.pool_size != 0:
            raise ValueError(
                "opponents.snapshot.pool_size must be 0 when opponents.self_play.enabled is false."
            )
        if opponents.snapshot.interval_updates != 0:
            raise ValueError(
                "opponents.snapshot.interval_updates must be 0 when opponents.self_play.enabled is false."
            )
        historical_weight = float(opponents.mix.weights.get("historical", 0.0))
        if historical_weight > 0.0:
            raise ValueError(
                "opponents.mix.weights.historical must be 0 when opponents.self_play.enabled is false."
            )
    else:
        if opponents.snapshot.pool_size <= 0:
            raise ValueError(
                "opponents.snapshot.pool_size must be > 0 when opponents.self_play.enabled is true."
            )
        if opponents.snapshot.interval_updates <= 0:
            raise ValueError(
                "opponents.snapshot.interval_updates must be > 0 when opponents.self_play.enabled is true."
            )


def _validate_output_config(cfg: TrainConfig) -> None:
    output = cfg.output
    if not str(output.root).strip():
        raise ValueError("output.root must be non-empty.")
    if Path(output.root).is_absolute():
        raise ValueError("output.root must be relative to the workspace by default.")
    if ".." in Path(output.root).parts:
        raise ValueError("output.root must not contain '..'.")
    if not _SLUG_RE.match(str(output.campaign)):
        raise ValueError(
            "output.campaign must be a non-empty slug using letters, numbers, '.', '_', or '-'."
        )
    if not str(output.run_id).strip():
        raise ValueError("output.run_id must be non-empty.")
    if not _SLUG_RE.match(str(output.run_id)):
        raise ValueError(
            "output.run_id must be a non-empty slug using letters, numbers, '.', '_', or '-'."
        )
    if not _SLUG_RE.match(str(output.retention_class)):
        raise ValueError(
            "output.retention_class must be a non-empty slug using letters, numbers, '.', '_', or '-'."
        )
    for field_name in (
        "indexes_dir",
        "cache_dir",
        "wandb_dir",
        "wandb_artifact_dir",
        "wandb_data_dir",
    ):
        value = str(getattr(output, field_name)).strip()
        if not value:
            raise ValueError(f"output.{field_name} must be non-empty.")
        if Path(value).is_absolute():
            raise ValueError(
                f"output.{field_name} must be relative to output.root or the run directory."
            )
        if ".." in Path(value).parts:
            raise ValueError(f"output.{field_name} must not contain '..'.")


def _validate_relative_path_fragment(value: str, *, field_name: str) -> None:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{field_name} must be relative to the run directory.")
    if ".." in path.parts:
        raise ValueError(f"{field_name} must not contain '..'.")


def _validate_curriculum_config(cfg: TrainConfig) -> None:
    curriculum = cfg.curriculum
    if not curriculum.enabled:
        return
    if cfg.format.phases:
        raise ValueError(
            "format.phases is not supported when curriculum.enabled is true; "
            "migrate progressive difficulty to curriculum.stages."
        )
    if not curriculum.stages:
        raise ValueError(
            "curriculum.stages must be non-empty when curriculum.enabled is true."
        )
    snapshot = cfg.opponents.snapshot
    if snapshot.selection not in {"uniform", "recent_biased"}:
        raise ValueError(
            "opponents.snapshot.selection must be 'uniform' or 'recent_biased'."
        )
    if snapshot.fallback != "latest":
        raise ValueError(
            "opponents.snapshot.fallback currently supports only 'latest'."
        )
    seen_ids: set[str] = set()
    for index, stage in enumerate(curriculum.stages):
        if not isinstance(stage, dict):
            raise ValueError("curriculum.stages entries must be mappings.")
        stage_id = str(stage.get("id", "")).strip()
        if not stage_id:
            raise ValueError(f"curriculum.stages[{index}].id must be non-empty.")
        if stage_id in seen_ids:
            raise ValueError(f"curriculum.stages id {stage_id!r} is duplicated.")
        seen_ids.add(stage_id)
        if int(stage.get("min_updates", 0)) < 0:
            raise ValueError(
                f"curriculum.stages[{index}].min_updates must be non-negative."
            )
        if int(stage.get("cooldown_updates", 0)) < 0:
            raise ValueError(
                f"curriculum.stages[{index}].cooldown_updates must be non-negative."
            )
        weights = dict(stage.get("opponent_families", {}))
        if not weights:
            raise ValueError(
                f"curriculum.stages[{index}].opponent_families must be non-empty."
            )
        unknown = sorted(set(weights) - _CURRICULUM_FAMILIES)
        if unknown:
            raise ValueError(
                f"curriculum.stages[{index}].opponent_families contains unknown families: "
                f"{', '.join(unknown)}."
            )
        total = 0.0
        for family, raw_weight in weights.items():
            weight = float(raw_weight)
            if weight < 0.0 or weight == float("inf") or weight != weight:
                raise ValueError(
                    f"curriculum.stages[{index}].opponent_families.{family} must be finite and non-negative."
                )
            total += weight
        if total <= 0.0:
            raise ValueError(
                f"curriculum.stages[{index}].opponent_families must sum to > 0."
            )
        if float(weights.get("historical", 0.0)) > 0.0:
            if int(snapshot.pool_size) <= 0 or int(snapshot.interval_updates) <= 0:
                raise ValueError(
                    "curriculum historical opponents require opponents.snapshot.pool_size > 0 "
                    "and opponents.snapshot.interval_updates > 0."
                )
        promote_if = stage.get("promote_if")
        if promote_if:
            if not isinstance(promote_if, dict):
                raise ValueError(
                    f"curriculum.stages[{index}].promote_if must be a mapping."
                )
            metric = str(promote_if.get("metric", "")).strip()
            if metric not in CURRICULUM_PROMOTION_METRIC_NAMES:
                raise ValueError(
                    f"curriculum.stages[{index}].promote_if.metric must be one of "
                    f"{', '.join(sorted(CURRICULUM_PROMOTION_METRIC_NAMES))}."
                )
            if str(promote_if.get("op", ">=")).strip() not in {">=", ">", "<=", "<"}:
                raise ValueError(
                    f"curriculum.stages[{index}].promote_if.op is invalid."
                )
            if int(promote_if.get("window_updates", 1)) <= 0:
                raise ValueError(
                    f"curriculum.stages[{index}].promote_if.window_updates must be positive."
                )


def _validate_registered_update_metric_name(name: str, *, field_name: str) -> None:
    metric_name = str(name or "").strip()
    if not metric_name:
        raise ValueError(
            f"{field_name} must be a non-empty registered telemetry metric."
        )
    try:
        validate_scalar_update_metric_name(metric_name)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be a registered canonical scalar telemetry metric, got {metric_name!r}."
        ) from exc


def _parse_seed_set(raw: object) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if ".." in text:
            start_s, end_s = text.split("..", maxsplit=1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            return list(range(start, end + step, step))
        if "-" in text and text.count("-") == 1 and text.replace("-", "").isdigit():
            start_s, end_s = text.split("-", maxsplit=1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            return list(range(start, end + step, step))
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    if isinstance(raw, list | tuple | set):
        return [int(v) for v in raw]
    return []
