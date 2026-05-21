from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .conf_schema import (
    ArtifactPipelineConfig,
    EnvConfig,
    TrainConfig,
    register_config_schemas,
)

__all__ = [
    "EnvConfig",
    "ArtifactPipelineConfig",
    "TrainConfig",
    "compose_hydra_train_config",
    "default_train_config_path",
    "load_hydra_train_config",
    "train_config_from_omegaconf",
]


def default_train_config_path() -> Path:
    """Return the repository's default training YAML path."""

    return Path(__file__).resolve().parents[1] / "default_cfg.yaml"


def load_hydra_train_config(path: str | Path) -> TrainConfig:
    """Load training config through Hydra + structured schema validation."""

    config_path = Path(path).resolve()
    register_config_schemas()
    with initialize_config_dir(version_base="1.3", config_dir=str(config_path.parent)):
        composed = compose(config_name=config_path.stem)
    _validate_no_legacy_format_conflicts(composed)
    merged = OmegaConf.merge(OmegaConf.structured(TrainConfig), composed)
    cfg: TrainConfig = OmegaConf.to_object(merged)
    cfg.heldout_eval_seed_set = _parse_seed_set(cfg.heldout_eval_seed_set)
    _validate_train_config(cfg)
    return cfg


def compose_hydra_train_config(overrides: list[str] | None = None) -> TrainConfig:
    """Compose the repository root Hydra config with optional overrides."""

    config_dir = Path(__file__).resolve().parents[1] / "conf"
    register_config_schemas()
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        composed = compose(config_name="config", overrides=overrides or [])
    return train_config_from_omegaconf(composed)


def train_config_from_omegaconf(cfg_raw: Any) -> TrainConfig:
    """Convert a Hydra/OmegaConf object into a validated ``TrainConfig``."""

    _validate_no_legacy_format_conflicts(cfg_raw)
    merged = OmegaConf.merge(OmegaConf.structured(TrainConfig), cfg_raw)
    cfg: TrainConfig = OmegaConf.to_object(merged)
    cfg.heldout_eval_seed_set = _parse_seed_set(cfg.heldout_eval_seed_set)
    _validate_train_config(cfg)
    return cfg


def _validate_train_config(cfg: TrainConfig) -> None:
    ppo = cfg.ppo
    if int(ppo.update_chunk_rows_min) <= 0:
        raise ValueError("ppo.update_chunk_rows_min must be a positive integer.")
    if ppo.update_chunk_rows_max is not None and int(ppo.update_chunk_rows_max) <= 0:
        raise ValueError("ppo.update_chunk_rows_max must be a positive integer when set.")
    if (
        ppo.update_chunk_rows_max is not None
        and int(ppo.update_chunk_rows_max) < int(ppo.update_chunk_rows_min)
    ):
        raise ValueError(
            "ppo.update_chunk_rows_max must be >= ppo.update_chunk_rows_min when both are set."
        )
    if ppo.rollout_microbatch_envs is not None and int(ppo.rollout_microbatch_envs) <= 0:
        raise ValueError("ppo.rollout_microbatch_envs must be a positive integer when set.")

    artifact_pipeline = cfg.artifact_pipeline
    if int(artifact_pipeline.checkpoint_queue_size) <= 0:
        raise ValueError("artifact_pipeline.checkpoint_queue_size must be positive.")
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
            raise ValueError(f"artifact_pipeline.{field_name} must be positive.")
    if artifact_pipeline.replay_backend not in {"docker", "local"}:
        raise ValueError("artifact_pipeline.replay_backend must be 'docker' or 'local'.")
    if artifact_pipeline.docker_player_count not in {"2", "4", "both"}:
        raise ValueError("artifact_pipeline.docker_player_count must be '2', '4', or 'both'.")
    if int(artifact_pipeline.latest_lag_warning_updates) < 0:
        raise ValueError("artifact_pipeline.latest_lag_warning_updates must be non-negative.")
    if not str(artifact_pipeline.queue_dir).strip():
        raise ValueError("artifact_pipeline.queue_dir must be a non-empty relative path.")
    if Path(artifact_pipeline.queue_dir).is_absolute():
        raise ValueError("artifact_pipeline.queue_dir must be relative to the run directory.")

    if not cfg.self_play_enabled:
        if cfg.self_play_pool_size != 0:
            raise ValueError(
                "self_play_pool_size must be 0 when self_play_enabled is false."
            )
        if cfg.self_play_snapshot_interval != 0:
            raise ValueError(
                "self_play_snapshot_interval must be 0 when self_play_enabled is false."
            )
        if cfg.opponent_mix.curriculum:
            raise ValueError(
                "opponent_mix.curriculum must be empty when self_play_enabled is false."
            )
        historical_weight = float(cfg.opponent_mix.weights.get("historical", 0.0))
        if historical_weight > 0.0:
            raise ValueError(
                "opponent_mix.weights.historical must be 0 when self_play_enabled is false."
            )
    else:
        if cfg.self_play_pool_size <= 0:
            raise ValueError(
                "self_play_pool_size must be > 0 when self_play_enabled is true."
            )
        if cfg.self_play_snapshot_interval <= 0:
            raise ValueError(
                "self_play_snapshot_interval must be > 0 when self_play_enabled is true."
            )


def _validate_no_legacy_format_conflicts(cfg_raw: Any) -> None:
    """Reject ambiguous configs that define rollout/grouping fields in both old/new locations."""

    raw = OmegaConf.to_container(cfg_raw, resolve=False) if cfg_raw is not None else {}
    if not isinstance(raw, dict):
        return
    training_format = raw.get("training_format")
    ppo = raw.get("ppo")
    if not isinstance(training_format, dict) or not isinstance(ppo, dict):
        return

    if "rollout_groups" in training_format and "rollout_groups" in ppo:
        raise ValueError(
            "Conflicting rollout group definitions: use only training_format.rollout_groups; "
            "ppo.rollout_groups is deprecated and no longer supported."
        )
    if "phases" in training_format and "phases" in ppo:
        raise ValueError(
            "Conflicting phase definitions: use only training_format.phases; "
            "ppo.phases is deprecated and no longer supported."
        )
    if "num_envs_2p" in ppo or "num_envs_4p" in ppo:
        raise ValueError(
            "ppo.num_envs_2p/ppo.num_envs_4p are deprecated; configure per-format env counts "
            "via training_format.rollout_groups[*].num_envs only."
        )


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
