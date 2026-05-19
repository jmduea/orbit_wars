from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .conf_schema import (
    CheckpointRetentionConfig,
    EnvConfig,
    ModelConfig,
    OpponentMixConfig,
    PPOConfig,
    ReplayConfig,
    TrainConfig,
    TrainingFormatConfig,
    WandBConfig,
    register_config_schemas,
)


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


def train_config_from_omegaconf(cfg_raw: Any) -> TrainConfig:
    """Convert a Hydra/OmegaConf object into a validated ``TrainConfig``."""

    _validate_no_legacy_format_conflicts(cfg_raw)
    merged = OmegaConf.merge(OmegaConf.structured(TrainConfig), cfg_raw)
    cfg: TrainConfig = OmegaConf.to_object(merged)
    cfg.heldout_eval_seed_set = _parse_seed_set(cfg.heldout_eval_seed_set)
    _validate_train_config(cfg)
    return cfg


def load_train_config(path: str | Path) -> TrainConfig:
    """Temporary compatibility adapter; use ``load_hydra_train_config`` directly."""

    warnings.warn(
        "load_train_config() is deprecated; use Hydra-based load_hydra_train_config().",
        DeprecationWarning,
        stacklevel=2,
    )
    return load_hydra_train_config(path)


def train_config_from_dict(data: dict[str, Any]) -> TrainConfig:
    """Temporary compatibility adapter for dictionary-based config loading."""

    warnings.warn(
        "train_config_from_dict() is deprecated; switch entry points to Hydra compose.",
        DeprecationWarning,
        stacklevel=2,
    )
    merged = OmegaConf.merge(OmegaConf.structured(TrainConfig), data)
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
