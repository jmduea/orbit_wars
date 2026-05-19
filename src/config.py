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
    with initialize_config_dir(version_base=None, config_dir=str(config_path.parent)):
        composed = compose(config_name=config_path.stem)
    merged = OmegaConf.merge(OmegaConf.structured(TrainConfig), composed)
    cfg: TrainConfig = OmegaConf.to_object(merged)
    cfg.heldout_eval_seed_set = _parse_seed_set(cfg.heldout_eval_seed_set)
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
    return cfg


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
