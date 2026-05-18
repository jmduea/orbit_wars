from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class EnvConfig:
    """Environment and feature-shape configuration shared by all backends."""

    board_size: float = 100.0
    episode_steps: int = 500
    candidate_count: int = 8
    ship_bucket_count: int = 8
    max_planets: int = 48
    max_fleets: int = 256
    player_count: int = 2
    ship_speed: float = 6.0
    max_ships: float = 400.0
    max_production: float = 5.0
    reward_capture_planet: float = 0.0
    reward_ship_delta: float = 0.0
    reward_production_delta: float = 0.0
    reward_terminal_scale: float = 1.0


@dataclass(slots=True)
class ModelConfig:
    """Policy architecture and observation-normalization configuration."""

    architecture: str = "mlp"
    hidden_size: int = 128
    attention_heads: int = 4
    normalize_observations: bool = True
    obs_norm_clip: float = 10.0


@dataclass(slots=True)
class PPOConfig:
    """PPO rollout, optimization, and loss hyperparameters."""

    rollout_steps: int = 32
    num_envs: int = 4
    num_envs_2p: int | None = None
    num_envs_4p: int | None = None
    rollout_groups: list[dict[str, Any]] = field(default_factory=list)
    total_updates: int = 200
    epochs: int = 4
    minibatch_size: int = 512
    gamma: float = 0.99
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5


@dataclass(slots=True)
class TrainingFormatConfig:
    """Curriculum and mixture configuration for multi-format training.

    ``env.player_count`` remains the default environment format. The optional
    ``format_schedule`` and ``format_mix`` lists describe curriculum phases or
    sampling mixtures whose entries can override ``player_count`` and carry
    additional backend-specific metadata, such as update ranges or weights.
    ``rollout_groups`` can be used by trainers that allocate separate rollout
    workers to individual formats.
    """

    format_schedule: list[dict[str, Any]] = field(default_factory=list)
    format_mix: list[dict[str, Any]] = field(default_factory=list)
    rollout_groups: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class TrainConfig:
    """Top-level training configuration loaded from YAML files.

    ``env_backend`` selects either the Kaggle/Python environment or the JAX
    environment. ``rl_backend`` selects the Torch PPO loop or the end-to-end JAX
    PPO loop.
    """

    seed: int = 42
    run_name: str = "orbit_wars_template_ppo"
    device: str = "auto"
    save_dir: str = "artifacts/rl_template"
    checkpoint_every: int = 10
    log_every: int = 1
    opponent: str = "random"
    env_backend: str = "kaggle"
    rl_backend: str = "torch"
    self_play_update_interval: int = 10
    self_play_deterministic: bool = False
    self_play_enabled: bool = False
    self_play_pool_size: int = 5
    self_play_snapshot_interval: int = 25
    self_play_latest_probability: float = 0.5
    alternate_player_sides: bool = True
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    training_format: TrainingFormatConfig = field(default_factory=TrainingFormatConfig)


def default_train_config_path() -> Path:
    """Return the repository's default training YAML path."""

    return Path(__file__).resolve().parents[1] / "default_cfg.yaml"


def load_train_config(path: str | Path) -> TrainConfig:
    """Load a YAML training configuration into a typed ``TrainConfig``."""

    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")
    return train_config_from_dict(data)


def train_config_from_dict(data: dict[str, Any]) -> TrainConfig:
    """Build ``TrainConfig`` from a nested dictionary of overrides."""

    cfg = TrainConfig()
    _update_dataclass(cfg, data, skip={"env", "model", "ppo", "training_format"})
    _update_dataclass(cfg.env, data.get("env", {}))
    _update_dataclass(cfg.model, data.get("model", {}))
    _update_dataclass(cfg.ppo, data.get("ppo", {}))
    _update_dataclass(cfg.training_format, data.get("training_format", {}))
    return cfg


def _update_dataclass(
    instance: Any, values: dict[str, Any], skip: set[str] | None = None
) -> None:
    if not isinstance(values, dict):
        return
    skip = skip or set()
    for key, value in values.items():
        if key in skip or not hasattr(instance, key):
            continue
        default = getattr(instance, key)
        setattr(instance, key, _coerce_value(value, default))


def _coerce_value(value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value
