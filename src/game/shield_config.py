from __future__ import annotations

from typing import Any

from src.config import TaskConfig
from src.game.constants import MAX_STEPS

SAFE_REASON = "safe"
SUN_REASON = "sun"
BOUNDS_REASON = "bounds"
UNINTENDED_HIT_REASON = "unintended_hit"
HORIZON_REASON = "horizon"


def trajectory_shield_mode(env_cfg: TaskConfig | Any) -> str:
    mode = str(getattr(env_cfg, "trajectory_shield_mode", "exact")).strip().lower()
    if mode in {"none", "disabled", "false"}:
        return "off"
    if mode not in {"off", "cheap", "exact", "tiered"}:
        raise ValueError(
            f"Unsupported trajectory_shield_mode={mode!r}."
            "Expected one of: off, cheap, exact, tiered."
        )
    return mode


def trajectory_shield_final_validate_selected(env_cfg: Any) -> bool:
    return bool(getattr(env_cfg, "trajectory_shield_final_validate_selected", False))


def env_parity_mode(env_cfg: Any) -> str:
    mode = str(getattr(env_cfg, "env_parity_mode", "train")).strip().lower()
    if mode not in {"train", "kaggle", "legacy"}:
        raise ValueError(
            f"Unsupported env_parity_mode={mode!r}. "
            "Expected one of: train, kaggle, legacy."
        )
    return mode


def env_comet_physics_enabled(env_cfg: Any) -> bool:
    """False for legacy (pre-#188 comet-free hot path)."""

    return env_parity_mode(env_cfg) != "legacy"


def rollout_factorized_sampling_mode(env_cfg: Any) -> str:
    mode = (
        str(getattr(env_cfg, "rollout_factorized_sampling", "lattice")).strip().lower()
    )
    if mode not in {"lattice", "selected_validate"}:
        raise ValueError(
            f"Unsupported rollout_factorized_sampling={mode!r}. "
            "Expected one of: lattice, selected_validate."
        )
    return mode


def trajectory_shield_horizon(state_step: int, env_cfg: Any) -> int:
    configured = max(int(getattr(env_cfg, "trajectory_shield_horizon", MAX_STEPS)), 1)
    remaining = max(MAX_STEPS - int(state_step), 0)
    return min(configured, remaining)


def trajectory_shield_epsilon(env_cfg: Any) -> float:
    return max(float(getattr(env_cfg, "trajectory_shield_epsilon", 0.0)), 0.0)


def trajectory_shield_hit_mode(env_cfg: Any) -> str:
    return (
        str(getattr(env_cfg, "trajectory_shield_hit_mode", "selected_target"))
        .strip()
        .lower()
    )
