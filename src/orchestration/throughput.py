from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Observed accelerator profile used for first-pass throughput sizing."""

    accelerator_id: str
    gpu_name: str
    memory_gb: float


def estimate_training_overrides(
    hardware: HardwareProfile,
    model_config: Mapping[str, object],
    task_config: Mapping[str, object],
) -> tuple[str, ...]:
    """Estimate stable high-throughput Hydra overrides for a Kaggle worker.

    The estimator is intentionally conservative. Calibration sweeps should probe
    around these values before a long run.
    """

    memory_gb = max(float(hardware.memory_gb), 1.0)
    hidden_size = int(model_config.get("hidden_size", 128) or 128)
    layers = int(model_config.get("planet_transformer_layers", 2) or 2)
    history = int(task_config.get("feature_history_steps", 1) or 1)
    shield_horizon = int(task_config.get("trajectory_shield_horizon", 10) or 10)

    model_scale = max(hidden_size / 128.0, 1.0) * max(layers / 2.0, 1.0)
    horizon_scale = max(history / 2.0, 1.0) * max(shield_horizon / 10.0, 1.0)
    pressure = model_scale * horizon_scale

    env_budget = int(memory_gb * 2.0 / pressure)
    num_envs = _round_to_multiple(max(min(env_budget, 96), 8), 8)
    microbatch_envs = _round_to_multiple(max(min(num_envs // 2, 32), 4), 4)
    rollout_steps = 256 if memory_gb >= 24 and pressure <= 3.0 else 128
    minibatch_size = _round_to_multiple(max(min(num_envs * rollout_steps // 4, 4096), 256), 128)
    chunk_max = max(minibatch_size * 2, 1024)

    return (
        f"training.num_envs={num_envs}",
        f"training.rollout_microbatch_envs={microbatch_envs}",
        f"training.rollout_steps={rollout_steps}",
        f"training.minibatch_size={minibatch_size}",
        f"training.update_chunk_rows_min={min(chunk_max, 2048)}",
        f"training.update_chunk_rows_max={chunk_max}",
    )


def calibration_grid(base_overrides: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Generate a bounded calibration grid around estimator output."""

    parsed = dict(item.split("=", 1) for item in base_overrides)
    num_envs = int(parsed["training.num_envs"])
    rollout_steps = int(parsed["training.rollout_steps"])
    variants: list[tuple[str, ...]] = []
    for env_multiplier in (0.5, 1.0, 1.5):
        envs = _round_to_multiple(max(int(num_envs * env_multiplier), 4), 4)
        micro = _round_to_multiple(max(min(envs // 2, 32), 4), 4)
        minibatch = _round_to_multiple(max(envs * rollout_steps // 4, 128), 128)
        variant = tuple(
            _replace_override(
                _replace_override(
                    _replace_override(base_overrides, "training.num_envs", envs),
                    "training.rollout_microbatch_envs",
                    micro,
                ),
                "training.minibatch_size",
                minibatch,
            )
        )
        variants.append(variant)
    return variants


def _replace_override(
    overrides: tuple[str, ...], key: str, value: int
) -> tuple[str, ...]:
    return tuple(
        f"{key}={value}" if item.startswith(f"{key}=") else item for item in overrides
    )


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple) * multiple))
