"""JAX terminal reward helpers used by the training env."""

from __future__ import annotations

import jax.numpy as jnp

import jax
from src.config import RewardConfig


def apply_early_terminal_reward_shaping_jax(
    reward: jax.Array, step_index: jax.Array, cfg: RewardConfig
) -> jax.Array:
    if not cfg.early_terminal_reward_shaping_enabled:
        return reward
    horizon = jnp.asarray(
        max(int(cfg.early_terminal_reward_shaping_horizon), 1),
        dtype=jnp.float32,
    )
    step_number = jnp.maximum(step_index.astype(jnp.float32) + 1.0, 1.0)
    bonus = jnp.maximum(horizon - step_number, 0.0) / horizon
    return reward * (1.0 + bonus)
