"""Ship action mode helpers for discrete buckets vs continuous fractions."""

from __future__ import annotations

import jax.numpy as jnp

import jax
from src.config import TrainConfig
from src.jax.shield import ship_count_for_fraction_jax


def is_continuous_ship_mode(cfg: TrainConfig) -> bool:
    """Return True when the policy predicts a continuous ship fraction."""

    return cfg.task.ship_action_mode.strip().lower() == "continuous_fraction"


def ship_action_logit_width(cfg: TrainConfig) -> int:
    """Return the ship head width for the configured action mode."""

    if is_continuous_ship_mode(cfg):
        return 1
    return int(cfg.task.ship_bucket_count)


def fraction_from_logit(logit: jax.Array) -> jax.Array:
    """Map a scalar logit to a launch fraction in ``(0, 1]``."""

    return jnp.clip(jax.nn.sigmoid(logit), 1e-6, 1.0)


def logit_from_fraction(fraction: jax.Array) -> jax.Array:
    """Invert ``fraction_from_logit`` for replaying stored continuous ship draws."""

    clipped = jnp.clip(fraction.astype(jnp.float32), 1e-6, 1.0 - 1e-6)
    return jnp.log(clipped) - jnp.log1p(-clipped)


def continuous_fraction_log_prob(logit: jax.Array) -> jax.Array:
    """Log density of a standard logistic draw at latent ``logit``."""

    return -jax.nn.softplus(-logit) - jax.nn.softplus(logit)


def continuous_fraction_log_prob_at_action(
    policy_loc: jax.Array,
    action_fraction: jax.Array,
) -> jax.Array:
    """Log density of a stored launch fraction under ``Logistic(policy_loc, 1)``.

    Rollout draws ``z = policy_loc + epsilon`` with standard logistic ``epsilon``,
    then maps ``fraction = sigmoid(z)``. Replay evaluates ``log p(fraction | policy_loc)``
    via the change-of-variables through ``z = logit(fraction)``.
    """

    frac = jnp.clip(action_fraction.astype(jnp.float32), 1e-6, 1.0 - 1e-6)
    z = logit_from_fraction(frac)
    centered = z - policy_loc.astype(jnp.float32)
    log_prob_z = -jax.nn.softplus(-centered) - jax.nn.softplus(centered)
    log_abs_det_jacobian = -jnp.log(frac) - jnp.log1p(-frac)
    return log_prob_z + log_abs_det_jacobian


def ship_count_for_action(
    available_ships: jax.Array,
    ship_bucket: jax.Array,
    ship_fraction: jax.Array | None,
    cfg: TrainConfig,
) -> jax.Array:
    """Resolve a launch ship count from bucket ids or a continuous fraction."""

    if is_continuous_ship_mode(cfg):
        if ship_fraction is None:
            raise ValueError("Continuous ship mode requires ship_fraction.")
        return ship_count_for_fraction_jax(available_ships, ship_fraction)
    from src.jax.shield import ship_count_for_bucket_jax

    return ship_count_for_bucket_jax(
        available_ships, ship_bucket, cfg.task.ship_bucket_count
    )
