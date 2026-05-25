"""C51-style distributional value support for the JAX critic."""

from __future__ import annotations

import jax.numpy as jnp

import jax


def value_support(value_bins: int, value_max: float) -> jax.Array:
    """Return evenly spaced return atoms on ``[-value_max, value_max]``."""

    if value_bins < 2:
        raise ValueError("value_bins must be at least 2 for distributional value.")
    if value_max <= 0.0:
        raise ValueError("value_max must be positive for distributional value.")
    return jnp.linspace(-value_max, value_max, value_bins, dtype=jnp.float32)


def expected_value_from_logits(logits: jax.Array, support: jax.Array) -> jax.Array:
    """Compute the expected return from categorical value logits."""

    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.sum(probs * support, axis=-1)


def project_returns_to_two_hot(returns: jax.Array, support: jax.Array) -> jax.Array:
    """Project scalar returns onto the nearest two support atoms (C51)."""

    returns = returns.astype(jnp.float32)
    vmin = support[0]
    delta = support[1] - support[0]
    clipped = jnp.clip(returns, vmin, support[-1])
    position = (clipped - vmin) / delta
    lower = jnp.floor(position).astype(jnp.int32)
    lower = jnp.clip(lower, 0, support.shape[0] - 2)
    upper = lower + 1
    upper_weight = position - lower.astype(jnp.float32)
    lower_weight = 1.0 - upper_weight

    flat_returns = clipped.reshape(-1)
    flat_lower = lower.reshape(-1)
    flat_upper = upper.reshape(-1)
    flat_lower_weight = lower_weight.reshape(-1)
    flat_upper_weight = upper_weight.reshape(-1)
    row_indices = jnp.arange(flat_returns.shape[0], dtype=jnp.int32)
    targets = jnp.zeros((flat_returns.shape[0], support.shape[0]), dtype=jnp.float32)
    targets = targets.at[row_indices, flat_lower].add(flat_lower_weight)
    targets = targets.at[row_indices, flat_upper].add(flat_upper_weight)
    return targets.reshape(*returns.shape, support.shape[0])


def categorical_value_cross_entropy(
    logits: jax.Array,
    returns: jax.Array,
    support: jax.Array,
) -> jax.Array:
    """Cross-entropy between projected return targets and predicted value logits."""

    targets = project_returns_to_two_hot(returns, support)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(targets * log_probs, axis=-1)
