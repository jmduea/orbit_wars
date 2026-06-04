"""Shared JAX array helpers used across training and encoders."""

from __future__ import annotations

import jax.numpy as jnp

import jax


def masked_mean(
    values: jax.Array,
    mask: jax.Array,
    *,
    axis: int | None = None,
) -> jax.Array:
    """Average ``values`` over entries where ``mask`` is non-zero.

    When ``axis`` is ``None``, returns a scalar mean over all masked elements.
    When ``axis`` is set, reduces along that axis (for example pooling planets
    or edges per batch row). Masked-out positions use zero weight; NaN values
    at mask-zero positions are ignored via ``jnp.where`` on the scalar path.
    """

    if axis is None:
        safe_values = jnp.where(mask > 0, values, 0.0)
        weight = mask.astype(values.dtype)
        return safe_values.sum() / jnp.maximum(weight.sum(), 1.0)

    weights = mask.astype(values.dtype)[..., None]
    total = (values * weights).sum(axis=axis)
    count = jnp.maximum(weights.sum(axis=axis), 1.0)
    return total / count
