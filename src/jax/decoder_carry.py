"""Cross-turn decoder GRU carry helpers for flag-gated recurrence."""

from __future__ import annotations

import jax.numpy as jnp

import jax
from src.config import TrainConfig


def decoder_carry_enabled(cfg: TrainConfig) -> bool:
    """Return True when the policy should persist decoder GRU state across turns."""

    return bool(cfg.model.decoder_carry)


def empty_decoder_hidden(batch_size: int, hidden_size: int) -> jax.Array:
    """Return a zero-initialized decoder carry for ``batch_size`` rows."""

    return jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)


def resolve_decoder_initial_state(
    init_state: jax.Array,
    carry_hidden: jax.Array | None,
    *,
    enabled: bool,
) -> jax.Array:
    """Select fresh init state or the carried hidden state."""

    if not enabled or carry_hidden is None:
        return init_state
    return carry_hidden


def reset_decoder_hidden_on_done(
    hidden: jax.Array,
    done: jax.Array,
    fresh_hidden: jax.Array,
) -> jax.Array:
    """Zero decoder carry rows whose environments terminated."""

    cond = done.reshape(done.shape + (1,) * (hidden.ndim - 1))
    return jnp.where(cond, fresh_hidden, hidden)
