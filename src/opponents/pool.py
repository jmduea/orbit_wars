from __future__ import annotations

import jax.numpy as jnp

import jax
from src.opponents.constants import (
    OPPONENT_HISTORICAL,
    OPPONENT_LATEST,
    OPPONENT_NEAREST_SNIPER,
    OPPONENT_NOOP,
    OPPONENT_OPPORTUNISTIC,
    OPPONENT_RANDOM,
    OPPONENT_TURTLE,
)

OPPONENT_FAMILY_IDS = jnp.asarray(
    [
        OPPONENT_LATEST,
        OPPONENT_HISTORICAL,
        OPPONENT_NEAREST_SNIPER,
        OPPONENT_TURTLE,
        OPPONENT_OPPORTUNISTIC,
        OPPONENT_RANDOM,
        OPPONENT_NOOP,
    ],
    dtype=jnp.int32,
)


def sample_opponent_type_ids_jax(
    key: jax.Array,
    env_count: int,
    player_count: int,
    *,
    ids: jax.Array,
    probs: jax.Array,
) -> jax.Array:
    """Sample opponent IDs per [env, player] slot from a categorical mixture."""

    logits = jnp.log(jnp.maximum(probs, 1e-12))
    sampled = jax.random.categorical(
        key,
        logits[None, None, :],
        axis=-1,
        shape=(env_count, player_count),
    )
    return ids[sampled]
