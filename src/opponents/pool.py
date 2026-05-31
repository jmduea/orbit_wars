from __future__ import annotations

import jax.numpy as jnp

import jax

OPPONENT_LATEST = 0
OPPONENT_HISTORICAL = 1
OPPONENT_NEAREST_SNIPER = 2
OPPONENT_TURTLE = 3
OPPONENT_OPPORTUNISTIC = 4
OPPONENT_RANDOM = 5
OPPONENT_NOOP = 6
OPPONENT_SCRIPTED_SNIPER = OPPONENT_NEAREST_SNIPER
OPPONENT_FAMILY_NAMES = (
    "latest",
    "historical",
    "nearest_sniper",
    "turtle",
    "opportunistic",
    "random",
    "noop",
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
OPPONENT_FAMILY_COUNT = len(OPPONENT_FAMILY_NAMES)


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
