from __future__ import annotations

from typing import NamedTuple

import flax
import flax.struct
import optax

import jax


class JaxTransitionBatch(NamedTuple):
    """Rollout data consumed by the JAX PPO update.

    Arrays keep rollout, environment, and source-planet dimensions until the
    update step flattens them. ``decision_mask`` identifies valid learner-owned
    source rows that should contribute to PPO losses.
    """

    self_features: jax.Array
    candidate_features: jax.Array
    global_features: jax.Array
    candidate_mask: jax.Array
    player_count: jax.Array
    ship_bucket_mask: jax.Array
    decision_mask: jax.Array
    target_index: jax.Array
    ship_bucket: jax.Array
    log_prob: jax.Array
    returns: jax.Array
    advantages: jax.Array


@flax.struct.dataclass
class JaxTrainState:
    """Minimal immutable train state for Flax parameters and Optax state."""

    params: dict
    opt_state: optax.OptState
    optimizer: optax.GradientTransformation = flax.struct.field(pytree_node=False)


class ShieldedSequenceSample(NamedTuple):
    target_index: jax.Array
    ship_bucket: jax.Array
    log_prob: jax.Array
    entropy: jax.Array
    value: jax.Array
    ship_bucket_mask: jax.Array
    diagnostics: ShieldDiagnostics
