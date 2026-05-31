from __future__ import annotations

from typing import NamedTuple

import flax
import flax.struct
import optax

import jax
from src.jax.shield import ShieldDiagnostics


class JaxTransitionBatch(NamedTuple):
    """Rollout data for planet-edge encoding."""

    planet_features: jax.Array
    planet_mask: jax.Array
    edge_features: jax.Array
    edge_mask: jax.Array
    edge_src_ids: jax.Array
    edge_tgt_ids: jax.Array
    global_features: jax.Array
    theta_ref: jax.Array
    player_count: jax.Array
    ship_bucket_mask: jax.Array
    target_index: jax.Array
    ship_bucket: jax.Array
    log_prob: jax.Array
    returns: jax.Array
    advantages: jax.Array
    source_index: jax.Array
    target_slot: jax.Array
    stop_flag: jax.Array
    step_mask: jax.Array
    decoder_hidden: jax.Array | None = None
    ship_fraction: jax.Array | None = None
    initial_planet_ships: jax.Array | None = None


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
    source_index: jax.Array
    target_slot: jax.Array
    stop_flag: jax.Array
    step_mask: jax.Array
    decoder_hidden_out: jax.Array | None = None
    ship_fraction: jax.Array | None = None
