from __future__ import annotations

from typing import NamedTuple

import flax
import flax.struct
import optax

import jax
from src.jax.shield import ShieldDiagnostics


class FactorizedActionReplay(NamedTuple):
    """Stored factorized pointer-decoder actions for PPO replay."""

    ship_bucket_mask: jax.Array
    target_index: jax.Array
    ship_bucket: jax.Array
    log_prob: jax.Array
    source_index: jax.Array
    target_slot: jax.Array
    stop_flag: jax.Array
    step_mask: jax.Array
    decoder_hidden: jax.Array | None = None
    ship_fraction: jax.Array | None = None


class PlanetFlowActionReplay(NamedTuple):
    """Stored Planet Flow pressure actions for PPO replay."""

    target_bucket: jax.Array
    target_pressure: jax.Array
    target_mask: jax.Array
    log_prob: jax.Array


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
    returns: jax.Array
    advantages: jax.Array
    action_replay: FactorizedActionReplay | PlanetFlowActionReplay
    initial_planet_ships: jax.Array | None = None


def transition_env_rows(batch: JaxTransitionBatch) -> int:
    """Return flattened environment-step count from observation time×env axes."""

    return batch.planet_features.shape[0] * batch.planet_features.shape[1]


def require_factorized_replay(batch: JaxTransitionBatch) -> FactorizedActionReplay:
    """Return factorized replay fields or raise when the batch variant mismatches."""

    if not isinstance(batch.action_replay, FactorizedActionReplay):
        raise ValueError(
            "Expected factorized action replay fields on this transition batch."
        )
    return batch.action_replay


def require_planet_flow_replay(batch: JaxTransitionBatch) -> PlanetFlowActionReplay:
    """Return Planet Flow replay fields or raise when the batch variant mismatches."""

    if not isinstance(batch.action_replay, PlanetFlowActionReplay):
        raise ValueError(
            "Expected Planet Flow action replay fields on this transition batch."
        )
    return batch.action_replay


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
