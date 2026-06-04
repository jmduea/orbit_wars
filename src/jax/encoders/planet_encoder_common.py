"""Shared planet-edge encoder contracts and helpers."""

from __future__ import annotations

from typing import NamedTuple

import flax.linen as nn
import jax.numpy as jnp

import jax
from src.game.constants import MAX_PLANETS
from src.jax.array_ops import masked_mean
from src.jax.features import TurnBatch


class PlanetEdgeEncoderOutput(NamedTuple):
    """Policy encoder contract for ``TurnBatch`` inputs."""

    attended_edges: jax.Array
    edge_action_mask: jax.Array
    context_query: jax.Array
    value_input: jax.Array
    planet_states: jax.Array
    edge_hidden: jax.Array
    edge_mask: jax.Array
    planet_mask: jax.Array


def mlp(x: jax.Array, hidden_size: int, output_size: int, name: str) -> jax.Array:
    """Apply a named two-layer ReLU MLP block."""

    x = nn.Dense(hidden_size, name=f"{name}_0")(x)
    x = nn.relu(x)
    x = nn.Dense(output_size, name=f"{name}_1")(x)
    return nn.relu(x)


def planet_orbit_coords(planet_features: jax.Array) -> jax.Array:
    """Map planet feature slices to 2D orbit coordinates."""

    from src.features.catalog.planet import PLANET_FEATURE_CATALOG

    radius_slice = PLANET_FEATURE_CATALOG.base_slice("orbit_radius")
    angle_slice = PLANET_FEATURE_CATALOG.base_slice("orbit_angle")
    orbit_radius = planet_features[..., radius_slice].squeeze(-1)
    orbit_angle = planet_features[..., angle_slice].squeeze(-1) * jnp.pi
    return jnp.stack(
        [
            orbit_radius * jnp.cos(orbit_angle),
            orbit_radius * jnp.sin(orbit_angle),
        ],
        axis=-1,
    )


def planet_pairwise_spatial_bias(coords: jax.Array) -> jax.Array:
    """Build an additive attention bias from pairwise orbit distances."""

    diffs = coords[:, :, None, :] - coords[:, None, :, :]
    dist_sq = jnp.sum(diffs**2, axis=-1)
    return -dist_sq[:, None, :, :]


def planet_self_attention_mask(planet_mask: jax.Array) -> jax.Array:
    """Return a boolean mask for planet self-attention over padded rows."""

    mask = planet_mask[:, :, None] & planet_mask[:, None, :]
    has_valid_key = mask.any(axis=-1, keepdims=True)
    return jnp.where(has_valid_key, mask, jnp.ones_like(mask))


def planet_attention_mask_with_bias(
    planet_mask: jax.Array,
    coords: jax.Array,
    *,
    spatial_attention_bias: bool,
) -> jax.Array:
    """Build an additive attention mask for planet self-attention."""

    bool_mask = planet_self_attention_mask(planet_mask)
    float_mask = jnp.where(
        bool_mask,
        jnp.zeros((), dtype=jnp.float32),
        jnp.finfo(jnp.float32).min,
    )[:, None, :, :]
    if spatial_attention_bias:
        float_mask = float_mask + planet_pairwise_spatial_bias(coords)
    return float_mask


def gather_target_planet_states(
    current_planet_states: jax.Array,
    batch: TurnBatch,
) -> jax.Array:
    """Gather contextualized target planet embeddings for each edge slot."""

    match = (
        batch.edge_src_ids[:, :, None, None] == batch.edge_tgt_ids[:, None, :, :]
    ).astype(current_planet_states.dtype)
    match_sum = jnp.maximum(match.sum(axis=1, keepdims=True), 1e-6)
    weights = match / match_sum
    return jnp.einsum("btsk,bth->bskh", weights, current_planet_states)


def fuse_source_target_edges(
    current_planet_states: jax.Array,
    edge_hidden: jax.Array,
    batch: TurnBatch,
    *,
    hidden_size: int,
) -> jax.Array:
    """Fuse source planet, target planet, and edge MLP states."""

    src_planet = jnp.broadcast_to(
        current_planet_states[:, :, None, :],
        (*edge_hidden.shape[:3], hidden_size),
    )
    tgt_planet = gather_target_planet_states(current_planet_states, batch)
    edge_input = jnp.concatenate([src_planet, tgt_planet, edge_hidden], axis=-1)
    edge_hidden = mlp(edge_input, hidden_size, hidden_size, "edge_fuse")
    return nn.LayerNorm(name="edge_fuse_norm")(edge_hidden)


def finalize_planet_edge_encoder_output(
    *,
    current_planet_states: jax.Array,
    global_hidden: jax.Array,
    edge_hidden: jax.Array,
    edge_mask: jax.Array,
    planet_mask: jax.Array,
    hidden_size: int,
    edge_k: int,
) -> PlanetEdgeEncoderOutput:
    """Pool planet/edge states into the shared encoder output contract."""

    batch_size = current_planet_states.shape[0]
    if edge_k > 0:
        attended_edges = edge_hidden.reshape(batch_size, MAX_PLANETS * edge_k, -1)
        edge_action_mask = edge_mask.reshape(batch_size, MAX_PLANETS * edge_k)
    else:
        attended_edges = jnp.zeros((batch_size, 0, hidden_size), dtype=jnp.float32)
        edge_action_mask = jnp.zeros((batch_size, 0), dtype=bool)

    pooled_planets = masked_mean(current_planet_states, planet_mask, axis=1)
    pooled_edges = masked_mean(attended_edges, edge_action_mask, axis=1)
    context_query = mlp(
        jnp.concatenate([global_hidden, pooled_planets], axis=-1),
        hidden_size,
        hidden_size,
        "context_query",
    )
    value_input = jnp.concatenate(
        [context_query, global_hidden, pooled_planets, pooled_edges], axis=-1
    )
    return PlanetEdgeEncoderOutput(
        attended_edges=attended_edges,
        edge_action_mask=edge_action_mask,
        context_query=context_query,
        value_input=value_input,
        planet_states=current_planet_states,
        edge_hidden=edge_hidden,
        edge_mask=edge_mask,
        planet_mask=planet_mask,
    )
