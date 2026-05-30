"""Running observation normalization for planet-edge TurnBatch features."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

import jax
from src.config.schema import ModelConfig
from src.jax.features import TurnBatch
from src.jax.rollout.types import JaxTransitionBatch


class ObservationNormState(NamedTuple):
    """Welford running mean/variance for planet, edge, and global tensors."""

    planet_mean: jax.Array
    planet_var: jax.Array
    edge_mean: jax.Array
    edge_var: jax.Array
    global_mean: jax.Array
    global_var: jax.Array
    count: jax.Array


def init_observation_norm_state(batch: TurnBatch) -> ObservationNormState:
    """Initialize normalization state from feature shapes."""

    zero = jnp.zeros((), dtype=jnp.float32)
    return ObservationNormState(
        planet_mean=jnp.zeros(batch.planet_features.shape[-1], dtype=jnp.float32),
        planet_var=jnp.zeros(batch.planet_features.shape[-1], dtype=jnp.float32),
        edge_mean=jnp.zeros(batch.edge_features.shape[-1], dtype=jnp.float32),
        edge_var=jnp.zeros(batch.edge_features.shape[-1], dtype=jnp.float32),
        global_mean=jnp.zeros(batch.global_features.shape[-1], dtype=jnp.float32),
        global_var=jnp.zeros(batch.global_features.shape[-1], dtype=jnp.float32),
        count=zero,
    )


def _masked_feature_stats(
    features: jax.Array,
    mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return masked mean, variance, and valid element count per feature dim."""

    weights = mask.astype(jnp.float32)[..., None]
    count = weights.sum()
    denom = jnp.maximum(count, 1.0)
    mean = (features * weights).sum(axis=tuple(range(features.ndim - 1))) / denom
    centered = features - mean
    var = ((centered * centered) * weights).sum(axis=tuple(range(features.ndim - 1))) / denom
    return mean, var, count


def _merge_running_stats(
    old_mean: jax.Array,
    old_var: jax.Array,
    old_count: jax.Array,
    batch_mean: jax.Array,
    batch_var: jax.Array,
    batch_count: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Combine running and batch statistics (parallel Welford merge)."""

    total = old_count + batch_count
    safe_total = jnp.maximum(total, 1.0)
    delta = batch_mean - old_mean
    new_mean = old_mean + delta * (batch_count / safe_total)
    m_a = old_var * old_count
    m_b = batch_var * batch_count
    new_var = (m_a + m_b + (delta * delta) * old_count * batch_count / safe_total) / safe_total
    return new_mean, new_var, total


def update_observation_norm_state(
    state: ObservationNormState,
    batch: TurnBatch,
) -> ObservationNormState:
    """Update running statistics from a raw ``TurnBatch``."""

    planet_mean, planet_var, planet_count = _masked_feature_stats(
        batch.planet_features, batch.planet_mask
    )
    edge_mean, edge_var, edge_count = _masked_feature_stats(
        batch.edge_features, batch.edge_mask
    )
    global_count = jnp.asarray(batch.global_features.shape[0], dtype=jnp.float32)
    global_mean = batch.global_features.mean(axis=0)
    global_var = batch.global_features.var(axis=0)

    p_mean, p_var, count = _merge_running_stats(
        state.planet_mean,
        state.planet_var,
        state.count,
        planet_mean,
        planet_var,
        planet_count,
    )
    e_mean, e_var, count = _merge_running_stats(
        state.edge_mean,
        state.edge_var,
        count,
        edge_mean,
        edge_var,
        edge_count,
    )
    g_mean, g_var, count = _merge_running_stats(
        state.global_mean,
        state.global_var,
        count,
        global_mean,
        global_var,
        global_count,
    )
    return ObservationNormState(
        planet_mean=p_mean,
        planet_var=p_var,
        edge_mean=e_mean,
        edge_var=e_var,
        global_mean=g_mean,
        global_var=g_var,
        count=count,
    )


def _normalize_features(
    features: jax.Array,
    mean: jax.Array,
    var: jax.Array,
    *,
    clip: float,
    eps: float,
) -> jax.Array:
    std = jnp.sqrt(jnp.maximum(var, 0.0)) + eps
    normalized = (features - mean) / std
    return jnp.clip(normalized, -clip, clip)


def normalize_turn_batch(
    batch: TurnBatch,
    state: ObservationNormState,
    model_cfg: ModelConfig,
) -> TurnBatch:
    """Apply running-stat normalization when enabled in model config."""

    if not model_cfg.normalize_observations:
        return batch
    clip = float(model_cfg.obs_norm_clip)
    eps = 1e-8
    return TurnBatch(
        planet_features=_normalize_features(
            batch.planet_features, state.planet_mean, state.planet_var, clip=clip, eps=eps
        ),
        planet_mask=batch.planet_mask,
        edge_features=_normalize_features(
            batch.edge_features, state.edge_mean, state.edge_var, clip=clip, eps=eps
        ),
        edge_mask=batch.edge_mask,
        edge_src_ids=batch.edge_src_ids,
        edge_tgt_ids=batch.edge_tgt_ids,
        global_features=_normalize_features(
            batch.global_features, state.global_mean, state.global_var, clip=clip, eps=eps
        ),
        theta_ref=batch.theta_ref,
    )


def transition_batch_to_turn_batch(batch: JaxTransitionBatch) -> TurnBatch:
    """Project rollout transition features into a ``TurnBatch``."""

    return TurnBatch(
        planet_features=batch.planet_features,
        planet_mask=batch.planet_mask,
        edge_features=batch.edge_features,
        edge_mask=batch.edge_mask,
        edge_src_ids=batch.edge_src_ids,
        edge_tgt_ids=batch.edge_tgt_ids,
        global_features=batch.global_features,
        theta_ref=batch.theta_ref,
    )


def update_norm_state_from_transitions(
    state: ObservationNormState,
    transitions: JaxTransitionBatch,
) -> ObservationNormState:
    """Update running stats from a rollout transition batch."""

    flat = jax.tree.map(
        lambda x: x.reshape((-1,) + x.shape[2:]) if x.ndim > 2 else x,
        transition_batch_to_turn_batch(transitions),
    )
    return update_observation_norm_state(state, flat)


def normalize_transition_batch(
    transitions: JaxTransitionBatch,
    state: ObservationNormState,
    model_cfg: ModelConfig,
) -> JaxTransitionBatch:
    """Normalize feature tensors inside a transition batch."""

    if not model_cfg.normalize_observations:
        return transitions
    turn = transition_batch_to_turn_batch(transitions)
    normalized = normalize_turn_batch(turn, state, model_cfg)
    return transitions._replace(
        planet_features=normalized.planet_features,
        edge_features=normalized.edge_features,
        global_features=normalized.global_features,
    )
