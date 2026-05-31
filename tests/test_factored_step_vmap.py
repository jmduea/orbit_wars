from __future__ import annotations

import jax.numpy as jnp
import pytest

import jax
from src.config import TaskConfig
from src.features.registry import edge_k
from src.game.constants import MAX_PLANETS
from src.jax.action_sampling import _sample_factored_step_from_logits


def _task_cfg() -> TaskConfig:
    return TaskConfig(candidate_count=4, ship_bucket_count=4, max_fleets=8)


@pytest.mark.jax
def test_sample_factored_step_from_logits_is_vmap_compatible() -> None:
    """Regression: keyword-only helper broke jax.vmap in rollout collect."""
    cfg = _task_cfg()
    k = edge_k(cfg)
    env_count = 3
    keys = jax.random.split(jax.random.PRNGKey(0), env_count)
    source_logits = jnp.zeros((env_count, MAX_PLANETS))
    target_logits = jnp.zeros((env_count, k))
    stop_logits = jnp.zeros((env_count,))
    ship_logits = jnp.zeros((env_count, k, cfg.ship_bucket_count))
    source_mask = jnp.ones((env_count, MAX_PLANETS), dtype=bool)
    ship_bucket_mask = jnp.ones(
        (env_count, MAX_PLANETS, k, cfg.ship_bucket_count), dtype=bool
    )

    vmapped = jax.vmap(
        _sample_factored_step_from_logits,
        in_axes=(0, 0, 0, 0, 0, 0, 0, None, None),
    )
    source, target_slot, bucket, stop, log_prob, entropy, _ship_fraction = vmapped(
        keys,
        source_logits,
        target_logits,
        stop_logits,
        ship_logits,
        source_mask,
        ship_bucket_mask,
        False,
        False,
    )

    assert source.shape == (env_count,)
    assert target_slot.shape == (env_count,)
    assert bucket.shape == (env_count,)
    assert stop.shape == (env_count,)
    assert log_prob.shape == (env_count,)
    assert entropy.shape == (env_count,)
