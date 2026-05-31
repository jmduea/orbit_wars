from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jax
from src.config import compose_hydra_train_config
from src.game.constants import MAX_PLANETS
from src.jax.action_sampling import _sample_factored_step_from_logits
from src.jax.env import batched_reset
from src.jax.policy import build_jax_policy
from src.jax.rollout.collect import collect_rollout_jax
from src.jax.train import init_train_state


def test_continuous_ship_logit_mask_does_not_broadcast_to_discrete_path() -> None:
    """Regression: noop-only bucket mask must not broadcast (1,) logits to discrete path."""
    k = 3
    bucket_count = 4
    ship_logits = jnp.full((k, 1), 2.0, dtype=jnp.float32)
    source_logits = jnp.zeros((MAX_PLANETS,), dtype=jnp.float32)
    source_logits = source_logits.at[0].set(1.0)
    target_logits = jnp.zeros((k,), dtype=jnp.float32)
    stop_logits = jnp.array(-10.0, dtype=jnp.float32)
    source_mask = jnp.zeros((MAX_PLANETS,), dtype=bool)
    source_mask = source_mask.at[0].set(True)
    ship_bucket_mask = jnp.zeros((MAX_PLANETS, k, bucket_count), dtype=bool)
    ship_bucket_mask = ship_bucket_mask.at[..., 0].set(True)

    _source, _target_slot, bucket, stop, _log_prob, _entropy, ship_fraction = (
        _sample_factored_step_from_logits(
            jax.random.PRNGKey(0),
            source_logits,
            target_logits,
            stop_logits,
            ship_logits,
            source_mask,
            ship_bucket_mask,
            deterministic=True,
        )
    )

    assert int(stop) == 0
    assert float(np.asarray(ship_fraction)) > 0.0
    assert int(bucket) == 1


@pytest.mark.jax
def test_factorized_rollout_emits_launches_with_shield_disabled() -> None:
    cfg = compose_hydra_train_config(
        [
            "model=transformer_factorized",
            "training.rollout_steps=4",
            "training.num_envs=2",
            "task.trajectory_shield_mode=off",
        ]
    )
    reset_keys = jax.random.split(jax.random.PRNGKey(0), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.PRNGKey(1), policy, cfg)

    _key, _env_state, _turn_batch, transitions, rollout_metrics = collect_rollout_jax(
        jax.random.PRNGKey(2),
        env_state,
        turn_batch,
        train_state,
        policy,
        cfg,
    )

    ship_fraction = transitions.ship_fraction
    assert ship_fraction is not None
    launch_slots = np.asarray(ship_fraction > 0.0)
    assert launch_slots.any()
    assert float(rollout_metrics["mean_active_launches_per_turn"]) > 0.0
