"""Legacy env_parity_mode (comet-free hot path)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import jax
from src.config.schema import RewardConfig, TaskConfig
from src.game.shield_config import env_comet_physics_enabled, env_parity_mode
from src.jax.env import batched_reset, empty_action, step


def _cfg(**kwargs) -> TaskConfig:
    base = dict(candidate_count=4, ship_bucket_count=4, max_fleets=8, player_count=2)
    base.update(kwargs)
    return TaskConfig(**base)


def test_env_parity_mode_legacy_normalizes() -> None:
    cfg = _cfg(env_parity_mode="legacy")
    assert env_parity_mode(cfg) == "legacy"
    assert not env_comet_physics_enabled(cfg)


@pytest.mark.jax
def test_legacy_reset_and_noop_steps_smoke() -> None:
    cfg = _cfg(env_parity_mode="legacy")
    reward_cfg = RewardConfig()
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    state, _batch = batched_reset(keys, cfg)
    noop = jax.vmap(lambda _: empty_action(cfg))(jnp.arange(2))
    state, result = jax.vmap(
        lambda s, learner, opponent: step(s, learner, opponent, cfg, reward_cfg)
    )(state, noop, noop)
    assert jnp.all(jnp.isfinite(result.reward))


@pytest.mark.jax
def test_env_parity_ab_payload_shape() -> None:
    from src.benchmark.env_parity import run_env_parity_ab_benchmark

    payload = run_env_parity_ab_benchmark(
        batch_size=2,
        steps_per_episode=4,
        warmup=1,
        repeats=2,
        modes=("legacy", "train"),
    )
    assert payload["benchmark"] == "env_parity_ab"
    assert len(payload["arms"]) == 2
    for arm in payload["arms"]:
        assert float(arm["env_steps_per_sec"]) > 0.0
