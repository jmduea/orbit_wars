from __future__ import annotations

import jax.numpy as jnp

import jax
import pytest
from src.config import TrainConfig
from src.game.constants import MAX_PLANETS
from src.jax.action_codec import (
    PlanetFlowPolicyOutput,
    sample_planet_flow_pressure_action,
)
from src.jax.env import batched_reset
from src.jax.policy import build_jax_policy, make_synthetic_turn_batch
from src.jax.ppo_update import ppo_update_jax
from src.jax.rollout.collect import collect_rollout_jax
from src.jax.train import init_train_state
from src.jax.rollout.types import PlanetFlowActionReplay
from src.jax.train.metrics import sum_metric_dicts


def _planet_flow_cfg() -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.model.pointer_decoder = "planet_flow_target_heatmap"
    cfg.model.hidden_size = 32
    cfg.model.attention_heads = 4
    cfg.model.planet_transformer_layers = 1
    cfg.model.value_head = "shared"
    cfg.task.candidate_count = 4
    return cfg


def test_build_jax_policy_dispatches_planet_flow_target_heatmap() -> None:
    cfg = _planet_flow_cfg()

    policy = build_jax_policy(cfg)

    assert policy.__class__.__name__ == "ComposablePlanetFlowPolicy"


def test_planet_flow_policy_outputs_target_demand_logits() -> None:
    cfg = _planet_flow_cfg()
    policy = build_jax_policy(cfg)
    batch = make_synthetic_turn_batch(2, cfg.task, key=jax.random.PRNGKey(0))
    params = policy.init(jax.random.PRNGKey(1), batch)

    output = policy.apply(params, batch, deterministic=True)

    assert isinstance(output, PlanetFlowPolicyOutput)
    assert output.target_demand_logits.shape == (
        2,
        MAX_PLANETS,
        len(cfg.model.planet_flow.pressure_bucket_values),
    )
    assert output.value.shape == (2,)


def test_planet_flow_inactive_targets_sample_as_hold() -> None:
    cfg = _planet_flow_cfg()
    policy = build_jax_policy(cfg)
    batch = make_synthetic_turn_batch(1, cfg.task, key=jax.random.PRNGKey(2))
    target_mask = jnp.ones((1, MAX_PLANETS), dtype=bool)
    target_mask = target_mask.at[0, 3:].set(False)
    batch = batch._replace(planet_mask=target_mask)
    params = policy.init(jax.random.PRNGKey(3), batch)
    output = policy.apply(params, batch, deterministic=True)

    sample = sample_planet_flow_pressure_action(
        jax.random.PRNGKey(4),
        output,
        jnp.asarray(cfg.model.planet_flow.pressure_bucket_values, dtype=jnp.float32),
        target_mask,
        deterministic=True,
    )

    assert jnp.all(sample.target_bucket[:, 3:] == 0)
    assert jnp.all(sample.target_pressure[:, 3:] == 0.0)


def test_factorized_policy_dispatch_remains_default() -> None:
    cfg = TrainConfig()
    cfg.model.pointer_decoder = "factorized_topk"

    policy = build_jax_policy(cfg)

    assert policy.__class__.__name__ == "ComposableFactorizedPlanetPolicy"


@pytest.mark.jax
def test_collect_rollout_jax_updates_planet_flow_pressure_action() -> None:
    cfg = _planet_flow_cfg()
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.model.max_moves_k = 3
    cfg.task.max_fleets = 16
    cfg.training.num_envs = 2
    cfg.training.rollout_steps = 1
    cfg.opponents.mode.opponent = "random"
    cfg.telemetry.metric_groups.action_decision = True
    reset_keys = jax.random.split(jax.random.PRNGKey(10), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.PRNGKey(11), policy, cfg)

    _key, _env_state, _turn_batch, transitions, metrics = collect_rollout_jax(
        jax.random.PRNGKey(12), env_state, turn_batch, train_state, policy, cfg
    )
    _train_state, ppo_metrics = ppo_update_jax(train_state, policy, transitions, cfg)

    assert isinstance(transitions.action_replay, PlanetFlowActionReplay)
    replay = transitions.action_replay
    assert replay.target_bucket is not None
    assert replay.target_pressure is not None
    assert replay.target_mask is not None
    assert replay.target_bucket.shape == (
        cfg.training.rollout_steps,
        cfg.training.num_envs,
        MAX_PLANETS,
    )
    assert float(metrics["samples"]) == cfg.training.rollout_steps * cfg.training.num_envs
    assert "planet_flow_control_emitted_launch_count" in metrics
    finalized_metrics = sum_metric_dicts([metrics])
    assert "planet_flow_emitted_launch_count_delta_vs_control" in finalized_metrics
    assert "policy_loss" in ppo_metrics
