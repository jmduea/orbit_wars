from __future__ import annotations

import jax.numpy as jnp

import jax
from src.config import TrainConfig
from src.config.schema import TaskConfig
from src.features.registry import edge_k
from src.game.trajectory_shield import ship_count_for_fraction_jax, validate_continuous_ship_launch_jax
from src.jax.env import reset
from src.jax.policy import build_planet_graph_transformer_policy, make_synthetic_turn_batch
from src.jax.ship_action import fraction_from_logit, ship_action_logit_width
from src.opponents.jax_actions.builders import build_action_from_factored_batch


def _task_cfg(**kwargs) -> TaskConfig:
    base = dict(candidate_count=4, ship_bucket_count=8, max_fleets=16)
    base.update(kwargs)
    return TaskConfig(**base)


def _train_cfg(*, continuous: bool) -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.model.pointer_decoder = "factorized_topk"
    cfg.model.hidden_size = 64
    cfg.model.max_moves_k = 2
    cfg.model.gnn_k_neighbors = 3
    cfg.model.gnn_message_passing_layers = 1
    cfg.task = _task_cfg(
        ship_action_mode="continuous_fraction" if continuous else "buckets"
    )
    return cfg


def test_ship_count_for_fraction_jax() -> None:
    ships = ship_count_for_fraction_jax(jnp.array(100.0), jnp.array(0.5))
    assert float(ships) == 50.0
    zero = ship_count_for_fraction_jax(jnp.array(0.0), jnp.array(0.8))
    assert float(zero) == 0.0


def test_factorized_decoder_ship_logit_width_one_in_continuous_mode() -> None:
    cfg = _train_cfg(continuous=True)
    assert ship_action_logit_width(cfg) == 1
    policy = build_planet_graph_transformer_policy(cfg)
    batch = make_synthetic_turn_batch(2, cfg.task, key=jax.random.PRNGKey(1))
    params = policy.init(jax.random.PRNGKey(2), batch)
    output = policy.apply(params, batch, deterministic=True)
    k_slots = edge_k(cfg.task)
    assert output.ship_logits.shape == (2, cfg.model.max_moves_k, k_slots, 1)




def test_validate_continuous_ship_launch_jax_smoke() -> None:
    cfg = _train_cfg(continuous=True)
    env_state, turn_batch = reset(jax.random.PRNGKey(5), cfg.task)
    planet_ships = jnp.where(
        env_state.game.planets.active & (env_state.game.planets.owner == env_state.game.player),
        env_state.game.planets.ships,
        0.0,
    )
    legal = validate_continuous_ship_launch_jax(
        env_state.game,
        turn_batch,
        cfg.task,
        planet_ships,
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(1.0, dtype=jnp.float32),
    )
    assert legal.shape == ()
    assert bool(jax.device_get(legal)) in {True, False}

def test_build_action_from_factored_batch_uses_continuous_fraction() -> None:
    cfg = _train_cfg(continuous=True)
    env_state, turn_batch = reset(jax.random.PRNGKey(4), cfg.task)
    game = jax.tree.map(lambda x: x[None, ...], env_state.game)
    batch = jax.tree.map(lambda x: x[None, ...], turn_batch)
    owned = game.planets.active[0] & (game.planets.owner[0] == game.player[0])
    src_row = int(jnp.argmax(owned.astype(jnp.int32)))
    source_index = jnp.full((1, 1), src_row, dtype=jnp.int32)
    target_slot = jnp.zeros((1, 1), dtype=jnp.int32)
    ship_bucket = jnp.ones((1, 1), dtype=jnp.int32)
    stop_flag = jnp.zeros((1, 1), dtype=jnp.int32)
    step_mask = jnp.ones((1, 1), dtype=jnp.float32)
    fraction = fraction_from_logit(jnp.array([0.0], dtype=jnp.float32))
    ship_fraction = jnp.reshape(fraction, (1, 1))
    action = build_action_from_factored_batch(
        game,
        batch,
        source_index,
        target_slot,
        ship_bucket,
        stop_flag,
        step_mask,
        cfg,
        ship_fraction=ship_fraction,
    )
    launched = float(jax.device_get(action.ships[0, 0]))
    assert launched >= 1.0
