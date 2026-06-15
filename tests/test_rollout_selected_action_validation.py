"""Rollout selected_validate sampling and replay parity."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jax
from src.config import TrainConfig
from src.config.schema import TaskConfig
from src.features.registry import edge_k
from src.game.constants import MAX_PLANETS
from src.jax.action_sampling import _sample_shielded_factored_sequence_with_params
from src.jax.env import batched_reset
from src.jax.factored_sequence_scan import (
    owned_planet_ships_from_turn_batch,
    replay_factored_sequence_logprob,
)
from src.jax.policy import build_planet_graph_transformer_policy
from src.jax.shield import (
    apply_cheap_trajectory_shield_factorized_topk,
    rollout_factorized_sampling_mode,
    selected_factored_launch_passes_cheap_shield_jax,
    ship_count_for_bucket_jax,
)

from tests.test_trajectory_shield_factorized import _modes_cfg, _two_planet_game


def _task_cfg(**kwargs) -> TaskConfig:
    base = dict(candidate_count=4, ship_bucket_count=4, max_fleets=8)
    base.update(kwargs)
    return TaskConfig(**base)


def _train_cfg(**kwargs) -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.model.pointer_decoder = "factorized_topk"
    cfg.model.hidden_size = 32
    cfg.model.max_moves_k = 2
    cfg.task = _task_cfg(**kwargs.pop("task", {}))
    for key, value in kwargs.pop("model", {}).items():
        setattr(cfg.model, key, value)
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


@pytest.mark.jax
def test_rollout_factorized_sampling_mode_accepts_selected_validate() -> None:
    cfg = _modes_cfg(rollout_factorized_sampling="selected_validate")
    assert rollout_factorized_sampling_mode(cfg) == "selected_validate"


@pytest.mark.jax
def test_pointwise_cheap_shield_matches_lattice_cell() -> None:
    cfg = _modes_cfg()
    game = _two_planet_game(x0=20.0, y0=20.0, x1=80.0, y1=20.0)
    from src.jax.features import encode_turn

    batch = encode_turn(game, cfg)
    lattice = apply_cheap_trajectory_shield_factorized_topk(game, batch, cfg)
    mask = lattice.ship_bucket_mask
    k = edge_k(cfg)
    ships = game.planets.ships
    for src in range(MAX_PLANETS):
        for slot in range(k):
            for bucket in range(1, cfg.ship_bucket_count):
                lattice_ok = bool(np.asarray(mask[src, slot, bucket]))
                launch_ships = float(
                    np.asarray(
                        ship_count_for_bucket_jax(
                            ships[src], jnp.asarray(bucket, dtype=jnp.int32), cfg.ship_bucket_count
                        )
                    )
                )
                if launch_ships <= 0.0:
                    continue
                point_ok = bool(
                    np.asarray(
                        selected_factored_launch_passes_cheap_shield_jax(
                            game,
                            batch,
                            cfg,
                            jnp.asarray(src, dtype=jnp.int32),
                            jnp.asarray(slot, dtype=jnp.int32),
                            jnp.asarray(bucket, dtype=jnp.int32),
                            jnp.asarray(launch_ships, dtype=jnp.float32),
                            jnp.asarray(0.0, dtype=jnp.float32),
                            jnp.asarray(True),
                        )
                    )
                )
                assert point_ok == lattice_ok


@pytest.mark.jax
def test_replay_rebuilt_cheap_mask_matches_rollout_stack() -> None:
    """Replay cheap-mask rebuild must match rollout lattice for same remaining_ships."""
    from src.jax.factored_sequence_scan import (
        owned_planet_ships_from_turn_batch,
        shield_bucket_mask_for_replay_step,
    )
    from src.jax.features import encode_turn
    from src.jax.shield.trajectory import cheap_factorized_topk_masks_from_remaining

    train_cfg = _train_cfg(
        task={
            "trajectory_shield_mode": "cheap",
            "rollout_factorized_sampling": "selected_validate",
        }
    )
    game = _two_planet_game(x0=20.0, y0=20.0, x1=80.0, y1=20.0)
    batch = encode_turn(game, train_cfg.task)
    remaining = owned_planet_ships_from_turn_batch(batch, train_cfg.task)
    ships_arg = remaining[0] if remaining.ndim > 1 else remaining
    rollout_lattice = apply_cheap_trajectory_shield_factorized_topk(
        game, batch, train_cfg.task, remaining_planet_ships=ships_arg
    ).ship_bucket_mask
    stored_unshielded = jnp.zeros_like(rollout_lattice)
    rebuilt = shield_bucket_mask_for_replay_step(
        train_cfg,
        batch,
        remaining,
        stored_unshielded,
    )
    direct = cheap_factorized_topk_masks_from_remaining(
        batch, train_cfg.task, remaining
    ).ship_bucket_mask
    np.testing.assert_array_equal(np.asarray(rebuilt), np.asarray(rollout_lattice))
    np.testing.assert_array_equal(np.asarray(rebuilt), np.asarray(direct))


@pytest.mark.jax
def test_rollout_replay_logprob_parity_selected_validate() -> None:
    cfg = _train_cfg(
        task={
            "trajectory_shield_mode": "cheap",
            "rollout_factorized_sampling": "selected_validate",
        },
        model={"max_moves_k": 3},
    )
    state, batch = batched_reset(jax.random.split(jax.random.PRNGKey(81), 1), cfg.task)
    policy = build_planet_graph_transformer_policy(cfg)
    params = policy.init(jax.random.PRNGKey(82), batch)
    player_count = jnp.full((1,), cfg.task.player_count, dtype=jnp.int32)

    sample = _sample_shielded_factored_sequence_with_params(
        jax.random.PRNGKey(83),
        state.game,
        batch,
        params,
        policy,
        cfg,
        deterministic=True,
    )
    initial_ships = owned_planet_ships_from_turn_batch(batch, cfg.task)
    replay = replay_factored_sequence_logprob(
        params,
        policy,
        batch,
        cfg,
        player_count=player_count,
        source_index=sample.source_index,
        target_slot=sample.target_slot,
        ship_bucket=sample.ship_bucket,
        stop_flag=sample.stop_flag.astype(jnp.float32),
        step_mask=sample.step_mask,
        ship_bucket_mask=sample.ship_bucket_mask,
        ship_fraction=sample.ship_fraction,
        initial_remaining_ships=initial_ships,
    )
    np.testing.assert_allclose(
        np.asarray(replay.log_prob),
        np.asarray(sample.log_prob),
        rtol=1e-5,
        atol=1e-4,
    )
