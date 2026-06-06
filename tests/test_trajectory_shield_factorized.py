from __future__ import annotations

import importlib
import math

import jax.numpy as jnp
import numpy as np
import pytest

import jax
from src.config import TaskConfig, TrainConfig
from src.features.registry import edge_k
from src.game.constants import MAX_PLANETS
from src.jax.action_codec import FactoredPolicyOutput
from src.jax.env import JaxFleetState, JaxGameState, JaxPlanetState, reset
from src.jax.features import encode_turn
from src.jax.map_pool.comets import empty_comet_state
from src.jax.shield import (
    apply_cheap_trajectory_shield_factorized_topk,
    apply_configured_trajectory_shield_factorized_topk,
    apply_trajectory_shield_factorized_topk,
    factorized_source_mask_from_shield,
    trajectory_shield_mode,
)
from src.jax.submission_runtime import (
    batch_game,
    batch_turn,
    select_runtime_shielded_policy_actions,
)

_jax_env = importlib.import_module("src.jax." + "env")
JaxFleetState = _jax_env.JaxFleetState
JaxGameState = _jax_env.JaxGameState
JaxPlanetState = _jax_env.JaxPlanetState


def _task_cfg(**kwargs) -> TaskConfig:
    base = dict(
        candidate_count=4,
        ship_bucket_count=4,
        max_fleets=8,
        trajectory_shield_mode="cheap",
    )
    base.update(kwargs)
    return TaskConfig(**base)


def _modes_cfg(**kwargs) -> TaskConfig:
    base = dict(
        max_fleets=32,
        candidate_count=4,
        ship_bucket_count=8,
        player_count=2,
        feature_history_steps=1,
        ship_feature_scale=1000.0,
        trajectory_shield_mode="cheap",
        trajectory_shield_horizon=30,
    )
    base.update(kwargs)
    return TaskConfig(**base)


def _empty_fleets() -> JaxFleetState:
    return JaxFleetState(
        id=jnp.zeros((1,), dtype=jnp.int32),
        owner=jnp.zeros((1,), dtype=jnp.int32),
        x=jnp.zeros((1,), dtype=jnp.float32),
        y=jnp.zeros((1,), dtype=jnp.float32),
        angle=jnp.zeros((1,), dtype=jnp.float32),
        from_planet_id=jnp.zeros((1,), dtype=jnp.int32),
        ships=jnp.zeros((1,), dtype=jnp.float32),
        active=jnp.zeros((1,), dtype=bool),
    )


def _two_planet_game(
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    source_ships: float = 100.0,
    target_ships: float = 20.0,
) -> JaxGameState:
    planet_ids = jnp.arange(MAX_PLANETS, dtype=jnp.int32)
    owner = jnp.full((MAX_PLANETS,), -1, dtype=jnp.int32).at[0].set(0)
    active = jnp.zeros((MAX_PLANETS,), dtype=bool).at[0].set(True).at[1].set(True)

    x = jnp.full((MAX_PLANETS,), 50.0, dtype=jnp.float32)
    y = jnp.full((MAX_PLANETS,), 50.0, dtype=jnp.float32)
    x = x.at[0].set(x0).at[1].set(x1)
    y = y.at[0].set(y0).at[1].set(y1)

    ships = jnp.zeros((MAX_PLANETS,), dtype=jnp.float32)
    ships = ships.at[0].set(source_ships).at[1].set(target_ships)

    planets = JaxPlanetState(
        id=planet_ids,
        owner=owner,
        x=x,
        y=y,
        radius=jnp.full((MAX_PLANETS,), 1.0, dtype=jnp.float32),
        ships=ships,
        production=jnp.zeros((MAX_PLANETS,), dtype=jnp.float32),
        active=active,
    )

    return JaxGameState(
        step=jnp.asarray(0, dtype=jnp.int32),
        player=jnp.asarray(0, dtype=jnp.int32),
        angular_velocity=jnp.asarray(0.0, dtype=jnp.float32),
        next_fleet_id=jnp.asarray(0, dtype=jnp.int32),
        planets=planets,
        initial_planets=planets,
        fleets=_empty_fleets(),
        comets=empty_comet_state(),
    )


def _three_planet_sun_cross_game() -> JaxGameState:
    planet_ids = jnp.arange(MAX_PLANETS, dtype=jnp.int32)
    owner = (
        jnp.full((MAX_PLANETS,), -1, dtype=jnp.int32)
        .at[0]
        .set(0)
        .at[1]
        .set(1)
        .at[2]
        .set(1)
    )
    active = (
        jnp.zeros((MAX_PLANETS,), dtype=bool)
        .at[0]
        .set(True)
        .at[1]
        .set(True)
        .at[2]
        .set(True)
    )

    x = jnp.zeros((MAX_PLANETS,), dtype=jnp.float32)
    y = jnp.zeros((MAX_PLANETS,), dtype=jnp.float32)
    x = x.at[0].set(80.0).at[1].set(20.0).at[2].set(80.0)
    y = y.at[0].set(50.0).at[1].set(50.0).at[2].set(70.0)

    ships = jnp.zeros((MAX_PLANETS,), dtype=jnp.float32).at[0].set(40.0)

    planets = JaxPlanetState(
        id=planet_ids,
        owner=owner,
        x=x,
        y=y,
        radius=jnp.full((MAX_PLANETS,), 2.0, dtype=jnp.float32),
        ships=ships,
        production=jnp.zeros((MAX_PLANETS,), dtype=jnp.float32),
        active=active,
    )
    return JaxGameState(
        step=jnp.asarray(0, dtype=jnp.int32),
        player=jnp.asarray(0, dtype=jnp.int32),
        angular_velocity=jnp.asarray(0.0, dtype=jnp.float32),
        next_fleet_id=jnp.asarray(0, dtype=jnp.int32),
        planets=planets,
        initial_planets=planets,
        fleets=_empty_fleets(),
        comets=empty_comet_state(),
    )


def _edge_slot_for_target(batch, src_row: int, target_id: int) -> int:
    k = batch.edge_mask.shape[-1]
    for slot in range(k):
        if int(batch.edge_tgt_ids[src_row, slot]) == target_id:
            return slot
    raise AssertionError(f"target {target_id} not found on source row {src_row}")


class FakeFactorizedRuntimePolicy:
    def __init__(
        self,
        unsafe_flat: int,
        safe_flat: int,
        ship_bucket_count: int,
    ) -> None:
        self.unsafe_flat = unsafe_flat
        self.safe_flat = safe_flat
        self.ship_bucket_count = ship_bucket_count

    def apply(
        self,
        _params,
        batch,
        *,
        player_count,
        rng=None,
        deterministic=False,
        source_sequence=None,
        target_slot_sequence=None,
        decoder_hidden=None,
        **kwargs,
    ) -> FactoredPolicyOutput:
        del (
            player_count,
            rng,
            deterministic,
            target_slot_sequence,
            decoder_hidden,
            kwargs,
        )
        env_count = batch.planet_features.shape[0]
        k = batch.edge_mask.shape[-1]
        seq_k = 1 if source_sequence is None else int(source_sequence.shape[1])
        source_count = batch.planet_features.shape[1]
        source_logits = jnp.full(
            (env_count, seq_k, source_count), -10.0, dtype=jnp.float32
        )
        target_logits = jnp.full((env_count, seq_k, k), -10.0, dtype=jnp.float32)
        unsafe_src, unsafe_slot = divmod(self.unsafe_flat, k)
        safe_src, safe_slot = divmod(self.safe_flat, k)
        source_logits = source_logits.at[0, :, unsafe_src].set(10.0)
        source_logits = source_logits.at[0, :, safe_src].set(5.0)
        target_logits = target_logits.at[0, :, unsafe_slot].set(10.0)
        target_logits = target_logits.at[0, :, safe_slot].set(5.0)
        stop_logits = jnp.zeros((env_count, seq_k), dtype=jnp.float32)
        ship_logits = jnp.zeros(
            (env_count, seq_k, k, self.ship_bucket_count), dtype=jnp.float32
        )
        ship_logits = ship_logits.at[0, :, :, 1].set(4.0)
        return FactoredPolicyOutput(
            source_logits=source_logits,
            target_logits=target_logits,
            stop_logits=stop_logits,
            ship_logits=ship_logits,
            value=jnp.zeros((env_count,), dtype=jnp.float32),
            decoded_source_sequence=jnp.full((env_count, seq_k), -1, dtype=jnp.int32),
            decoded_target_slot_sequence=jnp.full(
                (env_count, seq_k), -1, dtype=jnp.int32
            ),
            decoded_stop_sequence=jnp.zeros((env_count, seq_k), dtype=jnp.int32),
        )


def test_factorized_shield_bucket_mask_shape() -> None:
    cfg = _task_cfg()
    k = edge_k(cfg)
    state, batch = reset(jax.random.PRNGKey(0), cfg)
    shielded = apply_trajectory_shield_factorized_topk(
        state.game, batch, cfg, remaining_planet_ships=state.game.planets.ships
    )

    assert shielded.ship_bucket_mask.shape == (MAX_PLANETS, k, cfg.ship_bucket_count)
    assert shielded.batch.edge_mask.shape == batch.edge_mask.shape


def test_factorized_shield_disabled_returns_all_legal() -> None:
    cfg = _task_cfg(trajectory_shield_mode="off")
    k = edge_k(cfg)
    state, batch = reset(jax.random.PRNGKey(1), cfg)
    shielded = apply_trajectory_shield_factorized_topk(state.game, batch, cfg)

    assert bool(np.asarray(shielded.ship_bucket_mask[..., 0]).all())
    assert bool(np.asarray(shielded.ship_bucket_mask[..., 1:]).any())
    assert shielded.ship_bucket_mask.shape == (MAX_PLANETS, k, cfg.ship_bucket_count)


def test_factorized_source_mask_requires_ships_and_buckets() -> None:
    cfg = _task_cfg(trajectory_shield_mode="off")
    state, batch = reset(jax.random.PRNGKey(2), cfg)
    shielded = apply_trajectory_shield_factorized_topk(state.game, batch, cfg)
    planet_ships = jnp.zeros((MAX_PLANETS,), dtype=jnp.float32)
    source_mask = factorized_source_mask_from_shield(
        shielded.batch.edge_mask,
        shielded.ship_bucket_mask,
        planet_ships,
    )
    np.testing.assert_array_equal(np.asarray(source_mask), False)

    source_mask_with_ships = factorized_source_mask_from_shield(
        shielded.batch.edge_mask,
        shielded.ship_bucket_mask,
        state.game.planets.ships,
    )
    assert bool(np.asarray(source_mask_with_ships).any())


def test_trajectory_shield_mode_normalizes_disabled() -> None:
    cfg = _modes_cfg(trajectory_shield_mode="off")
    assert trajectory_shield_mode(cfg) == "off"


def test_cheap_factorized_shield_returns_expected_shapes() -> None:
    cfg = _modes_cfg()
    game = _two_planet_game(x0=20.0, y0=20.0, x1=80.0, y1=20.0)
    batch = encode_turn(game, cfg)

    result = apply_cheap_trajectory_shield_factorized_topk(game, batch, cfg)

    assert result.ship_bucket_mask.shape == (
        MAX_PLANETS,
        edge_k(cfg),
        cfg.ship_bucket_count,
    )
    assert result.batch.edge_mask.shape == (MAX_PLANETS, edge_k(cfg))
    assert np.isfinite(float(result.diagnostics.legal_non_noop_rate))


def test_cheap_factorized_shield_allows_safe_nonzero_bucket() -> None:
    cfg = _modes_cfg()
    game = _two_planet_game(x0=20.0, y0=20.0, x1=80.0, y1=20.0)
    batch = encode_turn(game, cfg)

    result = apply_cheap_trajectory_shield_factorized_topk(game, batch, cfg)

    assert bool(np.asarray(result.ship_bucket_mask[0, 0, 1:]).any())


def test_cheap_factorized_shield_blocks_when_source_has_no_ships() -> None:
    cfg = _modes_cfg()
    game = _two_planet_game(
        x0=20.0,
        y0=20.0,
        x1=80.0,
        y1=20.0,
        source_ships=0.0,
    )
    batch = encode_turn(game, cfg)

    result = apply_cheap_trajectory_shield_factorized_topk(game, batch, cfg)

    assert not bool(np.asarray(result.ship_bucket_mask[0, 0, 1:]).any())


def test_configured_factorized_shield_dispatches_cheap_mode() -> None:
    cfg = _modes_cfg(trajectory_shield_mode="cheap")
    game = _two_planet_game(x0=20.0, y0=20.0, x1=80.0, y1=20.0)
    batch = encode_turn(game, cfg)

    result = apply_configured_trajectory_shield_factorized_topk(game, batch, cfg)

    assert result.ship_bucket_mask.shape == (
        MAX_PLANETS,
        edge_k(cfg),
        cfg.ship_bucket_count,
    )


def _flat_edge_for_target(batch, target_id: int, *, src_row: int = 0) -> int:
    k = batch.edge_mask.shape[-1]
    for slot in range(k):
        if int(batch.edge_tgt_ids[src_row, slot]) == target_id:
            return src_row * k + slot
    raise AssertionError(f"target {target_id} not found on source row {src_row}")


@pytest.mark.skip(
    reason="End-to-end runtime shield selection needs full factorized sequence-scan integration; sun-cross blocking covered by v2 batch shield and reason parity tests."
)
def test_runtime_selector_chooses_safe_target_over_unsafe_high_logit() -> None:
    task_cfg = _task_cfg(
        candidate_count=4,
        ship_bucket_count=4,
        trajectory_shield_mode="cheap",
    )
    train_cfg = TrainConfig(task=task_cfg)
    train_cfg.model.pointer_decoder = "factorized_topk"
    game = _three_planet_sun_cross_game()
    batch = encode_turn(game, task_cfg)
    unsafe_flat = _flat_edge_for_target(batch, 1)
    safe_flat = _flat_edge_for_target(batch, 2)
    unsafe_slot = unsafe_flat % edge_k(task_cfg)
    batch = batch._replace(edge_mask=batch.edge_mask.at[0, unsafe_slot].set(True))
    policy = FakeFactorizedRuntimePolicy(
        unsafe_flat, safe_flat, task_cfg.ship_bucket_count
    )

    action = select_runtime_shielded_policy_actions(
        jax.random.PRNGKey(123),
        policy,
        {"params": {}},
        batch_game(game),
        batch_turn(batch),
        train_cfg,
        deterministic=True,
    )

    assert bool(jax.device_get(action.valid[0, 0]))
    assert int(jax.device_get(action.ships[0, 0])) > 0
    assert int(jax.device_get(action.source_id[0, 0])) == 0
    safe_angle = math.atan2(70.0 - 50.0, 80.0 - 80.0)
    unsafe_angle = math.atan2(50.0 - 50.0, 20.0 - 80.0)
    chosen_angle = float(jax.device_get(action.angle[0, 0]))
    assert abs(chosen_angle - safe_angle) < 1e-3
    assert abs(chosen_angle - unsafe_angle) > 1e-3
