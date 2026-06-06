from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from src.config import TaskConfig
from src.game.constants import MAX_PLANETS
from src.game.shield import trajectory_shield_reason_for_launch
from src.game.types import GameState, PlanetState
from src.jax.env import JaxFleetState, JaxGameState, JaxPlanetState
from src.jax.features import encode_turn
from src.jax.map_pool.comets import empty_comet_state
from src.jax.policy import JaxPolicyOutput
from src.jax.shield import (
    apply_trajectory_shield_to_turn_batch_v2,
    mask_policy_output_for_shield_v2,
    trajectory_shield_reason_for_launch_jax,
    trajectory_shield_reason_name,
)


def _cfg(**overrides) -> TaskConfig:
    cfg = TaskConfig(
        candidate_count=4,
        ship_bucket_count=4,
        max_fleets=8,
        trajectory_shield_mode="exact",
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _planet(pid: int, owner: int, x: float, y: float, ships: int = 30) -> PlanetState:
    return PlanetState(pid, owner, x, y, 2.0, ships, 1)


def _state(
    planets: list[PlanetState],
    *,
    player: int = 0,
    step: int = 0,
    angular_velocity: float = 0.0,
) -> GameState:
    return GameState(
        step=step,
        player=player,
        planets=planets,
        fleets=[],
        angular_velocity=angular_velocity,
        initial_planets=[
            PlanetState(
                planet.id,
                planet.owner,
                planet.x,
                planet.y,
                planet.radius,
                planet.ships,
                planet.production,
            )
            for planet in planets
        ],
    )


def _jax_planets(planets: list[PlanetState]) -> JaxPlanetState:
    pad = MAX_PLANETS - len(planets)
    ids = [planet.id for planet in planets] + list(range(len(planets), MAX_PLANETS))
    owner = [planet.owner for planet in planets] + [-1] * pad
    x = [planet.x for planet in planets] + [0.0] * pad
    y = [planet.y for planet in planets] + [0.0] * pad
    radius = [planet.radius for planet in planets] + [0.0] * pad
    ships = [float(planet.ships) for planet in planets] + [0.0] * pad
    production = [float(planet.production) for planet in planets] + [0.0] * pad
    active = [True] * len(planets) + [False] * pad
    return JaxPlanetState(
        id=jnp.asarray(ids, dtype=jnp.int32),
        owner=jnp.asarray(owner, dtype=jnp.int32),
        x=jnp.asarray(x, dtype=jnp.float32),
        y=jnp.asarray(y, dtype=jnp.float32),
        radius=jnp.asarray(radius, dtype=jnp.float32),
        ships=jnp.asarray(ships, dtype=jnp.float32),
        production=jnp.asarray(production, dtype=jnp.float32),
        active=jnp.asarray(active, dtype=bool),
    )


def _jax_game(
    planets: list[PlanetState],
    *,
    player: int = 0,
    step: int = 0,
    angular_velocity: float = 0.0,
) -> JaxGameState:
    planet_state = _jax_planets(planets)
    fleet_state = JaxFleetState(
        id=jnp.full((8,), -1, dtype=jnp.int32),
        owner=jnp.full((8,), -1, dtype=jnp.int32),
        x=jnp.zeros((8,), dtype=jnp.float32),
        y=jnp.zeros((8,), dtype=jnp.float32),
        angle=jnp.zeros((8,), dtype=jnp.float32),
        from_planet_id=jnp.full((8,), -1, dtype=jnp.int32),
        ships=jnp.zeros((8,), dtype=jnp.float32),
        active=jnp.zeros((8,), dtype=bool),
    )
    return JaxGameState(
        step=jnp.asarray(step, dtype=jnp.int32),
        player=jnp.asarray(player, dtype=jnp.int32),
        angular_velocity=jnp.asarray(angular_velocity, dtype=jnp.float32),
        next_fleet_id=jnp.asarray(0, dtype=jnp.int32),
        planets=planet_state,
        initial_planets=planet_state,
        fleets=fleet_state,
        comets=empty_comet_state(),
    )


def _flat_edge_for_target(batch, target_id: int, *, src_row: int = 0) -> int:
    k = batch.edge_mask.shape[-1]
    for slot in range(k):
        if int(batch.edge_tgt_ids[src_row, slot]) == target_id:
            return src_row * k + slot
    raise AssertionError(f"target {target_id} not found on source row {src_row}")


def _edge_slot_for_target(batch, src_row: int, target_id: int) -> int:
    k = batch.edge_mask.shape[-1]
    for slot in range(k):
        if int(batch.edge_tgt_ids[src_row, slot]) == target_id:
            return slot
    raise AssertionError(f"target {target_id} not found on source row {src_row}")


def test_python_and_jax_launch_reasons_match_for_sun_and_hit_modes() -> None:
    cases = [
        (
            _cfg(),
            [_planet(0, 0, 80.0, 50.0), _planet(1, 1, 20.0, 50.0)],
            0,
            1,
            "sun",
        ),
        (
            _cfg(),
            [
                _planet(0, 0, 20.0, 20.0),
                _planet(1, 1, 80.0, 20.0),
                _planet(2, -1, 50.0, 20.0),
            ],
            0,
            1,
            "unintended_hit",
        ),
        (
            _cfg(trajectory_shield_hit_mode="non_friendly"),
            [
                _planet(0, 0, 20.0, 20.0),
                _planet(1, 1, 80.0, 20.0),
                _planet(2, -1, 50.0, 20.0),
            ],
            0,
            1,
            "safe",
        ),
    ]

    for cfg, planets, source_id, target_id, expected in cases:
        state = _state(planets)
        source = planets[source_id]
        target = planets[target_id]
        angle = math.atan2(target.y - source.y, target.x - source.x)
        reason = trajectory_shield_reason_for_launch(
            state, source_id, target_id, angle, 20, cfg
        )
        game = _jax_game(planets)
        reason_code = trajectory_shield_reason_for_launch_jax(
            game,
            jnp.asarray(source_id, dtype=jnp.int32),
            jnp.asarray(target_id, dtype=jnp.int32),
            jnp.asarray(angle, dtype=jnp.float32),
            jnp.asarray(20.0, dtype=jnp.float32),
            game.player,
            cfg,
        )
        assert reason == expected
        assert trajectory_shield_reason_name(reason_code) == expected


def test_v2_shield_hit_mode_blocks_unintended_hits_but_non_friendly_allows() -> None:
    planets = [
        _planet(0, 0, 20.0, 20.0, ships=40),
        _planet(1, 1, 80.0, 20.0),
        _planet(2, -1, 50.0, 20.0),
    ]
    game = _jax_game(planets)
    batch = encode_turn(game, _cfg())
    target_slot = _edge_slot_for_target(batch, 0, 1)

    selected_shielded = apply_trajectory_shield_to_turn_batch_v2(game, batch, _cfg())
    non_friendly_shielded = apply_trajectory_shield_to_turn_batch_v2(
        game,
        batch,
        _cfg(trajectory_shield_hit_mode="non_friendly"),
    )

    assert not bool(selected_shielded.batch.edge_mask[0, target_slot])
    assert bool(non_friendly_shielded.batch.edge_mask[0, target_slot])


def test_mask_policy_output_for_shield_v2_applies_bucket_masks_to_all_pointer_steps() -> (
    None
):
    edge_count = 3
    bucket_count = 4
    target_logits = jnp.asarray(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], dtype=jnp.float32
    )
    ship_logits = jnp.zeros((1, 3, edge_count, bucket_count), dtype=jnp.float32)
    policy_output = JaxPolicyOutput(
        target_logits=target_logits,
        ship_logits=ship_logits,
        value=jnp.asarray([0.0], dtype=jnp.float32),
        decoded_target_sequence=jnp.asarray([[-1, -1, -1]], dtype=jnp.int32),
    )
    masked = mask_policy_output_for_shield_v2(
        policy_output,
        jnp.asarray([[True, True, False]], dtype=bool),
        bucket_count,
    )

    later_steps = np.asarray(masked.target_logits[0, 1:])
    assert np.isfinite(later_steps[:, 0]).all()
    assert np.isfinite(later_steps[:, 1]).all()
    assert (later_steps[:, 2:] < -1.0e30).all()
    noop_ship_logits = np.asarray(masked.ship_logits[0, :, 0, :])
    assert (noop_ship_logits[:, 0] < -1.0e30).all()
    assert np.isfinite(noop_ship_logits[:, 1:]).all()
    real_ship_logits = np.asarray(masked.ship_logits[0, :, 1, :])
    assert (real_ship_logits[:, 0] < -1.0e30).all()
    assert np.isfinite(real_ship_logits[:, 1:]).all()


def test_v2_batch_shield_allows_static_launches_on_mixed_rotating_maps() -> None:
    cfg = _cfg()
    planets = [
        _planet(0, 0, 90.0, 90.0, ships=40),
        _planet(1, 1, 90.0, 80.0),
        _planet(2, -1, 50.0, 20.0),
    ]
    game = _jax_game(planets)
    batch = encode_turn(game, cfg)
    shielded = apply_trajectory_shield_to_turn_batch_v2(game, batch, cfg)
    target_slot = _edge_slot_for_target(batch, 0, 1)
    flat_edge = target_slot

    assert bool(shielded.batch.edge_mask[0, target_slot])
    assert bool(shielded.ship_bucket_mask[flat_edge, 1])
    assert float(shielded.diagnostics.legal_non_noop_rate) == 1.0


def test_v2_batch_shield_keeps_target_when_some_ship_buckets_are_safe() -> None:
    cfg = _cfg(trajectory_shield_horizon=1)
    planets = [_planet(0, 0, 20.0, 20.0, ships=1000), _planet(1, 1, 29.0, 20.0)]
    game = _jax_game(planets)
    batch = encode_turn(game, cfg)
    shielded = apply_trajectory_shield_to_turn_batch_v2(game, batch, cfg)
    target_slot = _edge_slot_for_target(batch, 0, 1)
    flat_edge = target_slot

    assert bool(shielded.batch.edge_mask[0, target_slot])
    assert not bool(shielded.ship_bucket_mask[flat_edge, 0])
    assert not bool(shielded.ship_bucket_mask[flat_edge, 1])
    assert bool(shielded.ship_bucket_mask[flat_edge, 2])
    assert bool(shielded.ship_bucket_mask[flat_edge, 3])


def test_v2_batch_shield_recomputes_bucket_legality_from_remaining_ships() -> None:
    cfg = _cfg(trajectory_shield_horizon=1)
    planets = [_planet(0, 0, 20.0, 20.0, ships=1000), _planet(1, 1, 27.0, 20.0)]
    game = _jax_game(planets)
    batch = encode_turn(game, cfg)
    full_ships = game.planets.ships
    reduced_ships = full_ships.at[0].set(100.0)

    initial = apply_trajectory_shield_to_turn_batch_v2(
        game,
        batch,
        cfg,
        remaining_planet_ships=full_ships,
    )
    later = apply_trajectory_shield_to_turn_batch_v2(
        game,
        batch,
        cfg,
        remaining_planet_ships=reduced_ships,
    )
    flat_edge = _edge_slot_for_target(batch, 0, 1)

    assert bool(initial.ship_bucket_mask[flat_edge, 1])
    assert not bool(later.ship_bucket_mask[flat_edge, 1])
    assert bool(later.ship_bucket_mask[flat_edge, 2])


def test_v2_batch_shield_reports_blocked_metrics_for_sun_crossing() -> None:
    cfg = _cfg()
    planets = [_planet(0, 0, 80.0, 50.0, ships=40), _planet(1, 1, 20.0, 50.0)]
    game = _jax_game(planets)
    batch = encode_turn(game, cfg)
    target_slot = _edge_slot_for_target(batch, 0, 1)
    batch = batch._replace(edge_mask=batch.edge_mask.at[0, target_slot].set(True))

    shielded = apply_trajectory_shield_to_turn_batch_v2(game, batch, cfg)

    assert float(shielded.diagnostics.blocked_count) >= 1.0
    assert not bool(shielded.batch.edge_mask[0, target_slot])
