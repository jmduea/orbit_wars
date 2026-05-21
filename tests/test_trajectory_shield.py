import math

import jax
import jax.numpy as jnp
import numpy as np

from src.config import EnvConfig
from src.constants import MAX_PLANETS
from src.features import encode_turn
from src.game_types import GameState, PlanetState
from src.jax_env import JaxFleetState, JaxGameState, JaxPlanetState
from src.jax_policy import JaxPolicyOutput
from src.trajectory_shield import (
    apply_trajectory_shield_to_turn_batch,
    mask_policy_output_for_shield,
    select_runtime_shielded_policy_actions,
    trajectory_shield_reason_for_launch,
    trajectory_shield_reason_for_launch_jax,
    trajectory_shield_reason_name,
)


class FakeShieldBatch:
    def __init__(
        self,
        candidate_mask: jax.Array,
        candidate_ids: jax.Array,
        target_angles: jax.Array,
        source_ids: jax.Array,
        source_ships: jax.Array,
    ) -> None:
        self.candidate_mask = candidate_mask
        self.candidate_ids = candidate_ids
        self.target_angles = target_angles
        self.source_ids = source_ids
        self.source_ships = source_ships

    def _replace(self, **kwargs):
        return FakeShieldBatch(
            kwargs.get("candidate_mask", self.candidate_mask),
            kwargs.get("candidate_ids", self.candidate_ids),
            kwargs.get("target_angles", self.target_angles),
            kwargs.get("source_ids", self.source_ids),
            kwargs.get("source_ships", self.source_ships),
        )


class FakeRuntimePolicy:
    def __init__(self, unsafe_slot: int, safe_slot: int, candidate_count: int) -> None:
        self.unsafe_slot = unsafe_slot
        self.safe_slot = safe_slot
        self.candidate_count = candidate_count

    def apply(self, *_args, **_kwargs) -> JaxPolicyOutput:
        target_logits = jnp.full((1, 1, self.candidate_count), -10.0, dtype=jnp.float32)
        target_logits = target_logits.at[0, 0, 0].set(0.0)
        target_logits = target_logits.at[0, 0, self.safe_slot].set(5.0)
        target_logits = target_logits.at[0, 0, self.unsafe_slot].set(10.0)
        ship_logits = jnp.zeros((1, 1, self.candidate_count, 4), dtype=jnp.float32)
        ship_logits = ship_logits.at[0, 0, :, 1].set(4.0)
        return JaxPolicyOutput(
            target_logits=target_logits,
            ship_logits=ship_logits,
            value=jnp.zeros((1,), dtype=jnp.float32),
            decoded_target_sequence=jnp.full((1, 1), -1, dtype=jnp.int32),
        )


def _cfg(**overrides) -> EnvConfig:
    cfg = EnvConfig(candidate_count=4, ship_bucket_count=4, max_fleets=8)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _planet(pid: int, owner: int, x: float, y: float, ships: int = 30) -> PlanetState:
    return PlanetState(pid, owner, x, y, 2.0, ships, 1)


def _state(planets: list[PlanetState], *, player: int = 0, step: int = 0, angular_velocity: float = 0.0) -> GameState:
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


def _jax_game(planets: list[PlanetState], *, player: int = 0, step: int = 0, angular_velocity: float = 0.0) -> JaxGameState:
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
    )


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
            [_planet(0, 0, 20.0, 20.0), _planet(1, 1, 80.0, 20.0), _planet(2, -1, 50.0, 20.0)],
            0,
            1,
            "unintended_hit",
        ),
        (
            _cfg(trajectory_shield_hit_mode="non_friendly"),
            [_planet(0, 0, 20.0, 20.0), _planet(1, 1, 80.0, 20.0), _planet(2, -1, 50.0, 20.0)],
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


def test_python_candidate_mask_keeps_targets_visible_while_shield_blocks_illegal_hits() -> None:
    planets = [
        _planet(0, 0, 20.0, 20.0, ships=40),
        _planet(1, 1, 80.0, 20.0),
        _planet(2, -1, 50.0, 20.0),
    ]
    selected_target_batch = encode_turn(_state(planets), _cfg(), env_index=0)
    non_friendly_batch = encode_turn(
        _state(planets), _cfg(trajectory_shield_hit_mode="non_friendly"), env_index=0
    )
    selected_context = selected_target_batch.contexts[0]
    non_friendly_context = non_friendly_batch.contexts[0]
    selected_target_slot = selected_context.candidate_ids.index(1)
    non_friendly_target_slot = non_friendly_context.candidate_ids.index(1)

    assert bool(selected_target_batch.candidate_mask[0, selected_target_slot])
    assert bool(non_friendly_batch.candidate_mask[0, non_friendly_target_slot])

    def fake_from_batch(batch):
        return FakeShieldBatch(
            candidate_mask=jnp.asarray(batch.candidate_mask),
            candidate_ids=jnp.asarray(
                [context.candidate_ids for context in batch.contexts], dtype=jnp.int32
            ),
            target_angles=jnp.asarray(
                [context.target_angles for context in batch.contexts], dtype=jnp.float32
            ),
            source_ids=jnp.asarray(
                [context.source_id for context in batch.contexts], dtype=jnp.int32
            ),
            source_ships=jnp.asarray(
                [context.source_ships for context in batch.contexts], dtype=jnp.float32
            ),
        )

    selected_shielded = apply_trajectory_shield_to_turn_batch(
        _jax_game(planets), fake_from_batch(selected_target_batch), _cfg()
    )
    non_friendly_shielded = apply_trajectory_shield_to_turn_batch(
        _jax_game(planets),
        fake_from_batch(non_friendly_batch),
        _cfg(trajectory_shield_hit_mode="non_friendly"),
    )

    assert not bool(selected_shielded.batch.candidate_mask[0, selected_target_slot])
    assert bool(non_friendly_shielded.batch.candidate_mask[0, non_friendly_target_slot])


def test_mask_policy_output_for_shield_applies_bucket_masks_to_all_pointer_steps() -> None:
    target_logits = jnp.asarray(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], dtype=jnp.float32
    )
    ship_logits = jnp.zeros((1, 3, 3, 4), dtype=jnp.float32)
    policy_output = JaxPolicyOutput(
        target_logits=target_logits,
        ship_logits=ship_logits,
        value=jnp.asarray([0.0], dtype=jnp.float32),
        decoded_target_sequence=jnp.asarray([[-1, -1, -1]], dtype=jnp.int32),
    )
    masked = mask_policy_output_for_shield(
        policy_output,
        jnp.asarray([[True, True, False]], dtype=bool),
        ship_bucket_count=4,
    )

    later_steps = np.asarray(masked.target_logits[0, 1:])
    assert np.isfinite(later_steps[:, 0]).all()
    assert np.isfinite(later_steps[:, 1]).all()
    assert (later_steps[:, 2:] < -1.0e30).all()
    noop_ship_logits = np.asarray(masked.ship_logits[0, :, 0, :])
    assert np.isfinite(noop_ship_logits[:, 0]).all()
    assert (noop_ship_logits[:, 1:] < -1.0e30).all()
    real_ship_logits = np.asarray(masked.ship_logits[0, :, 1, :])
    assert (real_ship_logits[:, 0] < -1.0e30).all()
    assert np.isfinite(real_ship_logits[:, 1:]).all()


def test_runtime_selector_chooses_safe_target_over_unsafe_high_logit() -> None:
    cfg = _cfg(candidate_count=4, ship_bucket_count=4)
    planets = [
        _planet(0, 0, 80.0, 50.0, ships=40),
        _planet(1, 1, 20.0, 50.0),
        _planet(2, 1, 80.0, 70.0),
    ]
    batch = encode_turn(_state(planets), cfg, env_index=0)
    context = batch.contexts[0]
    unsafe_slot = context.candidate_ids.index(1)
    safe_slot = context.candidate_ids.index(2)
    policy = FakeRuntimePolicy(unsafe_slot, safe_slot, cfg.candidate_count)

    selected = select_runtime_shielded_policy_actions(
        jax.random.PRNGKey(123),
        policy,
        {"params": {}},
        batch,
        cfg,
        deterministic=True,
    )

    assert int(selected.target_index[0, 0]) == safe_slot
    assert int(selected.ship_bucket[0, 0]) > 0


def test_jax_batch_shield_reports_blocked_metrics_for_sun_crossing() -> None:
    cfg = _cfg()
    planets = [_planet(0, 0, 80.0, 50.0, ships=40), _planet(1, 1, 20.0, 50.0)]
    game = _jax_game(planets)
    batch = encode_turn(_state(planets), cfg, env_index=0)
    original_candidate_mask = jnp.asarray(batch.candidate_mask).at[:, 1:].set(True)
    fake_batch = FakeShieldBatch(
        candidate_mask=original_candidate_mask,
        candidate_ids=jnp.asarray(
            [context.candidate_ids for context in batch.contexts], dtype=jnp.int32
        ),
        target_angles=jnp.asarray(
            [context.target_angles for context in batch.contexts], dtype=jnp.float32
        ),
        source_ids=jnp.asarray([context.source_id for context in batch.contexts], dtype=jnp.int32),
        source_ships=jnp.asarray(
            [context.source_ships for context in batch.contexts], dtype=jnp.float32
        ),
    )
    shielded = apply_trajectory_shield_to_turn_batch(
        game,
        fake_batch,
        cfg,
    )

    assert float(shielded.diagnostics.blocked_count) >= 1.0
    assert float(shielded.diagnostics.blocked_sun_count) >= 1.0


def test_jax_batch_shield_allows_static_launches_on_mixed_rotating_maps() -> None:
    cfg = _cfg()
    planets = [
        _planet(0, 0, 90.0, 90.0, ships=40),
        _planet(1, 1, 90.0, 80.0),
        _planet(2, -1, 50.0, 20.0),
    ]
    batch = FakeShieldBatch(
        candidate_mask=jnp.asarray([[True, True, False, False]], dtype=bool),
        candidate_ids=jnp.asarray([[-1, 1, -1, -1]], dtype=jnp.int32),
        target_angles=jnp.asarray([[0.0, -math.pi / 2.0, 0.0, 0.0]], dtype=jnp.float32),
        source_ids=jnp.asarray([0], dtype=jnp.int32),
        source_ships=jnp.asarray([40.0], dtype=jnp.float32),
    )

    shielded = apply_trajectory_shield_to_turn_batch(_jax_game(planets), batch, cfg)

    assert bool(shielded.batch.candidate_mask[0, 1])
    assert bool(shielded.ship_bucket_mask[0, 1, 1])
    assert float(shielded.diagnostics.legal_non_noop_rate) == 1.0


def test_jax_batch_shield_keeps_target_when_some_ship_buckets_are_safe() -> None:
    cfg = _cfg(trajectory_shield_horizon=1)
    planets = [_planet(0, 0, 20.0, 20.0, ships=1000), _planet(1, 1, 29.0, 20.0)]
    batch = FakeShieldBatch(
        candidate_mask=jnp.asarray([[True, True, False, False]], dtype=bool),
        candidate_ids=jnp.asarray([[0, 1, -1, -1]], dtype=jnp.int32),
        target_angles=jnp.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
        source_ids=jnp.asarray([0], dtype=jnp.int32),
        source_ships=jnp.asarray([1000.0], dtype=jnp.float32),
    )

    shielded = apply_trajectory_shield_to_turn_batch(_jax_game(planets), batch, cfg)

    assert bool(shielded.batch.candidate_mask[0, 1])
    assert not bool(shielded.ship_bucket_mask[0, 1, 0])
    assert not bool(shielded.ship_bucket_mask[0, 1, 1])
    assert bool(shielded.ship_bucket_mask[0, 1, 2])
    assert bool(shielded.ship_bucket_mask[0, 1, 3])


def test_jax_batch_shield_recomputes_bucket_legality_from_remaining_ships() -> None:
    cfg = _cfg(trajectory_shield_horizon=1)
    planets = [_planet(0, 0, 20.0, 20.0, ships=1000), _planet(1, 1, 27.0, 20.0)]
    batch = FakeShieldBatch(
        candidate_mask=jnp.asarray([[True, True, False, False]], dtype=bool),
        candidate_ids=jnp.asarray([[0, 1, -1, -1]], dtype=jnp.int32),
        target_angles=jnp.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
        source_ids=jnp.asarray([0], dtype=jnp.int32),
        source_ships=jnp.asarray([1000.0], dtype=jnp.float32),
    )

    initial = apply_trajectory_shield_to_turn_batch(
        _jax_game(planets),
        batch,
        cfg,
        source_ships_override=jnp.asarray([1000.0], dtype=jnp.float32),
    )
    later = apply_trajectory_shield_to_turn_batch(
        _jax_game(planets),
        batch,
        cfg,
        source_ships_override=jnp.asarray([100.0], dtype=jnp.float32),
    )

    assert bool(initial.ship_bucket_mask[0, 1, 1])
    assert not bool(later.ship_bucket_mask[0, 1, 1])
    assert bool(later.ship_bucket_mask[0, 1, 2])
