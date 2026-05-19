"""JAX feature encoder for fixed-shape Orbit Wars states."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import EnvConfig
from .features import (
    BOARD_CENTER,
    MAX_OWNER_FEATURE_PLAYERS,
    NO_OP_CANDIDATE_INDEX,
    PLANET_LAUNCH_RADIUS_OFFSET,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    BASE_CANDIDATE_FEATURE_DIM,
    BASE_GLOBAL_FEATURE_DIM,
    BASE_SELF_FEATURE_DIM,
)


class JaxFeatureHistory(NamedTuple):
    self_features: jax.Array
    candidate_features: jax.Array
    global_features: jax.Array
    source_ids: jax.Array
    candidate_ids: jax.Array
    cursor: jax.Array
    length: jax.Array


class JaxTurnBatch(NamedTuple):
    """Fixed-shape decision batch emitted by the JAX feature encoder.

    Leading dimensions are ``(num_envs, max_planets, ...)`` after batching or
    ``(max_planets, ...)`` for a single environment. ``decision_mask`` marks
    rows that correspond to learner-owned source planets; padding and non-source
    rows remain present to preserve static shapes.
    """

    self_features: jax.Array
    candidate_features: jax.Array
    global_features: jax.Array
    candidate_mask: jax.Array
    decision_mask: jax.Array
    source_ids: jax.Array
    source_ships: jax.Array
    candidate_ids: jax.Array
    target_angles: jax.Array


def encode_turn(
    game, env_cfg: EnvConfig, history: JaxFeatureHistory | None = None
) -> JaxTurnBatch:
    """Encode a JAX game state into fixed-shape policy inputs.

    The encoder mirrors the Torch/Python feature schema while avoiding dynamic
    lists. Candidate slot ``0`` is reserved for no-op and real targets are sorted
    into the remaining candidate slots using a JAX-friendly distance key.
    """

    planets = game.planets
    player = game.player
    mine = planets.active & (planets.owner == player)
    global_features = _global_features(game, env_cfg, history)
    self_features = _self_features(planets, game.fleets, player, env_cfg, history)
    candidate_ids, candidate_features, candidate_mask, target_angles = (
        _candidate_features(planets, game.fleets, player, env_cfg, history)
    )
    return JaxTurnBatch(
        self_features=_stack_self_history(self_features, history, env_cfg),
        candidate_features=_stack_candidate_history(
            candidate_features, planets.id, candidate_ids, history, env_cfg
        ),
        global_features=jnp.repeat(
            _stack_global_history(global_features, history, env_cfg)[None, :],
            env_cfg.max_planets,
            axis=0,
        ),
        candidate_mask=candidate_mask & mine[:, None],
        decision_mask=mine,
        source_ids=planets.id,
        source_ships=jnp.maximum(planets.ships, 0.0),
        candidate_ids=candidate_ids,
        target_angles=target_angles,
    )


def feature_history_steps(env_cfg: EnvConfig) -> int:
    return max(1, int(getattr(env_cfg, "feature_history_steps", 1)))


def empty_feature_history(env_cfg: EnvConfig) -> JaxFeatureHistory:
    steps = max(0, feature_history_steps(env_cfg) - 1)
    return JaxFeatureHistory(
        self_features=jnp.zeros(
            (steps, env_cfg.max_planets, BASE_SELF_FEATURE_DIM), dtype=jnp.float32
        ),
        candidate_features=jnp.zeros(
            (
                steps,
                env_cfg.max_planets,
                env_cfg.candidate_count,
                BASE_CANDIDATE_FEATURE_DIM,
            ),
            dtype=jnp.float32,
        ),
        global_features=jnp.zeros((steps, BASE_GLOBAL_FEATURE_DIM), dtype=jnp.float32),
        source_ids=jnp.full((steps, env_cfg.max_planets), -1, dtype=jnp.int32),
        candidate_ids=jnp.full(
            (steps, env_cfg.max_planets, env_cfg.candidate_count),
            -1,
            dtype=jnp.int32,
        ),
        cursor=jnp.asarray(0, dtype=jnp.int32),
        length=jnp.asarray(0, dtype=jnp.int32),
    )


def current_feature_snapshot(game, env_cfg: EnvConfig) -> JaxFeatureHistory:
    candidate_ids, candidates, _candidate_mask, _target_angles = _candidate_features(
        game.planets, game.fleets, game.player, env_cfg
    )
    return JaxFeatureHistory(
        self_features=_self_features(game.planets, game.fleets, game.player, env_cfg)[
            None, ...
        ],
        candidate_features=candidates[None, ...],
        global_features=_global_features(game, env_cfg)[None, ...],
        source_ids=game.planets.id[None, ...],
        candidate_ids=candidate_ids[None, ...],
        cursor=jnp.asarray(0, dtype=jnp.int32),
        length=jnp.asarray(1, dtype=jnp.int32),
    )


def append_feature_history(
    history: JaxFeatureHistory | None, game, env_cfg: EnvConfig
) -> JaxFeatureHistory:
    steps = max(0, feature_history_steps(env_cfg) - 1)
    if steps == 0:
        return empty_feature_history(env_cfg)
    base = empty_feature_history(env_cfg) if history is None else history
    current = current_feature_snapshot(game, env_cfg)
    idx = base.cursor
    next_cursor = (idx + 1) % steps
    next_length = jnp.minimum(base.length + 1, steps)
    return JaxFeatureHistory(
        self_features=base.self_features.at[idx].set(current.self_features[0]),
        candidate_features=base.candidate_features.at[idx].set(current.candidate_features[0]),
        global_features=base.global_features.at[idx].set(current.global_features[0]),
        source_ids=base.source_ids.at[idx].set(current.source_ids[0]),
        candidate_ids=base.candidate_ids.at[idx].set(current.candidate_ids[0]),
        cursor=next_cursor,
        length=next_length,
    )




def _ordered_history_indices(history: JaxFeatureHistory, steps: int) -> jax.Array:
    start = (history.cursor - history.length) % steps
    return (start + jnp.arange(steps, dtype=jnp.int32)) % steps


def _ordered_history(history: JaxFeatureHistory, env_cfg: EnvConfig) -> JaxFeatureHistory:
    steps = max(0, feature_history_steps(env_cfg) - 1)
    if steps == 0:
        return history
    indices = _ordered_history_indices(history, steps)
    valid = jnp.arange(steps, dtype=jnp.int32) >= (steps - history.length)
    return JaxFeatureHistory(
        self_features=history.self_features[indices] * valid[:, None, None],
        candidate_features=history.candidate_features[indices] * valid[:, None, None, None],
        global_features=history.global_features[indices] * valid[:, None],
        source_ids=jnp.where(valid[:, None], history.source_ids[indices], -1),
        candidate_ids=jnp.where(valid[:, None, None], history.candidate_ids[indices], -1),
        cursor=history.cursor,
        length=history.length,
    )

def _stack_self_history(
    current: jax.Array, history: JaxFeatureHistory | None, env_cfg: EnvConfig
) -> jax.Array:
    if feature_history_steps(env_cfg) <= 1:
        return current
    base = empty_feature_history(env_cfg) if history is None else history
    ordered = _ordered_history(base, env_cfg)
    return jnp.transpose(
        jnp.concatenate([ordered.self_features, current[None, ...]], axis=0), (1, 0, 2)
    ).reshape(env_cfg.max_planets, -1)


def _stack_candidate_history(
    current: jax.Array,
    source_ids: jax.Array,
    candidate_ids: jax.Array,
    history: JaxFeatureHistory | None,
    env_cfg: EnvConfig,
) -> jax.Array:
    if feature_history_steps(env_cfg) <= 1:
        return current
    base = empty_feature_history(env_cfg) if history is None else history
    ordered = _ordered_history(base, env_cfg)
    history_features = _target_aligned_candidate_history(
        source_ids, candidate_ids, ordered
    )
    stacked = jnp.concatenate([history_features, current[None, ...]], axis=0)
    return jnp.transpose(stacked, (1, 2, 0, 3)).reshape(
        env_cfg.max_planets, env_cfg.candidate_count, -1
    )


def _target_aligned_candidate_history(
    source_ids: jax.Array,
    candidate_ids: jax.Array,
    history: JaxFeatureHistory,
) -> jax.Array:
    source_matches = (
        history.source_ids[:, None, None, :, None]
        == source_ids[None, :, None, None, None]
    )
    target_matches = (
        history.candidate_ids[:, None, None, :, :]
        == candidate_ids[None, :, :, None, None]
    )
    valid_targets = candidate_ids[None, :, :, None, None] != -1
    matches = source_matches & target_matches & valid_targets
    selected = jnp.where(
        matches[..., None], history.candidate_features[:, None, None, :, :, :], 0.0
    )
    return selected.sum(axis=(3, 4))


def _stack_global_history(
    current: jax.Array, history: JaxFeatureHistory | None, env_cfg: EnvConfig
) -> jax.Array:
    if feature_history_steps(env_cfg) <= 1:
        return current
    base = empty_feature_history(env_cfg) if history is None else history
    ordered = _ordered_history(base, env_cfg)
    return jnp.concatenate([ordered.global_features, current[None, ...]], axis=0).reshape(
        -1
    )


def _self_features(
    planets,
    fleets,
    player,
    env_cfg: EnvConfig,
    history: JaxFeatureHistory | None = None,
):
    mine = planets.active & (planets.owner == player)
    enemy = planets.active & (planets.owner != -1) & (planets.owner != player)
    my_count = mine.astype(jnp.float32).sum()
    enemy_count = enemy.astype(jnp.float32).sum()
    my_ships = jnp.where(mine, planets.ships, 0.0).sum()
    enemy_ships = jnp.where(enemy, planets.ships, 0.0).sum()
    owner_counts, owner_ships, _owner_fleets, active_mask, player_count_feature = (
        owner_relative_summary(planets, fleets, player, env_cfg)
    )
    p = env_cfg.max_planets
    owner_context = jnp.broadcast_to(
        jnp.concatenate(
            [
                owner_counts,
                owner_ships,
                active_mask,
                jnp.asarray([player_count_feature], dtype=jnp.float32),
            ]
        ),
        (p, 13),
    )
    rotating = is_rotating_xy(planets.x, planets.y, planets.radius)
    base_features = jnp.stack(
        [
            planets.active.astype(jnp.float32),
            planets.x / env_cfg.board_size,
            planets.y / env_cfg.board_size,
            planets.radius / 5.0,
            jnp.minimum(planets.ships, env_cfg.max_ships) / env_cfg.max_ships,
            planets.production / env_cfg.max_production,
            rotating.astype(jnp.float32),
            jnp.full_like(planets.x, my_count / env_cfg.max_planets),
            jnp.full_like(planets.x, enemy_count / env_cfg.max_planets),
            jnp.full_like(
                planets.x, my_ships / (env_cfg.max_planets * env_cfg.max_ships)
            ),
            jnp.full_like(
                planets.x, enemy_ships / (env_cfg.max_planets * env_cfg.max_ships)
            ),
        ],
        axis=-1,
    )
    previous_self, previous_present = _previous_self_features(
        planets.id, history, env_cfg
    )
    previous_source_ships = jnp.where(
        previous_present > 0.5, previous_self[:, 4] * env_cfg.max_ships, planets.ships
    )
    source_ship_delta = (planets.ships - previous_source_ships) / env_cfg.max_ships
    outgoing_friendly = (
        _outgoing_friendly_fleet_ships(planets.id, fleets, player) / env_cfg.max_ships
    )
    incoming_friendly, incoming_enemy = _incoming_fleet_pressure(
        planets.x, planets.y, planets.radius, fleets, player
    )
    temporal_features = jnp.stack(
        [
            source_ship_delta,
            previous_present,
            previous_present,
            outgoing_friendly,
            incoming_friendly / env_cfg.max_ships,
            incoming_enemy / env_cfg.max_ships,
        ],
        axis=-1,
    )
    features = jnp.concatenate(
        [base_features, owner_context, temporal_features], axis=-1
    )
    return jnp.where(planets.active[:, None], features, jnp.zeros_like(features))


def _candidate_features(
    planets,
    fleets,
    player,
    env_cfg: EnvConfig,
    history: JaxFeatureHistory | None = None,
):
    p = env_cfg.max_planets
    c = env_cfg.candidate_count
    src_x = planets.x[:, None]
    src_y = planets.y[:, None]
    dx = planets.x[None, :] - src_x
    dy = planets.y[None, :] - src_y
    dist = jnp.sqrt(dx * dx + dy * dy)
    valid_target = planets.active[None, :] & (
        planets.id[None, :] != planets.id[:, None]
    )
    # JAX-friendly approximation of the Python encoder's sorted candidate list:
    # closest active non-self planets fill slots 1..N while slot 0 remains no-op.
    sort_distance = jnp.where(valid_target, dist, jnp.inf)
    sort_id = jnp.broadcast_to(planets.id[None, :], dist.shape)
    order = jnp.lexsort((sort_id, sort_distance), axis=1)[:, : max(0, c - 1)]
    pad_width = max(0, c - 1) - order.shape[1]
    if pad_width:
        order = jnp.pad(order, ((0, 0), (0, pad_width)), constant_values=0)
    ordered_valid = jnp.take_along_axis(valid_target, order, axis=1)

    tgt_x = jnp.take(planets.x, order, axis=0)
    tgt_y = jnp.take(planets.y, order, axis=0)
    tgt_owner = jnp.take(planets.owner, order, axis=0)
    tgt_ships = jnp.take(planets.ships, order, axis=0)
    tgt_prod = jnp.take(planets.production, order, axis=0)
    tgt_radius = jnp.take(planets.radius, order, axis=0)
    tgt_active = jnp.take(planets.active, order, axis=0)
    real_dx = tgt_x - src_x
    real_dy = tgt_y - src_y
    angle = jnp.arctan2(real_dy, real_dx)
    crosses = shot_crosses_sun_xy(
        src_x, src_y, planets.radius[:, None], angle, tgt_x, tgt_y
    )
    base_real_features = jnp.stack(
        [
            tgt_active.astype(jnp.float32),
            (tgt_owner == -1).astype(jnp.float32),
            (tgt_owner == player).astype(jnp.float32),
            ((tgt_owner != -1) & (tgt_owner != player)).astype(jnp.float32),
            tgt_x / env_cfg.board_size,
            tgt_y / env_cfg.board_size,
            real_dx / env_cfg.board_size,
            real_dy / env_cfg.board_size,
            jnp.sqrt(real_dx * real_dx + real_dy * real_dy) / env_cfg.board_size,
            jnp.minimum(tgt_ships, env_cfg.max_ships) / env_cfg.max_ships,
            tgt_prod / env_cfg.max_production,
            is_rotating_xy(tgt_x, tgt_y, tgt_radius).astype(jnp.float32),
            crosses.astype(jnp.float32),
            jnp.broadcast_to(
                jnp.minimum(planets.ships[:, None], env_cfg.max_ships)
                / env_cfg.max_ships,
                tgt_x.shape,
            ),
        ],
        axis=-1,
    )
    current_owner = target_owner_one_hot(tgt_owner, player, env_cfg)
    target_ids = jnp.take(planets.id, order, axis=0)
    previous_candidate, previous_present = _previous_candidate_features(
        planets.id, target_ids, history, env_cfg
    )
    incoming_friendly, incoming_enemy = _incoming_fleet_pressure(
        tgt_x, tgt_y, tgt_radius, fleets, player
    )
    previous_target_ships = jnp.where(
        previous_present > 0.5,
        previous_candidate[..., 9] * env_cfg.max_ships,
        tgt_ships,
    )
    ship_delta = (tgt_ships - previous_target_ships) / env_cfg.max_ships
    owner_changed = (
        (jnp.abs(current_owner - previous_candidate[..., 14:18]).sum(axis=-1) > 0.5)
        & (previous_present > 0.5)
    ).astype(jnp.float32)
    temporal_features = jnp.stack(
        [
            jnp.sqrt(real_dx * real_dx + real_dy * real_dy)
            / jnp.maximum(env_cfg.ship_speed, 1e-6)
            / env_cfg.episode_steps,
            incoming_friendly / env_cfg.max_ships,
            incoming_enemy / env_cfg.max_ships,
            ship_delta,
            owner_changed,
        ],
        axis=-1,
    )
    history_present = ordered_valid[..., None].astype(jnp.float32)
    real_features = jnp.concatenate(
        [
            base_real_features,
            current_owner,
            temporal_features,
            history_present,
        ],
        axis=-1,
    )
    real_features = jnp.where(ordered_valid[..., None], real_features, 0.0)
    noop_features = jnp.zeros((p, 1, BASE_CANDIDATE_FEATURE_DIM), dtype=jnp.float32)
    features = jnp.concatenate([noop_features, real_features], axis=1)
    noop_ids = jnp.full((p, 1), -1, dtype=jnp.int32)
    real_ids = jnp.where(ordered_valid, jnp.take(planets.id, order, axis=0), -1)
    ids = jnp.concatenate([noop_ids, real_ids], axis=1)
    angles = jnp.concatenate([jnp.zeros((p, 1), dtype=jnp.float32), angle], axis=1)
    noop_mask = (
        jnp.ones((p, 1), dtype=bool)
        if c > NO_OP_CANDIDATE_INDEX
        else jnp.zeros((p, 0), dtype=bool)
    )
    real_mask = ordered_valid & (~crosses)
    mask = jnp.concatenate([noop_mask, real_mask], axis=1)[:, :c]
    return ids[:, :c], features[:, :c, :], mask, angles[:, :c]


def _global_features(
    game, env_cfg: EnvConfig, history: JaxFeatureHistory | None = None
):
    planets = game.planets
    fleets = game.fleets
    player = game.player
    mine = planets.active & (planets.owner == player)
    enemy = planets.active & (planets.owner != -1) & (planets.owner != player)
    neutral = planets.active & (planets.owner == -1)
    my_fleet = fleets.active & (fleets.owner == player)
    enemy_fleet = fleets.active & (fleets.owner != player)
    denom = env_cfg.max_planets * env_cfg.max_ships
    owner_counts, owner_ships, owner_fleets, active_mask, player_count_feature = (
        owner_relative_summary(planets, fleets, player, env_cfg)
    )
    owner_production = owner_relative_production(planets, player, env_cfg)
    previous_global, previous_global_present = _previous_global_features(
        history, env_cfg
    )
    base_features = jnp.asarray(
        [
            game.step.astype(jnp.float32) / env_cfg.episode_steps,
            mine.astype(jnp.float32).sum() / env_cfg.max_planets,
            enemy.astype(jnp.float32).sum() / env_cfg.max_planets,
            neutral.astype(jnp.float32).sum() / env_cfg.max_planets,
            jnp.where(mine, planets.ships, 0.0).sum() / denom,
            jnp.where(enemy, planets.ships, 0.0).sum() / denom,
            jnp.where(my_fleet, fleets.ships, 0.0).sum() / denom,
            jnp.where(enemy_fleet, fleets.ships, 0.0).sum() / denom,
        ],
        dtype=jnp.float32,
    )
    return jnp.concatenate(
        [
            base_features,
            owner_counts,
            owner_ships,
            owner_fleets,
            active_mask,
            jnp.asarray([player_count_feature], dtype=jnp.float32),
            owner_production,
            (owner_ships - previous_global[12:16]) * previous_global_present,
            (owner_counts - previous_global[8:12]) * previous_global_present,
            (owner_fleets - previous_global[16:20]) * previous_global_present,
            (owner_production - previous_global[25:29]) * previous_global_present,
        ]
    )


def _previous_self_features(source_ids, history, env_cfg: EnvConfig):
    if history is None or feature_history_steps(env_cfg) <= 1:
        return (
            jnp.zeros((env_cfg.max_planets, BASE_SELF_FEATURE_DIM), dtype=jnp.float32),
            jnp.zeros((env_cfg.max_planets,), dtype=jnp.float32),
        )
    previous_ids = history.source_ids[-1]
    matches = previous_ids[None, :] == source_ids[:, None]
    selected = jnp.where(
        matches[..., None], history.self_features[-1][None, :, :], 0.0
    ).sum(axis=1)
    present = (selected[:, 0] > 0.5).astype(jnp.float32)
    return selected, present


def _previous_candidate_features(source_ids, target_ids, history, env_cfg: EnvConfig):
    if history is None or feature_history_steps(env_cfg) <= 1:
        return (
            jnp.zeros(
                (
                    env_cfg.max_planets,
                    max(0, env_cfg.candidate_count - 1),
                    BASE_CANDIDATE_FEATURE_DIM,
                ),
                dtype=jnp.float32,
            ),
            jnp.zeros(
                (env_cfg.max_planets, max(0, env_cfg.candidate_count - 1)),
                dtype=jnp.float32,
            ),
        )
    previous_source_matches = (
        history.source_ids[-1][None, None, :, None] == source_ids[:, None, None, None]
    )
    previous_target_matches = (
        history.candidate_ids[-1][None, None, :, :] == target_ids[:, :, None, None]
    )
    valid_targets = target_ids[:, :, None, None] != -1
    matches = previous_source_matches & previous_target_matches & valid_targets
    selected = jnp.where(
        matches[..., None], history.candidate_features[-1][None, None, :, :, :], 0.0
    ).sum(axis=(2, 3))
    present = (selected[..., 0] > 0.5).astype(jnp.float32)
    return selected, present


def _previous_global_features(history, env_cfg: EnvConfig):
    if history is None or feature_history_steps(env_cfg) <= 1:
        return (
            jnp.zeros((BASE_GLOBAL_FEATURE_DIM,), dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
    previous = history.global_features[-1]
    present = (jnp.abs(previous).sum() > 0.0).astype(jnp.float32)
    return previous, present


def _outgoing_friendly_fleet_ships(source_ids, fleets, player):
    matches = (
        (fleets.from_planet_id[None, :] == source_ids[:, None])
        & fleets.active[None, :]
        & (fleets.owner[None, :] == player)
    )
    return jnp.where(matches, fleets.ships[None, :], 0.0).sum(axis=1)


def _incoming_fleet_pressure(x, y, radius, fleets, player):
    target_x = jnp.asarray(x)[..., None]
    target_y = jnp.asarray(y)[..., None]
    target_radius = jnp.asarray(radius)[..., None]
    dx = target_x - fleets.x
    dy = target_y - fleets.y
    cos_a = jnp.cos(fleets.angle)
    sin_a = jnp.sin(fleets.angle)
    forward = dx * cos_a + dy * sin_a
    closest_x = fleets.x + cos_a * forward
    closest_y = fleets.y + sin_a * forward
    cross_track = jnp.sqrt((target_x - closest_x) ** 2 + (target_y - closest_y) ** 2)
    aims_at_target = fleets.active & (forward >= 0.0) & (cross_track <= target_radius)
    friendly = jnp.where(
        aims_at_target & (fleets.owner == player), fleets.ships, 0.0
    ).sum(axis=-1)
    enemy = jnp.where(aims_at_target & (fleets.owner != player), fleets.ships, 0.0).sum(
        axis=-1
    )
    return friendly, enemy


def owner_relative_production(planets, player, env_cfg: EnvConfig):
    player_count = clipped_player_count(env_cfg)
    slots = relative_owner_slots(planets.owner, player, player_count)
    valid_planets = (
        planets.active & (planets.owner >= 0) & (planets.owner < player_count)
    )
    production = jnp.bincount(
        slots,
        weights=jnp.where(valid_planets, planets.production, 0.0),
        length=MAX_OWNER_FEATURE_PLAYERS,
    )[:MAX_OWNER_FEATURE_PLAYERS]
    return production / (env_cfg.max_planets * env_cfg.max_production)


def owner_relative_summary(planets, fleets, player, env_cfg: EnvConfig):
    """Return fixed-size owner-relative summaries for up to four players."""

    player_count = clipped_player_count(env_cfg)
    denom = env_cfg.max_planets * env_cfg.max_ships

    planet_slots = relative_owner_slots(planets.owner, player, player_count)
    valid_planets = (
        planets.active & (planets.owner >= 0) & (planets.owner < player_count)
    )
    owner_counts = jnp.bincount(
        planet_slots,
        weights=valid_planets.astype(jnp.float32),
        length=MAX_OWNER_FEATURE_PLAYERS,
    )[:MAX_OWNER_FEATURE_PLAYERS]
    owner_ships = jnp.bincount(
        planet_slots,
        weights=jnp.where(valid_planets, planets.ships, 0.0),
        length=MAX_OWNER_FEATURE_PLAYERS,
    )[:MAX_OWNER_FEATURE_PLAYERS]

    fleet_slots = relative_owner_slots(fleets.owner, player, player_count)
    valid_fleets = fleets.active & (fleets.owner >= 0) & (fleets.owner < player_count)
    owner_fleets = jnp.bincount(
        fleet_slots,
        weights=jnp.where(valid_fleets, fleets.ships, 0.0),
        length=MAX_OWNER_FEATURE_PLAYERS,
    )[:MAX_OWNER_FEATURE_PLAYERS]
    active_mask = (
        jnp.arange(MAX_OWNER_FEATURE_PLAYERS, dtype=jnp.int32) < player_count
    ).astype(jnp.float32)
    return (
        owner_counts / env_cfg.max_planets,
        owner_ships / denom,
        owner_fleets / denom,
        active_mask,
        jnp.asarray(player_count / MAX_OWNER_FEATURE_PLAYERS, dtype=jnp.float32),
    )


def target_owner_one_hot(owner, player, env_cfg: EnvConfig):
    """Encode target owners relative to ``player`` with neutral as all-zero."""

    player_count = clipped_player_count(env_cfg)
    slots = relative_owner_slots(owner, player, player_count)
    valid_owner = (owner >= 0) & (owner < player_count)
    return jax.nn.one_hot(
        jnp.where(valid_owner, slots, 0),
        MAX_OWNER_FEATURE_PLAYERS,
        dtype=jnp.float32,
    ) * valid_owner[..., None].astype(jnp.float32)


def clipped_player_count(env_cfg: EnvConfig) -> int:
    return max(
        1, min(MAX_OWNER_FEATURE_PLAYERS, int(getattr(env_cfg, "player_count", 2)))
    )


def relative_owner_slots(owner, player, player_count: int):
    return (owner.astype(jnp.int32) - player.astype(jnp.int32)) % player_count


def is_rotating_xy(x, y, radius):
    """Return whether planets at ``(x, y)`` are inside the rotating orbit band."""

    dx = x - BOARD_CENTER[0]
    dy = y - BOARD_CENTER[1]
    return jnp.sqrt(dx * dx + dy * dy) + radius < ROTATION_RADIUS_LIMIT


def shot_crosses_sun_xy(src_x, src_y, src_radius, angle, tgt_x, tgt_y):
    """Return whether a launch ray from source to target intersects the sun."""

    start_x = src_x + jnp.cos(angle) * (src_radius + PLANET_LAUNCH_RADIUS_OFFSET)
    start_y = src_y + jnp.sin(angle) * (src_radius + PLANET_LAUNCH_RADIUS_OFFSET)
    return (
        point_to_segment_distance_xy(
            BOARD_CENTER[0], BOARD_CENTER[1], start_x, start_y, tgt_x, tgt_y
        )
        < SUN_RADIUS
    )


def point_to_segment_distance_xy(px, py, vx, vy, wx, wy):
    """Return the distance from point ``p`` to segment ``v``-``w``."""

    l2 = (vx - wx) ** 2 + (vy - wy) ** 2
    t = ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / jnp.maximum(l2, 1e-12)
    t = jnp.clip(t, 0.0, 1.0)
    proj_x = vx + t * (wx - vx)
    proj_y = vy + t * (wy - vy)
    return jnp.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
