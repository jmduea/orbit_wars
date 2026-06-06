"""Baked comet schedule activation and stepping (no inline path generation)."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

import jax
from src.game.constants import (
    COMET_OFF_BOARD,
    COMET_PRODUCTION,
    COMET_RADIUS,
    COMET_SPAWN_STEPS,
    COMETS_PER_GROUP,
    MAX_COMET_GROUPS,
    MAX_COMET_PATH_LEN,
    MAX_PLANETS,
    TOTAL_COMETS,
)

_COMET_SPAWN_STEPS = jnp.array(COMET_SPAWN_STEPS, dtype=jnp.int32)
DEFAULT_COMET_SHIPS = 1.0


class JaxCometState(NamedTuple):
    """Fixed-shape comet groups with pre-baked paths from the map pool."""

    path_index: jax.Array
    planet_ids: jax.Array
    paths_x: jax.Array
    paths_y: jax.Array
    path_lengths: jax.Array
    group_active: jax.Array
    wave_ok: jax.Array


def empty_comet_state() -> JaxCometState:
    return JaxCometState(
        path_index=jnp.full((MAX_COMET_GROUPS,), -1, dtype=jnp.int32),
        planet_ids=jnp.full((MAX_COMET_GROUPS, COMETS_PER_GROUP), -1, dtype=jnp.int32),
        paths_x=jnp.zeros(
            (MAX_COMET_GROUPS, COMETS_PER_GROUP, MAX_COMET_PATH_LEN),
            dtype=jnp.float32,
        ),
        paths_y=jnp.zeros(
            (MAX_COMET_GROUPS, COMETS_PER_GROUP, MAX_COMET_PATH_LEN),
            dtype=jnp.float32,
        ),
        path_lengths=jnp.zeros((MAX_COMET_GROUPS, COMETS_PER_GROUP), dtype=jnp.int32),
        group_active=jnp.zeros((MAX_COMET_GROUPS,), dtype=bool),
        wave_ok=jnp.zeros((MAX_COMET_GROUPS,), dtype=bool),
    )


def comet_state_from_pool(
    *,
    planet_ids: jax.Array,
    paths_x: jax.Array,
    paths_y: jax.Array,
    path_lengths: jax.Array,
    wave_ok: jax.Array,
) -> JaxCometState:
    return JaxCometState(
        path_index=jnp.full((MAX_COMET_GROUPS,), -1, dtype=jnp.int32),
        planet_ids=planet_ids,
        paths_x=paths_x,
        paths_y=paths_y,
        path_lengths=path_lengths,
        group_active=jnp.zeros((MAX_COMET_GROUPS,), dtype=bool),
        wave_ok=wave_ok,
    )


def _active_comet_planet_ids(comets: JaxCometState) -> jax.Array:
    valid = comets.group_active[:, None] & (comets.planet_ids >= 0)
    return jnp.where(valid, comets.planet_ids, -1).reshape(
        (MAX_COMET_GROUPS * COMETS_PER_GROUP,)
    )


def is_comet_planet(comets: JaxCometState, planet_ids: jax.Array) -> jax.Array:
    return jnp.isin(planet_ids, _active_comet_planet_ids(comets))


def _deactivate_planets_by_id(planets, remove_ids: jax.Array):
    remove = jnp.isin(planets.id, remove_ids) & (remove_ids >= 0)
    return planets._replace(
        active=planets.active & (~remove),
        owner=jnp.where(remove, -1, planets.owner),
    )


def expire_comets_pre_launch(planets, initial_planets, comets: JaxCometState):
    def group_body(g, carry):
        planets, initial, comets = carry
        active = comets.group_active[g]
        idx = comets.path_index[g]

        def comet_body(i, inner):
            planets, initial, comets = inner
            pid = comets.planet_ids[g, i]
            path_len = comets.path_lengths[g, i]
            expire = active & (pid >= 0) & (idx >= path_len)
            planets = _deactivate_planets_by_id(
                planets, jnp.where(expire, pid, jnp.array(-1, dtype=jnp.int32))
            )
            initial = _deactivate_planets_by_id(
                initial, jnp.where(expire, pid, jnp.array(-1, dtype=jnp.int32))
            )
            comets = comets._replace(
                planet_ids=comets.planet_ids.at[g, i].set(jnp.where(expire, -1, pid))
            )
            return planets, initial, comets

        planets, initial, comets = jax.lax.fori_loop(
            0, COMETS_PER_GROUP, comet_body, (planets, initial, comets)
        )
        has_ids = (comets.planet_ids[g] >= 0).any()
        comets = comets._replace(
            group_active=comets.group_active.at[g].set(active & has_ids)
        )
        return planets, initial, comets

    return jax.lax.fori_loop(
        0, MAX_COMET_GROUPS, group_body, (planets, initial_planets, comets)
    )


def _spawn_group_index(spawn_step: jax.Array) -> jax.Array:
    return jnp.argmax((_COMET_SPAWN_STEPS == spawn_step).astype(jnp.int32))


def activate_baked_comet_group(
    planets, initial_planets, comets: JaxCometState, spawn_step: jax.Array
):
    g = _spawn_group_index(spawn_step)
    matched = _COMET_SPAWN_STEPS[g] == spawn_step
    can_activate = matched & comets.wave_ok[g] & (~comets.group_active[g])

    def activate(_):
        base_slot = MAX_PLANETS - TOTAL_COMETS + g * COMETS_PER_GROUP

        def place_comet(i, inner):
            p, initial, comets_local = inner
            slot = base_slot + i
            pid = comets_local.planet_ids[g, i]
            p = p._replace(
                id=p.id.at[slot].set(pid),
                owner=p.owner.at[slot].set(-1),
                x=p.x.at[slot].set(COMET_OFF_BOARD),
                y=p.y.at[slot].set(COMET_OFF_BOARD),
                radius=p.radius.at[slot].set(COMET_RADIUS),
                ships=p.ships.at[slot].set(DEFAULT_COMET_SHIPS),
                production=p.production.at[slot].set(COMET_PRODUCTION),
                active=p.active.at[slot].set(True),
            )
            initial = initial._replace(
                id=initial.id.at[slot].set(pid),
                owner=initial.owner.at[slot].set(-1),
                x=initial.x.at[slot].set(COMET_OFF_BOARD),
                y=initial.y.at[slot].set(COMET_OFF_BOARD),
                radius=initial.radius.at[slot].set(COMET_RADIUS),
                ships=initial.ships.at[slot].set(DEFAULT_COMET_SHIPS),
                production=initial.production.at[slot].set(COMET_PRODUCTION),
                active=initial.active.at[slot].set(True),
            )
            return p, initial, comets_local

        planets_out, initial_out, comets_out = jax.lax.fori_loop(
            0, COMETS_PER_GROUP, place_comet, (planets, initial_planets, comets)
        )
        comets_out = comets_out._replace(
            path_index=comets_out.path_index.at[g].set(-1),
            group_active=comets_out.group_active.at[g].set(True),
        )
        return planets_out, initial_out, comets_out

    return jax.lax.cond(
        can_activate,
        activate,
        lambda _: (planets, initial_planets, comets),
        None,
    )


def advance_comet_positions(
    comets: JaxCometState,
    planets,
    new_px: jax.Array,
    new_py: jax.Array,
):
    def group_body(g, carry):
        new_px, new_py, comets = carry
        active = comets.group_active[g]
        idx = comets.path_index[g] + jnp.where(active, 1, 0)
        comets = comets._replace(path_index=comets.path_index.at[g].set(idx))

        def comet_body(i, inner):
            new_px, new_py, comets = inner
            pid = comets.planet_ids[g, i]
            path_len = comets.path_lengths[g, i]
            on_group = active & (pid >= 0)
            match = (planets.id == pid) & planets.active
            slot = jnp.argmax(match.astype(jnp.int32))
            in_path = on_group & (idx < path_len) & match.any()
            safe_idx = jnp.clip(idx, 0, jnp.maximum(path_len - 1, 0))
            cx = comets.paths_x[g, i, safe_idx]
            cy = comets.paths_y[g, i, safe_idx]
            new_px = jnp.where(in_path, new_px.at[slot].set(cx), new_px)
            new_py = jnp.where(in_path, new_py.at[slot].set(cy), new_py)
            return new_px, new_py, comets

        return jax.lax.fori_loop(
            0, COMETS_PER_GROUP, comet_body, (new_px, new_py, comets)
        )

    return jax.lax.fori_loop(0, MAX_COMET_GROUPS, group_body, (new_px, new_py, comets))


__all__ = [
    "JaxCometState",
    "activate_baked_comet_group",
    "advance_comet_positions",
    "comet_state_from_pool",
    "empty_comet_state",
    "expire_comets_pre_launch",
    "is_comet_planet",
]
