"""JIT-safe home planet assignment for pool-gather reset (extracted from pick 4a)."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

import jax


class PlanetTables(NamedTuple):
    """Planet table layout shared by pool gather and home assignment."""

    id: jax.Array
    owner: jax.Array
    x: jax.Array
    y: jax.Array
    radius: jax.Array
    ships: jax.Array
    production: jax.Array
    active: jax.Array


def assign_home_planets(
    tables: PlanetTables,
    *,
    player_count: int,
    home_group: jax.Array,
) -> PlanetTables:
    """Assign home planets in-place using Kaggle home-group rules."""

    num_groups = jnp.maximum(tables.active.astype(jnp.int32).sum() // 4, 1)
    home_group = home_group % num_groups
    base = home_group * 4
    owner = tables.owner
    ships = tables.ships

    if player_count == 4:
        quadrants = jnp.arange(4, dtype=jnp.int32)

        def set_home(i, carry):
            o, s = carry
            slot = base + i
            is_home = tables.active[slot]
            o = o.at[slot].set(jnp.where(is_home, quadrants[i], o[slot]))
            s = s.at[slot].set(jnp.where(is_home, 10.0, s[slot]))
            return o, s

        owner, ships = jax.lax.fori_loop(0, 4, set_home, (owner, ships))
    else:
        slot0 = base
        slot3 = base + 3
        owner = owner.at[slot0].set(jnp.where(tables.active[slot0], 0, owner[slot0]))
        owner = owner.at[slot3].set(jnp.where(tables.active[slot3], 1, owner[slot3]))
        ships = ships.at[slot0].set(jnp.where(tables.active[slot0], 10.0, ships[slot0]))
        ships = ships.at[slot3].set(jnp.where(tables.active[slot3], 10.0, ships[slot3]))

    return tables._replace(owner=owner, ships=ships)


__all__ = ["PlanetTables", "assign_home_planets"]
