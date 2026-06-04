"""Final-score win metrics for SSOT tournament qualifiers (R18)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import jax


def learner_won_from_final_scores(
    scores: np.ndarray,
    learner_player: int,
) -> bool:
    """True when learner has strictly highest final score (planet + fleet ships).

    Ties at the maximum do not count as wins (R19 interim conservative rule).
    """

    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return False
    learner = int(learner_player)
    if learner < 0 or learner >= arr.size:
        return False
    learner_score = float(arr[learner])
    if learner_score <= 0.0:
        return False
    best = float(np.max(arr))
    if learner_score < best:
        return False
    winners = np.flatnonzero(arr == best)
    return winners.size == 1 and int(winners[0]) == learner


def win_fraction(wins: int, games: int) -> float | None:
    if games <= 0:
        return None
    return float(wins) / float(games)


def final_ship_scores(game, player_count: int) -> np.ndarray:
    """Per-player ship totals (planets + fleets) for terminal win scoring (R18)."""

    owners = jnp.arange(int(player_count), dtype=jnp.int32)

    def score_for(owner: jax.Array) -> jax.Array:
        planet_ships = jnp.where(
            (game.planets.owner == owner) & game.planets.active,
            game.planets.ships,
            0.0,
        ).sum()
        fleet_ships = jnp.where(
            (game.fleets.owner == owner) & game.fleets.active,
            game.fleets.ships,
            0.0,
        ).sum()
        return planet_ships + fleet_ships

    scores = jax.vmap(score_for)(owners)
    return np.asarray(jax.device_get(scores), dtype=np.float64)
