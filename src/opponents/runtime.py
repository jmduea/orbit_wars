from __future__ import annotations

import math
from collections import namedtuple
from typing import Any, Protocol

from src.config import TrainConfig
from src.game.shield import (
    filter_moves_with_trajectory_shield,
    is_trajectory_safe_for_launch,
)
from src.game.types import parse_observation

Planet = namedtuple(
    "Planet", ["id", "owner", "x", "y", "radius", "ships", "production"]
)
DEFAULT_RUNTIME_ENV = TrainConfig().task


class OpponentPolicy(Protocol):
    def act(self, observation: Any) -> list[list[float | int]]: ...


class SniperOpponent:
    def act(self, observation: Any) -> list[list[float | int]]:
        moves: list[list[float | int]] = []
        state = parse_observation(observation)
        player = obs_get(observation, "player", 0)
        raw_planets = obs_get(observation, "planets", [])
        planets = [Planet(*planet) for planet in raw_planets]
        my_planets = [planet for planet in planets if planet.owner == player]
        targets = [planet for planet in planets if planet.owner != player]
        if not targets:
            return moves
        for source in my_planets:
            nearest = min(
                targets,
                key=lambda target: math.hypot(source.x - target.x, source.y - target.y),
            )
            ships_needed = max(nearest.ships + 1, 20)
            if source.ships < ships_needed:
                continue
            angle = math.atan2(nearest.y - source.y, nearest.x - source.x)
            if not is_trajectory_safe_for_launch(
                state,
                int(source.id),
                int(nearest.id),
                angle,
                ships_needed,
                DEFAULT_RUNTIME_ENV,
            ):
                continue
            moves.append([source.id, angle, ships_needed])
        return moves


class KaggleRandomOpponent:
    def __init__(self) -> None:
        from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent

        self._agent = random_agent

    def act(self, observation: Any) -> list[list[float | int]]:
        payload = {
            "player": obs_get(observation, "player", 0),
            "planets": list(obs_get(observation, "planets", [])),
        }
        state = parse_observation(observation)
        return filter_moves_with_trajectory_shield(
            list(self._agent(payload)), state, DEFAULT_RUNTIME_ENV
        )


def build_opponent(
    name: str,
    cfg: TrainConfig | None = None,
    device: str | None = None,
) -> OpponentPolicy:
    del cfg, device
    if name == "sniper":
        return SniperOpponent()
    if name == "random":
        return KaggleRandomOpponent()
    raise ValueError(f"Unknown opponent: {name}")


def obs_get(observation: Any, key: str, default: Any) -> Any:
    if isinstance(observation, dict):
        return observation.get(key, default)
    return getattr(observation, key, default)
