from __future__ import annotations

import math
from collections import namedtuple
from dataclasses import replace
from typing import Any, Protocol

from src.config import TrainConfig
from src.game.shield import (
    is_trajectory_safe_for_launch,
)
from src.game.types import parse_observation

Planet = namedtuple(
    "Planet", ["id", "owner", "x", "y", "radius", "ships", "production"]
)
# Tournament/replay baselines run in Kaggle env; do not apply learner shield filtering
# (cheap shield can reject every sniper angle on some seeds, making baselines noop).
BASELINE_RUNTIME_ENV = replace(TrainConfig().task, trajectory_shield_mode="off")


class OpponentPolicy(Protocol):
    def act(
        self, observation: Any, configuration: Any = None
    ) -> list[list[float | int]]: ...


class SniperOpponent:
    def act(
        self, observation: Any, configuration: Any = None
    ) -> list[list[float | int]]:
        del configuration
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
                BASELINE_RUNTIME_ENV,
            ):
                continue
            moves.append([source.id, angle, ships_needed])
        return moves


class KaggleRandomOpponent:
    def __init__(self) -> None:
        from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent

        self._agent = random_agent

    def act(
        self, observation: Any, configuration: Any = None
    ) -> list[list[float | int]]:
        del configuration
        # Match Kaggle reference random_agent: no trajectory shield (shield rejects
        # almost all random angles and makes this baseline identical to noop).
        return list(self._agent(observation))


class NoopOpponent:
    """Pass/no-launch baseline (matches JAX ``opponents.dispatch=noop``)."""

    def act(
        self, observation: Any, configuration: Any = None
    ) -> list[list[float | int]]:
        del observation, configuration
        return []


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
    if name in {"noop", "noop_only"}:
        return NoopOpponent()
    raise ValueError(f"Unknown opponent: {name}")


def obs_get(observation: Any, key: str, default: Any) -> Any:
    if isinstance(observation, dict):
        return observation.get(key, default)
    return getattr(observation, key, default)
