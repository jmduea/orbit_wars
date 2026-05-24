from .constants import *
from .types import GameState, PlanetState, parse_observation

__all__ = [
    "GameState",
    "OrbitEnv",
    "PlanetState",
    "parse_observation",
]


def __getattr__(name: str):
    if name == "OrbitEnv":
        from .env import OrbitEnv

        return OrbitEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
