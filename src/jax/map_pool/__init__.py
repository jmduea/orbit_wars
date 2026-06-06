"""Offline map pool bake/load and JIT-safe home assignment."""

from .comets import JaxCometState, empty_comet_state
from .home_assignment import PlanetTables, assign_home_planets

__all__ = [
    "JaxCometState",
    "MapPoolConstants",
    "PlanetTables",
    "assign_home_planets",
    "empty_comet_state",
    "load_map_pool",
]


def __getattr__(name: str):
    if name in {"MapPoolConstants", "load_map_pool"}:
        from .load import MapPoolConstants, load_map_pool

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
