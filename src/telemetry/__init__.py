from .metric_registry import *

__all__ = [
    name
    for name in globals()
    if not name.startswith("_") and name not in {"logger", "metric_registry"}
]
__all__ += ["TelemetryLogger"]


def __getattr__(name: str):
    if name == "TelemetryLogger":
        from .logger import TelemetryLogger

        return TelemetryLogger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
