from .metric_registry import *

__all__ = [
    name
    for name in globals()
    if not name.startswith("_") and name not in {"logger", "metric_registry"}
]
__all__ += ["TelemetryLogger", "build_telemetry"]


def __getattr__(name: str):
    if name in {"TelemetryLogger", "build_telemetry"}:
        from .logger import TelemetryLogger, build_telemetry

        return {
            "TelemetryLogger": TelemetryLogger,
            "build_telemetry": build_telemetry,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
