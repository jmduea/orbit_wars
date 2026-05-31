from importlib import import_module

_REGISTRY_EXPORTS = {
    "FeatureGroupRegistry",
    "FeatureItem",
    "GLOBAL_FEATURE_SCHEMA",
    "PLANET_FEATURE_SCHEMA",
    "EDGE_FEATURE_SCHEMA",
    "edge_feature_dim",
    "edge_k",
    "feature_history_steps",
    "global_feature_dim",
    "global_feature_schema",
    "planet_feature_dim",
    "planet_feature_schema",
}

__all__ = sorted(_REGISTRY_EXPORTS)


def __getattr__(name: str):
    if name in _REGISTRY_EXPORTS:
        registry = import_module(".registry", __name__)
        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
