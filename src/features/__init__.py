from importlib import import_module

_ENCODING_EXPORTS = {
    "BASE_CANDIDATE_FEATURE_DIM",
    "BASE_GLOBAL_FEATURE_DIM",
    "BASE_SELF_FEATURE_DIM",
    "DecisionContext",
    "FeatureHistoryBuffer",
    "FeatureSnapshot",
    "NO_OP_CANDIDATE_INDEX",
    "TurnBatch",
    "build_candidate_features",
    "build_candidates",
    "build_feature_snapshot",
    "build_global_features",
    "build_self_features",
    "candidate_feature_dim",
    "clipped_player_count",
    "distance",
    "encode_turn",
    "fleet_aims_at_planet",
    "global_feature_dim",
    "incoming_fleet_pressure",
    "is_rotating_planet",
    "owner_relative_production",
    "owner_relative_summary",
    "point_to_segment_distance",
    "real_candidate_slots",
    "relative_owner_slot",
    "self_feature_dim",
    "ship_bucket_fraction",
    "ship_count_for_bucket",
    "shot_crosses_sun",
    "target_owner_one_hot",
    "total_ships",
}
_NORMALIZATION_EXPORTS = {"ObservationNormalizer"}
_REGISTRY_EXPORTS = {
    "CANDIDATE_FEATURE_SCHEMA",
    "FeatureGroupRegistry",
    "FeatureItem",
    "GLOBAL_FEATURE_SCHEMA",
    "SELF_FEATURE_SCHEMA",
    "candidate_feature_dim",
    "candidate_feature_schema",
    "feature_history_steps",
    "global_feature_dim",
    "global_feature_schema",
    "self_feature_dim",
    "self_feature_schema",
}

__all__ = sorted(_ENCODING_EXPORTS | _NORMALIZATION_EXPORTS | _REGISTRY_EXPORTS)


def __getattr__(name: str):
    if name in _NORMALIZATION_EXPORTS:
        module = import_module(".normalization", __name__)

        return getattr(module, name)
    if name in _REGISTRY_EXPORTS and name not in _ENCODING_EXPORTS:
        registry = import_module(".registry", __name__)

        return getattr(registry, name)
    if name in _ENCODING_EXPORTS:
        encoding = import_module(".encoding", __name__)

        return getattr(encoding, name)
    if name in _REGISTRY_EXPORTS:
        registry = import_module(".registry", __name__)

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
