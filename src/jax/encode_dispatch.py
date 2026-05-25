"""Version dispatch helpers for v1/v2 feature encoding at env boundaries."""

from __future__ import annotations

from src.config.schema import TaskConfig
from src.jax.features import (
    JaxFeatureHistory,
    JaxTurnBatch,
    append_feature_history,
    empty_feature_history,
    encode_turn,
)
from src.jax.features_v2 import (
    JaxFeatureHistoryV2,
    JaxTurnBatchV2,
    append_feature_history_v2,
    empty_feature_history_v2,
    encode_turn_v2,
)


def encoding_version_v2(cfg: TaskConfig) -> bool:
    return getattr(cfg, "encoding_version", "v1").strip().lower() == "v2"


def empty_feature_history_dispatch(
    cfg: TaskConfig,
) -> JaxFeatureHistory | JaxFeatureHistoryV2:
    if encoding_version_v2(cfg):
        return empty_feature_history_v2(cfg)
    return empty_feature_history(cfg)


def encode_turn_dispatch(
    game,
    cfg: TaskConfig,
    history: JaxFeatureHistory | JaxFeatureHistoryV2 | None = None,
) -> JaxTurnBatch | JaxTurnBatchV2:
    if encoding_version_v2(cfg):
        return encode_turn_v2(game, cfg, history)
    return encode_turn(game, cfg, history)


def append_feature_history_dispatch(
    history: JaxFeatureHistory | JaxFeatureHistoryV2 | None,
    game,
    cfg: TaskConfig,
) -> JaxFeatureHistory | JaxFeatureHistoryV2:
    if encoding_version_v2(cfg):
        return append_feature_history_v2(history, game, cfg)
    return append_feature_history(history, game, cfg)
