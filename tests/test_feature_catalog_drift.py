"""Drift guard: catalog slices must match encode_turn outputs."""

from __future__ import annotations

import jax
import numpy as np

from src.config import TaskConfig
from src.features.catalog import (
    EDGE_FEATURE_CATALOG,
    GLOBAL_FEATURE_CATALOG,
    PLANET_FEATURE_CATALOG,
)
from src.features.registry import (
    EDGE_FEATURE_SCHEMA,
    GLOBAL_FEATURE_SCHEMA,
    PLANET_FEATURE_SCHEMA,
)
from src.jax.env import reset
from src.jax.features import encode_turn


def _cfg(**kwargs) -> TaskConfig:
    base = dict(
        max_fleets=32,
        candidate_count=4,
        player_count=2,
        feature_history_steps=1,
        ship_feature_scale=1000.0,
    )
    base.update(kwargs)
    return TaskConfig(**base)


def test_encode_slices_match_planet_catalog_names() -> None:
    cfg = _cfg()
    state, _ = reset(jax.random.PRNGKey(11), cfg)
    batch = encode_turn(state.game, cfg)

    for entry in PLANET_FEATURE_CATALOG.entries:
        if not entry.definition.active:
            continue
        name = entry.definition.name
        encoded = batch.planet_features[..., PLANET_FEATURE_SCHEMA.base_slice(name)]
        assert encoded.shape[-1] == entry.definition.size


def test_encode_slices_match_edge_catalog_names() -> None:
    cfg = _cfg()
    state, _ = reset(jax.random.PRNGKey(21), cfg)
    batch = encode_turn(state.game, cfg)

    for entry in EDGE_FEATURE_CATALOG.entries:
        if not entry.definition.active:
            continue
        name = entry.definition.name
        encoded = batch.edge_features[..., EDGE_FEATURE_SCHEMA.base_slice(name)]
        assert encoded.shape[-1] == entry.definition.size


def test_encode_slices_match_global_catalog_names() -> None:
    cfg = _cfg()
    state, _ = reset(jax.random.PRNGKey(31), cfg)
    batch = encode_turn(state.game, cfg)
    frame = batch.global_features[..., GLOBAL_FEATURE_SCHEMA.frame_slice()]

    for entry in GLOBAL_FEATURE_CATALOG.entries:
        if not entry.definition.active:
            continue
        name = entry.definition.name
        encoded = frame[GLOBAL_FEATURE_SCHEMA.base_slice(name)]
        assert encoded.shape[-1] == entry.definition.size
