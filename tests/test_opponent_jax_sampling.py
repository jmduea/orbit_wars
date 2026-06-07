"""Unit tests for opponent JAX sampling helpers (jax tier, not slow)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from src.opponents.constants import (
    OPPONENT_HISTORICAL,
    OPPONENT_LATEST,
    OPPONENT_NOOP,
    OPPONENT_RANDOM,
)
from src.jax.env import JaxAction
from src.opponents.jax_actions.sampling import (
    OPPONENT_SLOT_COUNT_KEYS,
    _gather_action_by_env,
    _maybe_effective_single_family_id,
    _opponent_count_metrics,
    _select_env_action,
    _single_stage_family_id,
)
from src.training.curriculum import StageView


def _noop_action(env_count: int, *, source_fill: int = 0) -> JaxAction:
    fleets = 2
    return JaxAction(
        source_id=jnp.full((env_count, fleets), source_fill, dtype=jnp.int32),
        angle=jnp.zeros((env_count, fleets), dtype=jnp.float32),
        ships=jnp.zeros((env_count, fleets), dtype=jnp.int32),
        valid=jnp.zeros((env_count, fleets), dtype=bool),
    )


@pytest.mark.jax
def test_select_env_action_picks_per_env_branch() -> None:
    condition = jnp.array([True, False])
    true_action = _noop_action(2)
    false_action = _noop_action(2, source_fill=1)
    merged = _select_env_action(condition, true_action, false_action)
    assert int(merged.source_id[0, 0]) == 0
    assert int(merged.source_id[1, 0]) == 1


@pytest.mark.jax
def test_gather_action_by_env_indexes_pool_rows() -> None:
    pool = _noop_action(3)
    pool = JaxAction(
        source_id=jnp.arange(6, dtype=jnp.int32).reshape(3, 2),
        angle=pool.angle,
        ships=pool.ships,
        valid=pool.valid,
    )
    indices = jnp.array([2, 0], dtype=jnp.int32)
    gathered = _gather_action_by_env(pool, indices)
    assert gathered.source_id.shape == (2,)
    assert int(gathered.source_id[0]) == 4
    assert int(gathered.source_id[1]) == 1


@pytest.mark.jax
def test_opponent_count_metrics_excludes_learner_slot() -> None:
    effective = jnp.array(
        [
            [OPPONENT_LATEST, OPPONENT_NOOP],
            [OPPONENT_RANDOM, OPPONENT_LATEST],
        ],
        dtype=jnp.int32,
    )
    learner_player = jnp.array([0, 1], dtype=jnp.int32)
    metrics = _opponent_count_metrics(effective, learner_player)
    assert set(metrics) == set(OPPONENT_SLOT_COUNT_KEYS)
    assert float(metrics["opponent_slots_total"]) == 2.0
    assert float(metrics["opponent_slots_noop"]) == 1.0
    assert float(metrics["opponent_slots_random"]) == 1.0
    assert float(metrics["opponent_slots_latest"]) == 0.0


def _stage_view(
    *,
    family_ids: list[int],
    family_probs: list[float],
    snapshot_valid: list[bool] | None = None,
    fallback: int = OPPONENT_RANDOM,
) -> StageView:
    pool = 2
    valid = snapshot_valid if snapshot_valid is not None else [False, False]
    return StageView(
        stage_index=jnp.asarray(0, dtype=jnp.int32),
        family_ids=jnp.asarray(family_ids, dtype=jnp.int32),
        family_probs=jnp.asarray(family_probs, dtype=jnp.float32),
        family_mask=jnp.asarray(family_probs, dtype=jnp.float32) > 0.0,
        snapshot_pool_ids=jnp.zeros((pool,), dtype=jnp.int32),
        snapshot_valid_mask=jnp.asarray(valid, dtype=bool),
        snapshot_age_updates=jnp.zeros((pool,), dtype=jnp.int32),
        historical_selection_probs=jnp.ones((pool,), dtype=jnp.float32) / pool,
        fallback_family_id=jnp.asarray(fallback, dtype=jnp.int32),
    )


@pytest.mark.jax
def test_single_stage_family_id_detects_pure_mixture() -> None:
    view = _stage_view(
        family_ids=[OPPONENT_NOOP, OPPONENT_RANDOM],
        family_probs=[1.0, 0.0],
    )
    assert int(_single_stage_family_id(view)) == OPPONENT_NOOP

    mixed = _stage_view(
        family_ids=[OPPONENT_NOOP, OPPONENT_RANDOM],
        family_probs=[0.5, 0.5],
    )
    assert int(_single_stage_family_id(mixed)) == -1


@pytest.mark.jax
def test_maybe_effective_single_family_id_falls_back_without_snapshots() -> None:
    view = _stage_view(
        family_ids=[OPPONENT_HISTORICAL, OPPONENT_RANDOM],
        family_probs=[1.0, 0.0],
        snapshot_valid=[False, False],
        fallback=OPPONENT_NOOP,
    )
    effective = _maybe_effective_single_family_id(
        jnp.asarray(OPPONENT_HISTORICAL, dtype=jnp.int32),
        view,
    )
    assert int(effective) == OPPONENT_NOOP

    with_snapshot = _stage_view(
        family_ids=[OPPONENT_HISTORICAL, OPPONENT_RANDOM],
        family_probs=[1.0, 0.0],
        snapshot_valid=[True, False],
        fallback=OPPONENT_NOOP,
    )
    kept = _maybe_effective_single_family_id(
        jnp.asarray(OPPONENT_HISTORICAL, dtype=jnp.int32),
        with_snapshot,
    )
    assert int(kept) == OPPONENT_HISTORICAL
