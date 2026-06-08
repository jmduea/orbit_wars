"""Compose tests for opponent ablation ladder rungs (ce-optimize pre-loop)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from scripts.ce_optimize.opponent_ladder_rungs import LADDER_RUNG_OVERRIDES
from src.config import compose_hydra_train_config
from src.opponents.constants import (
    OPPONENT_FAMILY_NAMES,
    OPPONENT_HISTORICAL,
    OPPONENT_LATEST,
    OPPONENT_RANDOM,
    is_noop_jax_training_opponent_mode,
    normalize_jax_training_opponent_mode,
)
from src.training.curriculum import CurriculumController


def _stage_view_at_update(cfg, *, update: int = 5):
    controller = CurriculumController(cfg.curriculum, cfg.opponents.snapshot)
    pool_size = max(int(cfg.opponents.snapshot.pool_size), 1)
    return controller.stage_view(
        update,
        snapshot_ids=jnp.arange(pool_size, dtype=jnp.int32),
        snapshot_valid_mask=jnp.array([True] + [False] * (pool_size - 1)),
        snapshot_updates=jnp.zeros((pool_size,), dtype=jnp.int32),
    )


@pytest.mark.parametrize("rung", list(LADDER_RUNG_OVERRIDES))
def test_opponent_ladder_rung_composes(rung: str) -> None:
    cfg = compose_hydra_train_config(LADDER_RUNG_OVERRIDES[rung])
    assert cfg.training.num_envs > 0


def test_noop_rung_uses_jax_noop_mode() -> None:
    cfg = compose_hydra_train_config(LADDER_RUNG_OVERRIDES["noop"])
    assert is_noop_jax_training_opponent_mode(cfg.opponents.dispatch)
    assert cfg.curriculum.enabled is False
    assert cfg.opponents.self_play.enabled is False
    assert cfg.opponents.snapshot.pool_size == 0


def test_recovery_rung_uses_direct_random_mode_without_snapshot_pool() -> None:
    cfg = compose_hydra_train_config(LADDER_RUNG_OVERRIDES["recovery"])
    assert normalize_jax_training_opponent_mode(cfg.opponents.dispatch) == "random"
    assert cfg.curriculum.enabled is False
    assert cfg.opponents.self_play.enabled is False
    assert cfg.opponents.snapshot.pool_size == 0
    assert cfg.opponents.snapshot.interval_updates == 0


def test_scripted_heavy_stage_view_excludes_latest_and_historical() -> None:
    cfg = compose_hydra_train_config(LADDER_RUNG_OVERRIDES["scripted_heavy"])
    view = _stage_view_at_update(cfg)
    probs = {
        name: float(view.family_probs[i])
        for i, name in enumerate(OPPONENT_FAMILY_NAMES)
    }
    assert probs["latest"] == pytest.approx(0.0)
    assert probs["historical"] == pytest.approx(0.0)
    assert probs["random"] == pytest.approx(0.25)
    assert probs["nearest_sniper"] == pytest.approx(0.25)
    assert probs["turtle"] == pytest.approx(0.25)
    assert probs["opportunistic"] == pytest.approx(0.25)


def test_self_play_rung_latest_only() -> None:
    cfg = compose_hydra_train_config(LADDER_RUNG_OVERRIDES["self_play"])
    assert cfg.opponents.self_play.enabled is True
    view = _stage_view_at_update(cfg)
    assert float(view.family_probs[OPPONENT_LATEST]) == pytest.approx(1.0)
    assert float(view.family_probs[OPPONENT_HISTORICAL]) == pytest.approx(0.0)


def test_production_mix_rung_has_latest_weight_and_snapshot_pool() -> None:
    cfg = compose_hydra_train_config(LADDER_RUNG_OVERRIDES["production_mix"])
    assert cfg.opponents.snapshot.pool_size > 0
    assert cfg.curriculum.enabled is True
    view = _stage_view_at_update(cfg)
    assert float(view.family_probs[OPPONENT_LATEST]) > 0.0


def test_ladder_stage_views_differ_across_rungs() -> None:
    views = {
        rung: _stage_view_at_update(compose_hydra_train_config(overrides))
        for rung, overrides in LADDER_RUNG_OVERRIDES.items()
        if rung not in {"noop", "recovery"}
    }
    scripted = views["scripted_heavy"].family_probs
    self_play = views["self_play"].family_probs
    assert not jnp.allclose(scripted, self_play)
    assert float(scripted[OPPONENT_RANDOM]) > float(self_play[OPPONENT_RANDOM])
    assert float(self_play[OPPONENT_LATEST]) > float(scripted[OPPONENT_LATEST])
