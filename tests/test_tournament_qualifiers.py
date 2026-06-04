"""SSOT JAX tournament qualifier promotion and tick hooks."""

from __future__ import annotations

import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from src.artifacts.tournament.bracket.state import load_bracket_state
from src.config.schema import ArtifactsConfig, OutputConfig, SsotPipelineConfig, TrainConfig
from src.game.constants import MAX_PLANETS
from src.jax.env import JaxFleetState, JaxGameState, JaxPlanetState, empty_comet_state
from src.jax.qualifier_calibration import load_qualifier_calibration
from src.jax.tournament_qualifiers.eval import held_out_eval_seeds
from src.jax.tournament_qualifiers.metrics import (
    final_ship_scores,
    learner_won_from_final_scores,
)
from src.jax.tournament_qualifiers.promotion import (
    evaluate_stage_promotion,
    opponent_family_probs_for_stage,
    ssot_rollout_stage_view,
)
from src.jax.tournament_qualifiers.runner import (
    evaluate_qualifier_legs,
    ssot_qualifier_tick,
)


def test_learner_won_requires_strict_max_not_tie() -> None:
    scores = np.array([10.0, 10.0, 5.0], dtype=np.float64)
    assert not learner_won_from_final_scores(scores, learner_player=0)
    assert learner_won_from_final_scores(np.array([12.0, 10.0]), learner_player=0)


def test_stage_promotion_uses_final_score_win_rates_not_rollout_metric() -> None:
    verdict = evaluate_stage_promotion(
        stage=1,
        leg_wins={"random": (6, 10)},
    )
    assert verdict.promoted
    assert verdict.next_stage == 2
    assert verdict.leg_summaries[0].opponent == "random"


def test_missing_games_blocks_promotion() -> None:
    verdict = evaluate_stage_promotion(stage=2, leg_wins={"noop": (0, 0), "random": (0, 0)})
    assert not verdict.promoted
    assert verdict.fail_reason is not None


def test_calibration_loader_reads_committed_json() -> None:
    cal = load_qualifier_calibration()
    assert cal.min_win_rate_for(1, "random") == 0.55


def _ssot_cfg(tmp_path: Path, *, budget: int = 500) -> TrainConfig:
    cfg = TrainConfig()
    cfg.output = OutputConfig(root=str(tmp_path), campaign="ssot_demo")
    cfg.artifacts = ArtifactsConfig(
        ssot_pipeline=SsotPipelineConfig(
            enabled=True,
            qualifier_max_env_steps=budget,
            qualifier_eval_interval_updates=10,
        )
    )
    return cfg


def test_ssot_weak_config_at_env_step_budget(tmp_path: Path) -> None:
    cfg = _ssot_cfg(tmp_path, budget=1000)
    tick = ssot_qualifier_tick(
        cfg,
        update=50,
        total_env_steps=1000,
        checkpoint_path=None,
        output_root=tmp_path,
    )
    assert tick.weak_config is True
    state = load_bracket_state(
        tmp_path / "campaigns" / "ssot_demo" / "bracket" / "state.json"
    )
    assert state.phase == "weak_config"


def test_qualifier_eval_interval_zero_disables_tick_eval(tmp_path: Path) -> None:
    cfg = _ssot_cfg(tmp_path, budget=10_000_000)
    cfg.artifacts.ssot_pipeline.qualifier_eval_interval_updates = 0
    ckpt = tmp_path / "ckpt.pkl"
    ckpt.write_bytes(pickle.dumps({"update": 10}))
    tick = ssot_qualifier_tick(
        cfg,
        update=10,
        total_env_steps=100,
        checkpoint_path=ckpt,
        output_root=tmp_path,
    )
    assert tick.qualifier_stage == 1
    assert not any(e.get("event") == "ssot_qualifier_eval" for e in tick.events)


def test_unknown_stage_does_not_promote() -> None:
    verdict = evaluate_stage_promotion(stage=99, leg_wins={"random": (10, 10)})
    assert not verdict.promoted
    assert verdict.fail_reason == "unknown_stage"


def test_ssot_interval_skips_eval_when_games_per_seed_zero(tmp_path: Path) -> None:
    cfg = _ssot_cfg(tmp_path, budget=10_000_000)
    cfg.artifacts.ssot_pipeline.qualifier_games_per_seed = 0
    ckpt = tmp_path / "ckpt.pkl"
    ckpt.write_bytes(pickle.dumps({"update": 10}))
    tick = ssot_qualifier_tick(
        cfg,
        update=10,
        total_env_steps=100,
        checkpoint_path=ckpt,
        output_root=tmp_path,
    )
    assert tick.qualifier_stage == 1
    assert tick.promotion_event is None
    assert not any(e.get("event") == "ssot_qualifier_eval" for e in tick.events)
    assert tick.leg_summaries == ()


def test_held_out_eval_seeds_disjoint_from_training_pool() -> None:
    cfg = TrainConfig()
    cfg.training_seed_set = [1, 2, 3]
    cfg.eval_seed_set = [43, 44, 45, 46]
    cfg.seed = 99
    assert held_out_eval_seeds(cfg) == (43, 44, 45, 46)


def test_evaluate_qualifier_legs_delegates_to_eval_seed_set(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = _ssot_cfg(tmp_path)
    cfg.eval_seed_set = [43, 44]
    cfg.artifacts.ssot_pipeline.qualifier_games_per_seed = 1
    captured: dict[str, object] = {}

    def _fake_eval(
        train_cfg: TrainConfig,
        *,
        checkpoint_path: Path,
        stage: int,
    ) -> dict[str, tuple[int, int]]:
        captured["eval_seed_set"] = list(train_cfg.eval_seed_set)
        captured["stage"] = stage
        captured["path"] = checkpoint_path
        return {"random": (2, 4)}

    monkeypatch.setattr(
        "src.jax.tournament_qualifiers.runner.run_held_out_qualifier_eval",
        _fake_eval,
    )
    ckpt = tmp_path / "ckpt.pkl"
    ckpt.write_bytes(pickle.dumps({"params": {}}))
    result = evaluate_qualifier_legs(cfg, checkpoint_path=ckpt, stage=1)
    assert result == {"random": (2, 4)}
    assert captured["eval_seed_set"] == [43, 44]
    assert captured["stage"] == 1


def test_ssot_rollout_stage_view_matches_stage_probs() -> None:
    view = ssot_rollout_stage_view(
        2,
        5,
        snapshot_ids=jnp.zeros((1,), dtype=jnp.int32),
        snapshot_valid_mask=jnp.zeros((1,), dtype=bool),
        snapshot_updates=jnp.zeros((1,), dtype=jnp.int32),
    )
    expected = opponent_family_probs_for_stage(2)
    assert [float(x) for x in view.family_probs] == pytest.approx(list(expected))


def test_final_ship_scores_strict_max_winner() -> None:
    fleet_count = 16
    empty_fleets = JaxFleetState(
        id=jnp.full((fleet_count,), -1, dtype=jnp.int32),
        owner=jnp.full((fleet_count,), -1, dtype=jnp.int32),
        x=jnp.zeros((fleet_count,), dtype=jnp.float32),
        y=jnp.zeros((fleet_count,), dtype=jnp.float32),
        angle=jnp.zeros((fleet_count,), dtype=jnp.float32),
        from_planet_id=jnp.full((fleet_count,), -1, dtype=jnp.int32),
        ships=jnp.zeros((fleet_count,), dtype=jnp.float32),
        active=jnp.zeros((fleet_count,), dtype=bool),
    )
    planets = JaxPlanetState(
        id=jnp.arange(MAX_PLANETS, dtype=jnp.int32),
        owner=jnp.array([0, 1] + [-1] * (MAX_PLANETS - 2), dtype=jnp.int32),
        x=jnp.zeros((MAX_PLANETS,), dtype=jnp.float32),
        y=jnp.zeros((MAX_PLANETS,), dtype=jnp.float32),
        radius=jnp.ones((MAX_PLANETS,), dtype=jnp.float32),
        ships=jnp.array([12.0, 8.0] + [0.0] * (MAX_PLANETS - 2), dtype=jnp.float32),
        production=jnp.zeros((MAX_PLANETS,), dtype=jnp.float32),
        active=jnp.array([True, True] + [False] * (MAX_PLANETS - 2), dtype=bool),
    )
    game = JaxGameState(
        step=jnp.array(10, dtype=jnp.int32),
        player=jnp.array(0, dtype=jnp.int32),
        angular_velocity=jnp.array(0.03, dtype=jnp.float32),
        next_fleet_id=jnp.array(0, dtype=jnp.int32),
        episode_seed=jnp.array(43, dtype=jnp.int32),
        planets=planets,
        initial_planets=planets,
        fleets=empty_fleets,
        comets=empty_comet_state(),
    )
    scores = final_ship_scores(game, 2)
    assert learner_won_from_final_scores(scores, learner_player=0)
    assert not learner_won_from_final_scores(scores, learner_player=1)


def test_ssot_interval_skips_eval_when_not_qualifier_phase(tmp_path: Path) -> None:
    import json

    cfg = _ssot_cfg(tmp_path, budget=10_000_000)
    cfg.artifacts.ssot_pipeline.qualifier_games_per_seed = 2
    state_path = tmp_path / "campaigns" / "ssot_demo" / "bracket" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"phase": "main", "ssot_qualifier_stage": 2, "entries": {}}),
        encoding="utf-8",
    )
    ckpt = tmp_path / "ckpt.pkl"
    ckpt.write_bytes(pickle.dumps({"update": 10}))
    tick = ssot_qualifier_tick(
        cfg,
        update=10,
        total_env_steps=100,
        checkpoint_path=ckpt,
        output_root=tmp_path,
    )
    assert not any(e.get("event") == "ssot_qualifier_eval" for e in tick.events)
