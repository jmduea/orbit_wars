"""SSOT JAX tournament qualifier promotion and tick hooks."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from src.artifacts.tournament.bracket.state import load_bracket_state
from src.config.schema import ArtifactsConfig, OutputConfig, SsotPipelineConfig, TrainConfig
from src.jax.qualifier_calibration import load_qualifier_calibration
from src.jax.tournament_qualifiers.metrics import learner_won_from_final_scores
from src.jax.tournament_qualifiers.promotion import evaluate_stage_promotion
from src.jax.tournament_qualifiers.runner import ssot_qualifier_tick


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
