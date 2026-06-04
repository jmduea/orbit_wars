"""SSOT stage-3 → main bracket transition (U7)."""

from __future__ import annotations

import pickle
from pathlib import Path

from src.artifacts.tournament.bracket.state import load_bracket_state
from src.config.schema import ArtifactsConfig, OutputConfig, SsotPipelineConfig, TrainConfig
from src.jax.tournament_qualifiers.runner import ssot_qualifier_tick


def _cfg(tmp_path: Path) -> TrainConfig:
    cfg = TrainConfig()
    cfg.output = OutputConfig(root=str(tmp_path), campaign="bracket_demo")
    cfg.artifacts = ArtifactsConfig(
        ssot_pipeline=SsotPipelineConfig(
            enabled=True,
            qualifier_max_env_steps=10_000_000,
            qualifier_eval_interval_updates=10,
            qualifier_games_per_seed=2,
        )
    )
    return cfg


def test_stage_three_promotion_enters_main_bracket(tmp_path: Path, monkeypatch) -> None:
    from src.jax import tournament_qualifiers

    def _fake_legs() -> dict[str, tuple[int, int]]:
        return {"noop": (6, 10), "random": (6, 10), "nearest_sniper": (6, 10)}

    monkeypatch.setattr(
        tournament_qualifiers.runner,
        "evaluate_qualifier_legs",
        lambda **_: _fake_legs(),
    )
    cfg = _cfg(tmp_path)
    state_path = (
        tmp_path / "campaigns" / "bracket_demo" / "bracket" / "state.json"
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    state_path.write_text(
        json.dumps({"ssot_qualifier_stage": 3, "phase": "qualifier", "entries": {}}),
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
    assert tick.promotion_event is not None
    state = load_bracket_state(state_path)
    assert state.phase == "main"
    assert tick.qualifier_stage == 4
