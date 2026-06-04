"""Hydra composition for artifacts=ssot_pipeline."""

from __future__ import annotations

from src.config import compose_hydra_train_config


def test_ssot_pipeline_artifacts_profile_composes() -> None:
    cfg = compose_hydra_train_config(["artifacts=ssot_pipeline"])

    assert not cfg.artifacts.promotion.enabled
    assert not cfg.artifacts.tournament.enabled
    assert not cfg.artifacts.unified_tournament.enabled
    assert not cfg.artifacts.bracket_training.enabled
    assert cfg.artifacts.ssot_pipeline.enabled
    assert cfg.artifacts.ssot_pipeline.qualifier_eval_interval_updates == 50
    assert not cfg.artifacts.artifact_pipeline.checkpoint_eval_async
    assert not cfg.artifacts.artifact_pipeline.replay_async
    assert not cfg.artifacts.replay.enabled
    assert cfg.telemetry.wandb.enabled
    assert cfg.eval_seed_set == [43, 44, 45, 46]
