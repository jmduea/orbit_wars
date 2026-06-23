"""Tests for submit-valid pipeline selection."""

from __future__ import annotations

import pytest

from src.artifacts.submit_valid_pipeline import (
    HYBRID_CHECKPOINT_EVAL,
    active_submit_valid_pipelines,
    primary_submit_valid_pipeline,
    validate_submit_valid_pipelines_exclusive,
)
from src.config import compose_hydra_train_config


def test_hybrid_promotion_profile_selects_hybrid_pipeline() -> None:
    cfg = compose_hydra_train_config(["artifacts=hybrid_promotion"])
    assert primary_submit_valid_pipeline(cfg) == HYBRID_CHECKPOINT_EVAL


def test_ssot_pipeline_profile_selects_ssot() -> None:
    cfg = compose_hydra_train_config(["artifacts=ssot_pipeline"])
    assert primary_submit_valid_pipeline(cfg) == "ssot_pipeline"
    assert active_submit_valid_pipelines(cfg) == ("ssot_pipeline",)


def test_default_artifacts_has_no_submit_valid_pipeline() -> None:
    cfg = compose_hydra_train_config(["artifacts=default"])
    assert primary_submit_valid_pipeline(cfg) is None


def test_mutually_exclusive_submit_valid_pipelines_rejected() -> None:
    with pytest.raises(ValueError, match="mutually exclusive submit-valid pipelines"):
        cfg = compose_hydra_train_config(
            [
                "artifacts=ssot_pipeline",
                "artifacts.promotion.strategy=hybrid",
                "artifacts.artifact_pipeline.checkpoint_eval_async=true",
            ]
        )
        validate_submit_valid_pipelines_exclusive(cfg)
