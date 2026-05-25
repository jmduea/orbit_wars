from __future__ import annotations

from src.config import TrainConfig
from src.jax.submission_runtime import apply_feature_metadata_to_model_config


def test_apply_feature_metadata_sets_pointer_decoder() -> None:
    cfg = TrainConfig()
    cfg.model.pointer_decoder = "joint_flat"

    updated = apply_feature_metadata_to_model_config(
        cfg,
        {"pointer_decoder": "factorized_topk", "action_layout_version": 2},
    )

    assert updated.model.pointer_decoder == "factorized_topk"
