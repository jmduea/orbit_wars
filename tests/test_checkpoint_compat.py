from __future__ import annotations

import pytest

from src.artifacts.checkpoint_compat import (
    feature_metadata,
    infer_feature_metadata_from_state_dict,
    validate_checkpoint_encoder_compatibility,
    validate_checkpoint_feature_compatibility,
)
from src.config.schema import ModelConfig, TaskConfig


def _task(**kwargs) -> TaskConfig:
    base = dict(
        candidate_count=4,
        ship_feature_scale=1000.0,
        feature_history_steps=1,
    )
    base.update(kwargs)
    return TaskConfig(**base)


def test_feature_metadata_includes_schema_and_dims() -> None:
    metadata = feature_metadata(_task())

    assert metadata["schema_version"] == 5
    assert metadata["planet_feature_dim"] == 13
    assert metadata["edge_feature_dim"] == 19
    assert metadata["global_feature_dim"] == 46
    assert metadata["ship_feature_scale"] == 1000.0
    assert metadata["edge_layout"] == "top_k_per_source"
    assert metadata["edge_k"] == 3
    assert metadata["intercept_anchors"] == (1.0, 6.0)
    assert isinstance(metadata["intercept_anchors"], tuple)


def test_validate_rejects_v1_checkpoint_metadata() -> None:
    checkpoint = {
        "feature_metadata": {
            "self_feature_dim": 30,
            "candidate_feature_dim": 24,
            "global_feature_dim": 20,
            "feature_history_steps": 1,
        }
    }

    with pytest.raises(ValueError, match="legacy v1 feature metadata"):
        validate_checkpoint_feature_compatibility(checkpoint, _task())


def test_validate_rejects_v4_schema_version() -> None:
    env_cfg = _task()
    stored = dict(feature_metadata(env_cfg))
    stored["schema_version"] = 4
    checkpoint = {"feature_metadata": stored}

    with pytest.raises(ValueError, match="schema_version=4"):
        validate_checkpoint_feature_compatibility(checkpoint, env_cfg)


def test_validate_accepts_matching_dims() -> None:
    env_cfg = _task()
    checkpoint = {"feature_metadata": feature_metadata(env_cfg)}

    validate_checkpoint_feature_compatibility(checkpoint, env_cfg)


def test_validate_rejects_dim_mismatch() -> None:
    env_cfg = _task()
    stored = dict(feature_metadata(env_cfg))
    stored["planet_feature_dim"] = stored["planet_feature_dim"] + 1
    checkpoint = {"feature_metadata": stored}

    with pytest.raises(ValueError, match="incompatible with the current feature"):
        validate_checkpoint_feature_compatibility(checkpoint, env_cfg)


def test_infer_metadata_from_state_dict_keys() -> None:
    state_dict = {
        "params": {
            "encoder_module": {
                "planet_enc_0": {"kernel": __import__("numpy").zeros((13, 16))},
                "edge_enc_0": {"kernel": __import__("numpy").zeros((19, 16))},
                "global_enc_0": {"kernel": __import__("numpy").zeros((46, 16))},
            }
        }
    }

    inferred = infer_feature_metadata_from_state_dict(state_dict)

    assert inferred is not None
    assert inferred["schema_version"] == 5
    assert inferred["planet_feature_dim"] == 13
    assert inferred["edge_feature_dim"] == 19
    assert inferred["global_feature_dim"] == 46


def test_validate_accepts_matching_intercept_anchors() -> None:
    env_cfg = _task()
    checkpoint = {"feature_metadata": feature_metadata(env_cfg)}

    validate_checkpoint_feature_compatibility(checkpoint, env_cfg)


def test_validate_rejects_intercept_anchor_mismatch() -> None:
    env_cfg = _task()
    stored = dict(feature_metadata(env_cfg))
    stored["intercept_anchors"] = (1.0, 4.0)
    checkpoint = {"feature_metadata": stored}

    with pytest.raises(ValueError, match="intercept_anchors"):
        validate_checkpoint_feature_compatibility(checkpoint, env_cfg)


def test_feature_metadata_includes_encoder_backbone_for_gnn() -> None:
    metadata = feature_metadata(_task(), model_cfg=ModelConfig(architecture="gnn_pointer"))
    assert metadata["encoder_backbone"] == "planet_gnn"


def test_feature_metadata_includes_encoder_backbone_for_transformer() -> None:
    metadata = feature_metadata(
        _task(), model_cfg=ModelConfig(architecture="planet_graph_transformer")
    )
    assert metadata["encoder_backbone"] == "planet_self_attention"


def test_validate_rejects_encoder_backbone_mismatch() -> None:
    from src.config import TrainConfig

    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    stored = dict(feature_metadata(cfg.task, model_cfg=cfg.model))
    stored["encoder_backbone"] = "planet_gnn"

    with pytest.raises(ValueError, match="encoder_backbone"):
        validate_checkpoint_encoder_compatibility(stored, cfg)


def test_infer_metadata_rejects_v1_self_encoder() -> None:
    state_dict = {
        "params": {
            "self_encoder_0": {"kernel": __import__("numpy").zeros((30, 16))},
        }
    }

    assert infer_feature_metadata_from_state_dict(state_dict) is None


def test_feature_metadata_includes_factorized_pointer_decoder() -> None:
    metadata = feature_metadata(
        _task(),
        model_cfg=ModelConfig(
            architecture="gnn_pointer",
            pointer_decoder="factorized_topk",
        ),
    )
    assert metadata["pointer_decoder"] == "factorized_topk"
    assert metadata["action_layout_version"] == 2


def test_validate_rejects_pointer_decoder_mismatch() -> None:
    from src.config import TrainConfig

    cfg = TrainConfig()
    cfg.model.pointer_decoder = "joint_flat"
    stored = dict(feature_metadata(cfg.task, model_cfg=cfg.model))
    stored["pointer_decoder"] = "factorized_topk"
    stored["action_layout_version"] = 2

    from src.artifacts.checkpoint_compat import validate_checkpoint_pointer_decoder_compatibility

    with pytest.raises(ValueError, match="pointer_decoder"):
        validate_checkpoint_pointer_decoder_compatibility(stored, cfg)


def test_validate_rejects_action_layout_mismatch() -> None:
    from src.config import TrainConfig

    cfg = TrainConfig()
    cfg.model.pointer_decoder = "factorized_topk"
    stored = dict(feature_metadata(cfg.task, model_cfg=cfg.model))
    stored["action_layout_version"] = 1

    from src.artifacts.checkpoint_compat import validate_checkpoint_pointer_decoder_compatibility

    with pytest.raises(ValueError, match="action_layout_version"):
        validate_checkpoint_pointer_decoder_compatibility(stored, cfg)

