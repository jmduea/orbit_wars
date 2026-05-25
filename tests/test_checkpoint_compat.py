from __future__ import annotations

import pytest

from src.artifacts.checkpoint_compat import (
    feature_metadata,
    infer_feature_metadata_from_state_dict,
    validate_checkpoint_feature_compatibility,
)
from src.config.schema import TaskConfig


def _v2_task(**kwargs) -> TaskConfig:
    base = dict(
        candidate_count=4,
        encoding_version="v2",
        ship_feature_scale=1000.0,
        feature_history_steps=1,
    )
    base.update(kwargs)
    return TaskConfig(**base)


def test_feature_metadata_v2_includes_schema_and_dims() -> None:
    metadata = feature_metadata(_v2_task())

    assert metadata["schema_version"] == 2
    assert metadata["encoding_version"] == "v2"
    assert metadata["planet_feature_dim"] == 13
    assert metadata["edge_feature_dim"] == 12
    assert metadata["global_feature_dim"] == 46
    assert metadata["ship_feature_scale"] == 1000.0
    assert metadata["edge_layout"] == "top_k_per_source"
    assert metadata["edge_k"] == 3


def test_feature_metadata_v1_keeps_legacy_dims() -> None:
    metadata = feature_metadata(TaskConfig(encoding_version="v1"))

    assert metadata["schema_version"] == 1
    assert metadata["encoding_version"] == "v1"
    assert "self_feature_dim" in metadata
    assert "planet_feature_dim" not in metadata


def test_validate_rejects_v1_checkpoint_for_v2_config() -> None:
    checkpoint = {
        "feature_metadata": {
            "self_feature_dim": 30,
            "candidate_feature_dim": 24,
            "global_feature_dim": 20,
            "feature_history_steps": 1,
        }
    }

    with pytest.raises(ValueError, match="v1 feature metadata"):
        validate_checkpoint_feature_compatibility(checkpoint, _v2_task())


def test_validate_accepts_matching_v2_dims() -> None:
    env_cfg = _v2_task()
    checkpoint = {"feature_metadata": feature_metadata(env_cfg)}

    validate_checkpoint_feature_compatibility(checkpoint, env_cfg)


def test_validate_rejects_v2_dim_mismatch() -> None:
    env_cfg = _v2_task()
    stored = feature_metadata(env_cfg)
    stored = dict(stored)
    stored["planet_feature_dim"] = stored["planet_feature_dim"] + 1
    checkpoint = {"feature_metadata": stored}

    with pytest.raises(ValueError, match="v2 feature configuration"):
        validate_checkpoint_feature_compatibility(checkpoint, env_cfg)


def test_infer_v2_metadata_from_state_dict_keys() -> None:
    state_dict = {
        "params": {
            "encoder_module": {
                "planet_enc_0": {"kernel": __import__("numpy").zeros((13, 16))},
                "edge_enc_0": {"kernel": __import__("numpy").zeros((12, 16))},
                "global_enc_0": {"kernel": __import__("numpy").zeros((46, 16))},
            }
        }
    }

    inferred = infer_feature_metadata_from_state_dict(state_dict)

    assert inferred is not None
    assert inferred["schema_version"] == 2
    assert inferred["encoding_version"] == "v2"
    assert inferred["planet_feature_dim"] == 13
    assert inferred["edge_feature_dim"] == 12
    assert inferred["global_feature_dim"] == 46
