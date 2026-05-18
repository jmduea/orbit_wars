import pytest
import torch

from src.checkpoint_compat import (
    feature_metadata,
    validate_checkpoint_feature_compatibility,
)
from src.config import EnvConfig, TrainConfig
from src.features import candidate_feature_dim, global_feature_dim, self_feature_dim
from src.policy import build_policy
from src.train import save_checkpoint


def test_validate_checkpoint_feature_metadata_rejects_dim_mismatch() -> None:
    cfg = EnvConfig(feature_history_steps=2)
    checkpoint = {
        "feature_metadata": {
            "feature_history_steps": 1,
            "self_feature_dim": 30,
            "candidate_feature_dim": 24,
            "global_feature_dim": 41,
        },
        "policy": {},
    }

    with pytest.raises(ValueError, match="retrained with the current feature config or migrated") as exc_info:
        validate_checkpoint_feature_compatibility(checkpoint, cfg, checkpoint_path="old.pt")

    message = str(exc_info.value)
    assert "old.pt" in message
    assert "self_feature_dim: checkpoint=30, current=60" in message
    assert "candidate_feature_dim: checkpoint=24, current=48" in message
    assert "global_feature_dim: checkpoint=41, current=82" in message


def test_validate_legacy_checkpoint_without_metadata_allows_matching_weight_dims() -> None:
    cfg = TrainConfig()
    cfg.env.feature_history_steps = 2
    policy = build_policy(
        architecture=cfg.model.architecture,
        self_dim=self_feature_dim(cfg.env),
        candidate_dim=candidate_feature_dim(cfg.env),
        global_dim=global_feature_dim(cfg.env),
        candidate_count=cfg.env.candidate_count,
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        attention_heads=cfg.model.attention_heads,
    )

    validate_checkpoint_feature_compatibility(policy.state_dict(), cfg.env)


def test_save_checkpoint_persists_feature_metadata(tmp_path) -> None:
    cfg = TrainConfig()
    cfg.env.feature_history_steps = 3
    policy = build_policy(
        architecture=cfg.model.architecture,
        self_dim=self_feature_dim(cfg.env),
        candidate_dim=candidate_feature_dim(cfg.env),
        global_dim=global_feature_dim(cfg.env),
        candidate_count=cfg.env.candidate_count,
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        attention_heads=cfg.model.attention_heads,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.ppo.lr)

    checkpoint_path = save_checkpoint(
        tmp_path,
        "run",
        7,
        policy,
        optimizer,
        cfg,
    )

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["feature_metadata"] == feature_metadata(cfg.env)
    assert checkpoint["feature_metadata"]["feature_history_steps"] == 3
