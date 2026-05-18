from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from .config import EnvConfig
from .features import (
    candidate_feature_dim,
    feature_history_steps,
    global_feature_dim,
    self_feature_dim,
)

FEATURE_METADATA_KEY = "feature_metadata"


def feature_metadata(env_cfg: EnvConfig) -> dict[str, int]:
    """Return checkpoint metadata that describes feature-dependent input shapes."""

    return {
        "feature_history_steps": feature_history_steps(env_cfg),
        "self_feature_dim": self_feature_dim(env_cfg),
        "candidate_feature_dim": candidate_feature_dim(env_cfg),
        "global_feature_dim": global_feature_dim(env_cfg),
    }


def checkpoint_state_dict(checkpoint: object) -> Mapping[str, object] | None:
    """Extract a policy state-dict from either a full checkpoint or raw state-dict."""

    if not isinstance(checkpoint, Mapping):
        return None
    policy = checkpoint.get("policy")
    if isinstance(policy, Mapping):
        return policy
    if all(isinstance(key, str) for key in checkpoint.keys()):
        return checkpoint
    return None


def checkpoint_feature_metadata(checkpoint: object) -> dict[str, int] | None:
    """Return stored feature metadata, supporting the current and legacy locations."""

    if not isinstance(checkpoint, Mapping):
        return None
    raw = checkpoint.get(FEATURE_METADATA_KEY)
    if raw is None:
        metadata = checkpoint.get("metadata")
        if isinstance(metadata, Mapping):
            raw = metadata.get(FEATURE_METADATA_KEY)
    if not isinstance(raw, Mapping):
        return None
    parsed: dict[str, int] = {}
    for key in (
        "feature_history_steps",
        "self_feature_dim",
        "candidate_feature_dim",
        "global_feature_dim",
    ):
        if key in raw:
            parsed[key] = int(raw[key])
    return parsed or None


def infer_feature_metadata_from_state_dict(
    state_dict: Mapping[str, object] | None,
) -> dict[str, int] | None:
    """Infer input feature dimensions from the first policy encoder weights."""

    if state_dict is None:
        return None
    key_map = {
        "self_feature_dim": "self_encoder.0.weight",
        "candidate_feature_dim": "candidate_encoder.0.weight",
        "global_feature_dim": "global_encoder.0.weight",
    }
    inferred: dict[str, int] = {}
    for metadata_key, weight_key in key_map.items():
        weight = state_dict.get(weight_key)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            inferred[metadata_key] = int(weight.shape[1])
    return inferred or None


def validate_checkpoint_feature_compatibility(
    checkpoint: object,
    env_cfg: EnvConfig,
    *,
    checkpoint_path: str | Path | None = None,
) -> None:
    """Raise a clear error when checkpoint feature inputs differ from config."""

    expected = feature_metadata(env_cfg)
    stored = checkpoint_feature_metadata(checkpoint)
    source = "checkpoint metadata"
    if stored is None:
        stored = infer_feature_metadata_from_state_dict(checkpoint_state_dict(checkpoint))
        source = "policy weight shapes (checkpoint has no feature metadata)"
    if stored is None:
        return

    dimension_keys = (
        "self_feature_dim",
        "candidate_feature_dim",
        "global_feature_dim",
    )
    mismatches = [
        (key, stored.get(key), expected[key])
        for key in dimension_keys
        if key in stored and stored.get(key) != expected[key]
    ]
    if not mismatches:
        return

    location = f" at {checkpoint_path}" if checkpoint_path is not None else ""
    mismatch_text = ", ".join(
        f"{key}: checkpoint={checkpoint_value}, current={current_value}"
        for key, checkpoint_value, current_value in mismatches
    )
    stored_history = stored.get("feature_history_steps", "unknown")
    raise ValueError(
        f"Checkpoint{location} is incompatible with the current feature configuration. "
        f"Feature dimensions from {source} differ from the current config ({mismatch_text}). "
        f"Checkpoint feature_history_steps={stored_history}; current "
        f"feature_history_steps={expected['feature_history_steps']}. "
        "Policy input dimensions are part of the model architecture, so this "
        "checkpoint must be retrained with the current feature config or migrated "
        "with an explicit architecture conversion."
    )
