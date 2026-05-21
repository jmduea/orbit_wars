from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from .config import TaskConfig, TrainConfig
from .feature_registry import (
    candidate_feature_dim,
    feature_history_steps,
    global_feature_dim,
    self_feature_dim,
)

FEATURE_METADATA_KEY = "feature_metadata"

LEGACY_CONFIG_FIELDS = (
    "env",
    "ppo",
    "training_format",
    "opponent_mix",
    "wandb",
    "artifact_pipeline",
    "replay",
    "checkpoint_retention",
    "save_dir",
    "checkpoint_every",
    "self_play_enabled",
    "self_play_update_interval",
    "self_play_latest_probability",
    "self_play_pool_size",
    "self_play_snapshot_interval",
    "opponent",
    "multi_opponent_mode",
    "alternate_player_sides",
)

CANONICAL_CONFIG_FIELDS = (
    "model",
    "task",
    "reward",
    "training",
    "format",
    "curriculum",
    "opponents",
    "telemetry",
    "artifacts",
)


def load_checkpoint_payload(checkpoint_path: str | Path) -> object:
    """Load a checkpoint pickle, turning old config pickles into a clear error."""

    import pickle

    path = Path(checkpoint_path)
    try:
        with path.open("rb") as file:
            return pickle.load(file)
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        raise ValueError(
            f"Checkpoint at {path} was saved with the pre-migration legacy config "
            "schema and cannot be loaded by the canonical responsibility-group "
            "runtime. Retrain from the current config or migrate the checkpoint "
            "with an explicit one-off conversion."
        ) from exc


def _config_has_field(config: object, field_name: str) -> bool:
    if isinstance(config, Mapping):
        return field_name in config
    try:
        return hasattr(config, field_name)
    except Exception:
        return False


def validate_checkpoint_config_compatibility(
    checkpoint: object,
    *,
    checkpoint_path: str | Path | None = None,
) -> None:
    """Reject checkpoints that embed the old flat runtime config shape."""

    if not isinstance(checkpoint, Mapping) or "config" not in checkpoint:
        return
    stored_config = checkpoint["config"]
    if isinstance(stored_config, TrainConfig):
        return

    legacy_fields = [
        field_name
        for field_name in LEGACY_CONFIG_FIELDS
        if _config_has_field(stored_config, field_name)
    ]
    location = f" at {checkpoint_path}" if checkpoint_path is not None else ""
    if legacy_fields:
        fields = ", ".join(legacy_fields[:5])
        suffix = "" if len(legacy_fields) <= 5 else ", ..."
        raise ValueError(
            f"Checkpoint{location} contains legacy config fields ({fields}{suffix}). "
            "Legacy checkpoint configs are no longer normalized at runtime; retrain "
            "from the current responsibility-group config or migrate this checkpoint "
            "with an explicit one-off conversion."
        )

    missing_fields = [
        field_name
        for field_name in CANONICAL_CONFIG_FIELDS
        if not _config_has_field(stored_config, field_name)
    ]
    if missing_fields:
        fields = ", ".join(missing_fields[:5])
        suffix = "" if len(missing_fields) <= 5 else ", ..."
        raise ValueError(
            f"Checkpoint{location} does not contain the canonical responsibility-group "
            f"config fields ({fields}{suffix}). Retrain from the current config or "
            "migrate the checkpoint explicitly before loading it."
        )


def feature_metadata(env_cfg: TaskConfig) -> dict[str, int]:
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

    def _shape_2d_second_dim(value: object) -> int | None:
        ndim = getattr(value, "ndim", None)
        shape = getattr(value, "shape", None)
        if ndim is not None and shape is not None:
            try:
                if int(ndim) == 2 and len(shape) >= 2:
                    return int(shape[1])
            except (TypeError, ValueError):
                return None

        try:
            array = np.asarray(value)
        except Exception:
            return None

        if array.ndim != 2:
            return None
        try:
            return int(array.shape[1])
        except (TypeError, ValueError):
            return None

    key_map = {
        "self_feature_dim": "self_encoder.0.weight",
        "candidate_feature_dim": "candidate_encoder.0.weight",
        "global_feature_dim": "global_encoder.0.weight",
    }
    inferred: dict[str, int] = {}
    for metadata_key, weight_key in key_map.items():
        dim = _shape_2d_second_dim(state_dict.get(weight_key))
        if dim is not None:
            inferred[metadata_key] = dim
    return inferred or None


def validate_checkpoint_feature_compatibility(
    checkpoint: object,
    env_cfg: TaskConfig,
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
