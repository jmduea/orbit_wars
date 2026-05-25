"""Checkpoint feature-compatibility metadata and validation.

Floor schema_version is 4 (intercept-anchor edges). Earlier versions are not
loadable; migrate by retraining.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from src.config import ModelConfig, TaskConfig, TrainConfig
from src.features.registry import (
    edge_feature_dim,
    edge_k,
    feature_history_steps,
    global_feature_dim,
    planet_feature_dim,
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

METADATA_KEYS = (
    "schema_version",
    "feature_history_steps",
    "planet_feature_dim",
    "edge_feature_dim",
    "global_feature_dim",
    "ship_feature_scale",
    "edge_layout",
    "edge_k",
    "intercept_anchors",
    "encoder_backbone",
    "pointer_decoder",
    "action_layout_version",
)

POINTER_DECODER_JOINT_FLAT = "joint_flat"
POINTER_DECODER_FACTORIZED_TOPK = "factorized_topk"
ACTION_LAYOUT_JOINT_FLAT = 1
ACTION_LAYOUT_FACTORIZED_TOPK = 2


def encoder_backbone_for_architecture(architecture: str) -> str:
    """Map ``model.architecture`` to the checkpoint encoder-backbone slug."""

    normalized = architecture.strip().lower()
    if normalized in {"gnn_pointer", "gnn_pointer_v2"}:
        return "planet_gnn"
    if normalized == "planet_graph_transformer":
        return "planet_self_attention"
    raise ValueError(
        f"Unsupported model architecture for encoder_backbone metadata: {architecture!r}"
    )


def pointer_decoder_for_model(model_cfg: ModelConfig) -> str:
    """Map ``ModelConfig`` to the checkpoint pointer-decoder slug."""

    raw = getattr(model_cfg, "pointer_decoder", POINTER_DECODER_JOINT_FLAT)
    normalized = str(raw).strip().lower()
    if normalized in {POINTER_DECODER_JOINT_FLAT, "joint", "flat"}:
        return POINTER_DECODER_JOINT_FLAT
    if normalized in {
        POINTER_DECODER_FACTORIZED_TOPK,
        "factorized",
        "factorized_topk",
    }:
        return POINTER_DECODER_FACTORIZED_TOPK
    raise ValueError(
        f"Unsupported pointer_decoder {raw!r}. Expected "
        f"{POINTER_DECODER_JOINT_FLAT!r} or {POINTER_DECODER_FACTORIZED_TOPK!r}."
    )


def is_factorized_pointer_decoder(model_cfg: ModelConfig) -> bool:
    """Return True when the configured model uses the factorized top-K decoder."""

    return pointer_decoder_for_model(model_cfg) == POINTER_DECODER_FACTORIZED_TOPK


def action_layout_version_for_pointer_decoder(pointer_decoder: str) -> int:
    """Return the integer action-layout version for a pointer decoder slug."""

    if pointer_decoder == POINTER_DECODER_JOINT_FLAT:
        return ACTION_LAYOUT_JOINT_FLAT
    if pointer_decoder == POINTER_DECODER_FACTORIZED_TOPK:
        return ACTION_LAYOUT_FACTORIZED_TOPK
    raise ValueError(f"Unsupported pointer_decoder slug: {pointer_decoder!r}")


def validate_checkpoint_pointer_decoder_compatibility(
    stored: Mapping[str, object] | None,
    cfg: TrainConfig,
    *,
    checkpoint_path: str | Path | None = None,
) -> None:
    """Raise when checkpoint pointer decoder differs from the current model."""

    if stored is None:
        return

    stored_decoder = stored.get("pointer_decoder")
    if stored_decoder is None:
        return

    expected = pointer_decoder_for_model(cfg.model)
    if str(stored_decoder) != expected:
        location = f" at {checkpoint_path}" if checkpoint_path is not None else ""
        raise ValueError(
            f"Checkpoint{location} pointer_decoder={stored_decoder!r} is incompatible "
        f"with current model (expected pointer_decoder={expected!r}). Retrain or "
        "load a matching checkpoint."
        )

    stored_layout = stored.get("action_layout_version")
    if stored_layout is not None:
        expected_layout = action_layout_version_for_pointer_decoder(expected)
        if int(stored_layout) != expected_layout:
            location = f" at {checkpoint_path}" if checkpoint_path is not None else ""
            raise ValueError(
                f"Checkpoint{location} action_layout_version={stored_layout!r} is "
                f"incompatible with pointer_decoder={expected!r} (expected "
                f"action_layout_version={expected_layout}). Retrain or load a "
                "matching checkpoint."
            )
    return


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


def feature_metadata(
    env_cfg: TaskConfig,
    *,
    model_cfg: ModelConfig | None = None,
) -> dict[str, int | float | str | tuple]:
    """Return checkpoint metadata that describes feature-dependent input shapes."""

    history = feature_history_steps(env_cfg)
    metadata: dict[str, int | float | str | tuple] = {
        "schema_version": 4,
        "feature_history_steps": history,
        "planet_feature_dim": planet_feature_dim(env_cfg),
        "edge_feature_dim": edge_feature_dim(env_cfg),
        "global_feature_dim": global_feature_dim(env_cfg),
        "ship_feature_scale": float(getattr(env_cfg, "ship_feature_scale", 1000.0)),
        "edge_layout": "top_k_per_source",
        "edge_k": edge_k(env_cfg),
        "intercept_anchors": tuple(map(float, env_cfg.intercept_anchors)),
    }
    if model_cfg is not None:
        metadata["encoder_backbone"] = encoder_backbone_for_architecture(
            model_cfg.architecture
        )
        pointer_decoder = pointer_decoder_for_model(model_cfg)
        metadata["pointer_decoder"] = pointer_decoder
        metadata["action_layout_version"] = action_layout_version_for_pointer_decoder(
            pointer_decoder
        )
    return metadata


def checkpoint_state_dict(checkpoint: object) -> Mapping[str, object] | None:
    """Extract a policy state-dict from either a full checkpoint or raw state-dict."""

    if not isinstance(checkpoint, Mapping):
        return None
    policy = checkpoint.get("policy")
    if isinstance(policy, Mapping):
        return policy
    params = checkpoint.get("params")
    if isinstance(params, Mapping):
        return params
    if all(isinstance(key, str) for key in checkpoint.keys()):
        return checkpoint
    return None


def checkpoint_feature_metadata(
    checkpoint: object,
) -> dict[str, int | float | str | tuple] | None:
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
    parsed: dict[str, int | float | str | tuple] = {}
    for key in METADATA_KEYS:
        if key in raw:
            value = raw[key]
            if key in {"schema_version", "edge_k", "feature_history_steps"}:
                parsed[key] = int(value)
            elif key == "ship_feature_scale":
                parsed[key] = float(value)
            elif key in {"edge_layout", "encoder_backbone", "pointer_decoder"}:
                parsed[key] = str(value)
            elif key == "action_layout_version":
                parsed[key] = int(value)
            elif key == "intercept_anchors":
                parsed[key] = tuple(float(v) for v in value)
            else:
                parsed[key] = int(value)
    return parsed or None


def _find_flax_dense_input_dim(
    params_root: Mapping[str, object] | None,
    encoder_prefix: str,
) -> int | None:
    if params_root is None:
        return None
    dense_name = f"{encoder_prefix}_0"
    if isinstance(params_root, dict):
        module_payload = params_root.get(dense_name)
        if isinstance(module_payload, dict):
            kernel = module_payload.get("kernel")
            if kernel is not None and getattr(kernel, "ndim", 0) >= 1:
                return int(kernel.shape[0])
        for child in params_root.values():
            if isinstance(child, dict):
                found = _find_flax_dense_input_dim(child, encoder_prefix)
                if found is not None:
                    return found
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


def infer_feature_metadata_from_state_dict(
    state_dict: Mapping[str, object] | None,
) -> dict[str, int | float | str] | None:
    """Infer input feature dimensions from the first policy encoder weights."""

    if state_dict is None:
        return None

    params_root = state_dict.get("params") if isinstance(state_dict, Mapping) else None
    if not isinstance(params_root, Mapping):
        params_root = state_dict

    if _find_flax_dense_input_dim(params_root, "self_encoder") is not None:
        return None

    v2_dims = {
        "planet_feature_dim": _find_flax_dense_input_dim(params_root, "planet_enc"),
        "edge_feature_dim": _find_flax_dense_input_dim(params_root, "edge_enc"),
        "global_feature_dim": _find_flax_dense_input_dim(params_root, "global_enc"),
    }
    if all(value is not None for value in v2_dims.values()):
        return {
            "schema_version": 4,
            **{key: int(value) for key, value in v2_dims.items()},
        }

    return None


def _intercept_anchors_match(
    stored: object, expected: object, *, tolerance: float = 1e-6
) -> bool:
    """Element-wise float equality with tolerance for the anchor tuple."""

    if stored is None or expected is None:
        return stored is None and expected is None
    try:
        stored_tuple = tuple(float(v) for v in stored)  # type: ignore[arg-type]
        expected_tuple = tuple(float(v) for v in expected)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    if len(stored_tuple) != len(expected_tuple):
        return False
    return all(
        abs(a - b) <= tolerance for a, b in zip(stored_tuple, expected_tuple)
    )


def _is_legacy_v1_metadata(metadata: Mapping[str, object]) -> bool:
    if metadata.get("schema_version") == 1:
        return True
    if "self_feature_dim" in metadata and "planet_feature_dim" not in metadata:
        return True
    if str(metadata.get("encoding_version", "")).strip().lower() == "v1":
        return True
    return False


def validate_checkpoint_encoder_compatibility(
    stored: Mapping[str, object] | None,
    cfg: TrainConfig,
    *,
    checkpoint_path: str | Path | None = None,
) -> None:
    """Raise when checkpoint encoder backbone differs from the current model."""

    if stored is None:
        return

    stored_backbone = stored.get("encoder_backbone")
    if stored_backbone is None:
        return

    expected = encoder_backbone_for_architecture(cfg.model.architecture)
    if str(stored_backbone) == expected:
        return

    location = f" at {checkpoint_path}" if checkpoint_path is not None else ""
    raise ValueError(
        f"Checkpoint{location} encoder_backbone={stored_backbone!r} is incompatible "
        f"with current model architecture {cfg.model.architecture!r} "
        f"(expected encoder_backbone={expected!r}). Retrain or load a matching "
        "checkpoint."
    )


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

    location = f" at {checkpoint_path}" if checkpoint_path is not None else ""

    raw_metadata = None
    if isinstance(checkpoint, Mapping):
        raw_metadata = checkpoint.get(FEATURE_METADATA_KEY)
        if raw_metadata is None:
            metadata_root = checkpoint.get("metadata")
            if isinstance(metadata_root, Mapping):
                raw_metadata = metadata_root.get(FEATURE_METADATA_KEY)
    legacy_metadata = raw_metadata if isinstance(raw_metadata, Mapping) else stored

    if _is_legacy_v1_metadata(legacy_metadata):
        raise ValueError(
            f"Checkpoint{location} uses legacy v1 feature metadata or self/candidate "
            "encoder weights. v1 checkpoints cannot be loaded; retrain with the "
            "current planet-edge feature encoding."
        )

    stored_schema = stored.get("schema_version")
    if stored_schema is not None and int(stored_schema) < 4:
        raise ValueError(
            f"Checkpoint{location} uses feature schema_version={stored_schema}. "
            "Schema v4 (intercept-anchor edge features) is required; v3 → v4 "
            "migration required — retrain or run an explicit conversion."
        )

    dimension_keys = (
        "planet_feature_dim",
        "edge_feature_dim",
        "global_feature_dim",
    )
    mismatches: list[tuple[str, object, object]] = [
        (key, stored.get(key), expected[key])
        for key in dimension_keys
        if key in stored and stored.get(key) != expected[key]
    ]

    if "intercept_anchors" in stored:
        stored_anchors = stored.get("intercept_anchors")
        expected_anchors = expected.get("intercept_anchors")
        if not _intercept_anchors_match(stored_anchors, expected_anchors):
            mismatches.append(
                ("intercept_anchors", stored_anchors, expected_anchors)
            )

    if not mismatches:
        return

    mismatch_text = ", ".join(
        f"{key}: checkpoint={checkpoint_value}, current={current_value}"
        for key, checkpoint_value, current_value in mismatches
    )
    stored_history = stored.get("feature_history_steps", "unknown")
    raise ValueError(
        f"Checkpoint{location} is incompatible with the current feature "
        f"configuration. Feature dimensions from {source} differ from the "
        f"current config ({mismatch_text}). "
        f"Checkpoint feature_history_steps={stored_history}; current "
        f"feature_history_steps={expected['feature_history_steps']}. "
        "Policy input dimensions are part of the model architecture, so this "
        "checkpoint must be retrained with the current feature config or migrated "
        "with an explicit architecture conversion."
    )
