from __future__ import annotations

import jax.numpy as jnp
import optax

import jax
from src.config import TrainConfig
from src.config.schema import TaskConfig
from src.features.registry import (
    edge_feature_dim,
    global_feature_dim,
    planet_feature_dim,
)
from src.jax.policy import make_synthetic_turn_batch

from .rollout.types import JaxTrainState


def _find_encoder_input_dim(params_root: dict, encoder_prefix: str) -> int | None:
    """Return the input dimension for ``{encoder_prefix}_0`` anywhere in params."""

    dense_name = f"{encoder_prefix}_0"
    if isinstance(params_root, dict):
        module_payload = params_root.get(dense_name)
        if isinstance(module_payload, dict):
            kernel = module_payload.get("kernel")
            if kernel is not None and getattr(kernel, "ndim", 0) >= 1:
                return int(kernel.shape[0])
        for child in params_root.values():
            if isinstance(child, dict):
                found = _find_encoder_input_dim(child, encoder_prefix)
                if found is not None:
                    return found
    return None


def validate_policy_param_shapes(params: dict, env_cfg: TaskConfig) -> None:
    """Validate encoder input dimensions in Flax params against env features."""

    if not isinstance(params, dict):
        raise ValueError(
            "Policy params must be a Flax parameter dict. Received "
            f"{type(params).__name__}."
        )
    root = params.get("params", params)
    if not isinstance(root, dict):
        raise ValueError(
            "Policy params payload is malformed: expected a 'params' mapping."
        )

    if _find_encoder_input_dim(root, "self_encoder") is not None:
        raise ValueError(
            "Loaded policy params use the legacy v1 self/candidate/global encoder "
            "layout. Retrain with the current planet-edge feature encoding or migrate "
            "the checkpoint explicitly."
        )

    expected_dims = {
        "planet_enc": int(planet_feature_dim(env_cfg)),
        "edge_enc": int(edge_feature_dim(env_cfg)),
        "global_enc": int(global_feature_dim(env_cfg)),
    }

    mismatches: list[str] = []
    for encoder_name, expected_dim in expected_dims.items():
        actual_dim = _find_encoder_input_dim(root, encoder_name)
        if actual_dim is None:
            mismatches.append(
                f"{encoder_name}: missing module '{encoder_name}_0' in checkpoint params"
            )
            continue
        if actual_dim != expected_dim:
            mismatches.append(
                f"{encoder_name}: expected input dim {expected_dim}, got {actual_dim}"
            )

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            "Loaded policy params are incompatible with the configured environment "
            f"feature dimensions ({mismatch_text}). "
            "Use a checkpoint trained with matching env/model settings or retrain."
        )


def init_train_state(key: jax.Array, policy: object, cfg: TrainConfig) -> JaxTrainState:
    """Initialize policy parameters and optimizer state for JAX PPO."""

    dummy_player_count = jnp.full((1,), cfg.task.player_count, dtype=jnp.int32)
    dummy_batch = make_synthetic_turn_batch(1, cfg.task, key=key)
    params = policy.init(
        key,
        dummy_batch,
        player_count=dummy_player_count,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.training.max_grad_norm),
        optax.adam(cfg.training.lr),
    )
    return JaxTrainState(
        params=params, opt_state=optimizer.init(params), optimizer=optimizer
    )
