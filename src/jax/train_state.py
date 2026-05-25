from __future__ import annotations

import optax

import jax
import jax.numpy as jnp

from src.config import TrainConfig
from src.config.schema import TaskConfig
from src.features.registry import (
    candidate_feature_dim,
    global_feature_dim,
    self_feature_dim,
)

from .rollout.types import JaxTrainState


def validate_policy_param_shapes(params: dict, env_cfg: TaskConfig) -> None:
    """Validate encoder input dimensions in Flax params against env features.

    Checks the first Dense kernel for the self/candidate/global encoder MLPs.
    Raises ValueError with expected/actual dimensions and remediation guidance
    when params are incompatible with the active environment configuration.
    """

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

    expected_dims = {
        "self_encoder": int(self_feature_dim(env_cfg)),
        "candidate_encoder": int(candidate_feature_dim(env_cfg)),
        "global_encoder": int(global_feature_dim(env_cfg)),
    }

    mismatches: list[str] = []
    for encoder_name, expected_dim in expected_dims.items():
        dense_name = f"{encoder_name}_0"
        module_payload = root.get(dense_name)
        if not isinstance(module_payload, dict):
            mismatches.append(
                f"{encoder_name}: missing module '{dense_name}' in checkpoint params"
            )
            continue
        kernel = module_payload.get("kernel")
        if kernel is None or getattr(kernel, "ndim", 0) < 1:
            mismatches.append(
                f"{encoder_name}: missing/invalid kernel at '{dense_name}.kernel'"
            )
            continue
        actual_dim = int(kernel.shape[0])
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

    dummy_self = jnp.zeros((1, self_feature_dim(cfg.task)), dtype=jnp.float32)
    dummy_candidate = jnp.zeros(
        (1, cfg.task.candidate_count, candidate_feature_dim(cfg.task)),
        dtype=jnp.float32,
    )
    dummy_global = jnp.zeros((1, global_feature_dim(cfg.task)), dtype=jnp.float32)
    dummy_mask = jnp.ones((1, cfg.task.candidate_count), dtype=bool)
    dummy_player_count = jnp.full((1,), cfg.task.player_count, dtype=jnp.int32)
    params = policy.init(
        key,
        dummy_self,
        dummy_candidate,
        dummy_global,
        dummy_mask,
        player_count=dummy_player_count,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.training.max_grad_norm),
        optax.adam(cfg.training.lr),
    )
    return JaxTrainState(
        params=params, opt_state=optimizer.init(params), optimizer=optimizer
    )
