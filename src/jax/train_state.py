from __future__ import annotations

import jax.numpy as jnp
import optax

import jax
from src.config import TrainConfig
from src.config.schema import TaskConfig
from src.features.registry import (
    candidate_feature_dim,
    global_feature_dim,
    self_feature_dim,
)
from src.features.registry_v2 import (
    edge_feature_dim,
    global_v2_feature_dim,
    planet_feature_dim,
)
from src.jax.policy_v2 import make_synthetic_turn_batch_v2

from .rollout.types import JaxTrainState


def uses_v2_policy_batch(cfg: TrainConfig) -> bool:
    """Return True when policy init should consume ``JaxTurnBatchV2``."""

    architecture = cfg.model.architecture.strip().lower()
    encoding_version = getattr(cfg.task, "encoding_version", "v1").strip().lower()
    return architecture == "gnn_pointer_v2" or encoding_version == "v2"


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

    encoding_version = getattr(env_cfg, "encoding_version", "v1").strip().lower()
    if (
        _find_encoder_input_dim(root, "planet_enc") is not None
        or encoding_version == "v2"
    ):
        expected_dims = {
            "planet_enc": int(planet_feature_dim(env_cfg)),
            "edge_enc": int(edge_feature_dim(env_cfg)),
            "global_enc": int(global_v2_feature_dim(env_cfg)),
        }
    else:
        expected_dims = {
            "self_encoder": int(self_feature_dim(env_cfg)),
            "candidate_encoder": int(candidate_feature_dim(env_cfg)),
            "global_encoder": int(global_feature_dim(env_cfg)),
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
    if uses_v2_policy_batch(cfg):
        dummy_batch = make_synthetic_turn_batch_v2(1, cfg.task, key=key)
        params = policy.init(
            key,
            dummy_batch,
            player_count=dummy_player_count,
        )
    else:
        dummy_self = jnp.zeros((1, self_feature_dim(cfg.task)), dtype=jnp.float32)
        dummy_candidate = jnp.zeros(
            (1, cfg.task.candidate_count, candidate_feature_dim(cfg.task)),
            dtype=jnp.float32,
        )
        dummy_global = jnp.zeros((1, global_feature_dim(cfg.task)), dtype=jnp.float32)
        dummy_mask = jnp.ones((1, cfg.task.candidate_count), dtype=bool)
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
