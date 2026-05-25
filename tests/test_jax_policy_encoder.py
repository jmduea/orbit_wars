from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jax
from src.artifacts.checkpoint_compat import (
    encoder_backbone_for_architecture,
    feature_metadata,
    validate_checkpoint_encoder_compatibility,
)
from src.config import TrainConfig
from src.config.schema import TaskConfig
from src.features.catalog.planet import PLANET_FEATURE_CATALOG
from src.features.registry import edge_k, planet_feature_dim
from src.game.constants import MAX_PLANETS
from src.jax.policy import (
    build_gnn_pointer_policy,
    build_jax_policy,
    build_planet_graph_transformer_policy,
    make_synthetic_turn_batch,
)


def _task_cfg(**kwargs) -> TaskConfig:
    base = dict(candidate_count=4, ship_bucket_count=8, max_fleets=16)
    base.update(kwargs)
    return TaskConfig(**base)


def _train_cfg(*, architecture: str) -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.architecture = architecture
    cfg.model.hidden_size = 64
    cfg.model.attention_heads = 4
    cfg.model.planet_transformer_layers = 2
    cfg.model.gnn_k_neighbors = 3
    cfg.model.gnn_message_passing_layers = 1
    cfg.task = _task_cfg()
    return cfg


def _param_key_paths(params: object) -> set[str]:
    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    return {"/".join(str(k) for k in path) for path, _ in flat}


def _init_and_apply(policy, cfg: TrainConfig, batch_size: int = 2):
    batch = make_synthetic_turn_batch(batch_size, cfg.task, key=jax.random.PRNGKey(0))
    params = policy.init(jax.random.PRNGKey(1), batch)
    output = policy.apply(params, batch, deterministic=True)
    return params, batch, output


def test_planet_graph_transformer_forward_shapes() -> None:
    cfg = _train_cfg(architecture="planet_graph_transformer")
    policy = build_planet_graph_transformer_policy(cfg)
    _, batch, output = _init_and_apply(policy, cfg)

    k_slots = edge_k(cfg.task)
    edge_count = MAX_PLANETS * k_slots + 1
    assert output.target_logits.shape == (2, cfg.model.max_moves_k, edge_count)
    assert output.ship_logits.shape == (
        2,
        cfg.model.max_moves_k,
        edge_count,
        cfg.task.ship_bucket_count,
    )
    assert output.value.shape == (2,)
    assert batch.planet_features.shape[1] == MAX_PLANETS


def test_gnn_pointer_forward_shapes_after_tgt_fusion() -> None:
    cfg = _train_cfg(architecture="gnn_pointer")
    policy = build_gnn_pointer_policy(cfg)
    _, _, output = _init_and_apply(policy, cfg)
    assert output.value.shape == (2,)


def test_transformer_layer_depth_changes_param_tree() -> None:
    cfg_one = _train_cfg(architecture="planet_graph_transformer")
    cfg_one.model.planet_transformer_layers = 1
    cfg_two = _train_cfg(architecture="planet_graph_transformer")
    cfg_two.model.planet_transformer_layers = 2

    batch = make_synthetic_turn_batch(1, cfg_one.task)
    params_one = build_planet_graph_transformer_policy(cfg_one).init(
        jax.random.PRNGKey(0), batch
    )
    params_two = build_planet_graph_transformer_policy(cfg_two).init(
        jax.random.PRNGKey(0), batch
    )

    keys_one = _param_key_paths(params_one)
    keys_two = _param_key_paths(params_two)
    assert not any("planet_tx_attn_1" in key for key in keys_one)
    assert any("planet_tx_attn_1" in key for key in keys_two)


def test_all_masked_planets_produce_finite_outputs() -> None:
    cfg = _train_cfg(architecture="planet_graph_transformer")
    policy = build_planet_graph_transformer_policy(cfg)
    batch = make_synthetic_turn_batch(1, cfg.task)
    batch = batch._replace(
        planet_mask=jnp.zeros((1, MAX_PLANETS), dtype=bool),
        edge_mask=jnp.zeros((1, MAX_PLANETS, edge_k(cfg.task)), dtype=bool),
    )
    params = policy.init(jax.random.PRNGKey(0), batch)
    output = policy.apply(params, batch, deterministic=True)
    assert np.isfinite(np.asarray(output.value)).all()
    assert np.isfinite(np.asarray(output.target_logits)).all()


def test_spatial_bias_prefers_closer_planets() -> None:
    from src.jax.encoders.planet_encoder_common import (
        planet_orbit_coords,
        planet_pairwise_spatial_bias,
    )

    dim = planet_feature_dim(_task_cfg())
    radius_slice = PLANET_FEATURE_CATALOG.base_slice("orbit_radius")
    planet_features = jnp.zeros((1, 3, dim), dtype=jnp.float32)
    planet_features = planet_features.at[0, 0, radius_slice].set(1.0)
    planet_features = planet_features.at[0, 1, radius_slice].set(2.0)
    planet_features = planet_features.at[0, 2, radius_slice].set(10.0)
    coords = planet_orbit_coords(planet_features)
    bias = planet_pairwise_spatial_bias(coords)[0, 0, 0]
    assert float(bias[1]) > float(bias[2])


def test_build_jax_policy_dispatches_transformer() -> None:
    cfg = _train_cfg(architecture="planet_graph_transformer")
    policy = build_jax_policy(cfg)
    assert policy.__class__.__name__ == "ComposablePlanetPolicy"


def test_encoder_backbone_metadata_mapping() -> None:
    assert encoder_backbone_for_architecture("gnn_pointer") == "planet_gnn"
    assert (
        encoder_backbone_for_architecture("planet_graph_transformer")
        == "planet_self_attention"
    )


def test_validate_rejects_encoder_backbone_mismatch() -> None:
    cfg = _train_cfg(architecture="planet_graph_transformer")
    stored = dict(feature_metadata(cfg.task, model_cfg=cfg.model))
    stored["encoder_backbone"] = "planet_gnn"
    with pytest.raises(ValueError, match="encoder_backbone"):
        validate_checkpoint_encoder_compatibility(stored, cfg)


def test_transformer_hidden_size_must_divide_heads() -> None:
    cfg = _train_cfg(architecture="planet_graph_transformer")
    cfg.model.hidden_size = 63
    with pytest.raises(ValueError, match="divisible"):
        build_planet_graph_transformer_policy(cfg)
