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
from src.jax.encoders.planet_encoder_common import (
    planet_attention_mask_with_bias,
    planet_orbit_coords,
    planet_pairwise_spatial_bias,
    planet_self_attention_mask,
)
from src.jax.policy import (
    build_jax_policy,
    build_planet_graph_transformer_policy,
    make_synthetic_turn_batch,
)


def _task_cfg(**kwargs) -> TaskConfig:
    base = dict(candidate_count=4, ship_bucket_count=8, max_fleets=16)
    base.update(kwargs)
    return TaskConfig(**base)


def _train_cfg(
    *, architecture: str, pointer_decoder: str = "factorized_topk"
) -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.architecture = architecture
    cfg.model.pointer_decoder = pointer_decoder
    cfg.model.hidden_size = 64
    cfg.model.attention_heads = 4
    cfg.model.planet_transformer_layers = 2
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
    assert output.source_logits.shape == (2, cfg.model.max_moves_k, MAX_PLANETS)
    assert output.target_logits.shape == (2, cfg.model.max_moves_k, k_slots)
    assert output.ship_logits.shape == (
        2,
        cfg.model.max_moves_k,
        k_slots,
        cfg.task.ship_bucket_count,
    )
    assert output.value.shape == (2,)
    assert batch.planet_features.shape[1] == MAX_PLANETS


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
    cfg = _train_cfg(
        architecture="planet_graph_transformer", pointer_decoder="factorized_topk"
    )
    policy = build_jax_policy(cfg)
    assert policy.__class__.__name__ == "ComposableFactorizedPlanetPolicy"


def test_build_jax_policy_dispatches_planet_graph_transformer_small() -> None:
    cfg = _train_cfg(
        architecture="planet_graph_transformer_small", pointer_decoder="factorized_topk"
    )
    policy = build_jax_policy(cfg)
    assert policy.__class__.__name__ == "ComposableFactorizedPlanetPolicy"
    assert (
        encoder_backbone_for_architecture("planet_graph_transformer_small")
        == "planet_self_attention"
    )


@pytest.mark.jax
def test_scan_decode_step_logits_match_prefix_forward() -> None:
    """Incremental step logits must match prefix-forward oracle at each step_idx."""
    from src.jax.factored_decode_scan import (
        advance_scan_decode_carry,
        init_scan_decode_carry,
        scan_decode_step,
    )
    from src.jax.factored_sequence_scan import (
        build_shield_prefix_teacher_sequences,
        forward_factorized_encode,
        forward_factored_policy,
    )
    from src.jax.policy import factorized_decode

    cfg = _train_cfg(architecture="planet_graph_transformer")
    cfg.model.max_moves_k = 3
    cfg.model.decoder_carry = True
    policy = build_jax_policy(cfg)
    batch = make_synthetic_turn_batch(2, cfg.task, key=jax.random.PRNGKey(10))
    params = policy.init(jax.random.PRNGKey(11), batch)
    encoder_out = forward_factorized_encode(params, policy, batch)
    source_seq = jnp.array([[0, 1, 0], [0, 0, 1]], dtype=jnp.int32)
    slot_seq = jnp.array([[0, 1, 2], [1, 0, 0]], dtype=jnp.int32)
    player_count = jnp.full((2,), cfg.task.player_count, dtype=jnp.int32)

    carry = init_scan_decode_carry(params, policy, encoder_out, cfg)
    for step_idx in range(cfg.model.max_moves_k):
        source_prefix, target_prefix = build_shield_prefix_teacher_sequences(
            source_seq, slot_seq, step_idx
        )
        prefix_out = forward_factored_policy(
            params,
            policy,
            batch,
            cfg,
            player_count=player_count,
            source_sequence=source_prefix,
            target_slot_sequence=target_prefix,
            deterministic=True,
            encoder_out=encoder_out,
        )
        step_logits, carry = scan_decode_step(
            params,
            policy,
            encoder_out,
            carry,
            teacher_source=source_prefix[:, step_idx],
            teacher_target_slot=target_prefix[:, step_idx],
            deterministic=True,
        )
        np.testing.assert_allclose(
            step_logits.source_logits,
            prefix_out.source_logits[:, step_idx, :],
            rtol=0.0,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            step_logits.target_logits,
            prefix_out.target_logits[:, step_idx, :],
            rtol=0.0,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            step_logits.stop_logits,
            prefix_out.stop_logits[:, step_idx],
            rtol=0.0,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            step_logits.ship_logits,
            prefix_out.ship_logits[:, step_idx, :, :],
            rtol=0.0,
            atol=1e-5,
        )
        carry = advance_scan_decode_carry(
            encoder_out,
            carry,
            source=source_seq[:, step_idx],
            target_slot=slot_seq[:, step_idx],
        )

    full = factorized_decode(
        params,
        policy,
        encoder_out,
        source_sequence=source_seq,
        target_slot_sequence=slot_seq,
        deterministic=True,
    )
    assert carry is not None
    np.testing.assert_allclose(carry.state, full.decoder_hidden, rtol=0.0, atol=1e-5)


@pytest.mark.jax
def test_incremental_factorized_decode_matches_full_teacher_path() -> None:
    """Rollout incremental decode must match full factorized_decode on teacher prefix."""
    from src.jax.factored_sequence_scan import forward_factorized_encode
    from src.jax.policy import (
        factorized_decode,
        factorized_decode_advance_carry,
        factorized_decode_init_carry,
        factorized_decode_step,
    )

    cfg = _train_cfg(architecture="planet_graph_transformer")
    cfg.model.max_moves_k = 3
    cfg.model.decoder_carry = True
    policy = build_jax_policy(cfg)
    batch = make_synthetic_turn_batch(2, cfg.task, key=jax.random.PRNGKey(0))
    params = policy.init(jax.random.PRNGKey(1), batch)
    encoder_out = forward_factorized_encode(params, policy, batch)
    source_seq = jnp.array([[0, 1, 0], [0, 0, 1]], dtype=jnp.int32)
    slot_seq = jnp.array([[0, 1, 2], [1, 0, 0]], dtype=jnp.int32)

    full = factorized_decode(
        params,
        policy,
        encoder_out,
        source_sequence=source_seq,
        target_slot_sequence=slot_seq,
        deterministic=True,
    )

    carry = factorized_decode_init_carry(params, policy, encoder_out)
    for step_idx in range(cfg.model.max_moves_k):
        _, carry = factorized_decode_step(
            params,
            policy,
            encoder_out,
            carry,
            teacher_source=source_seq[:, step_idx],
            teacher_target_slot=slot_seq[:, step_idx],
            deterministic=True,
        )
        carry = factorized_decode_advance_carry(
            params,
            policy,
            encoder_out,
            carry,
            source=source_seq[:, step_idx],
            target_slot=slot_seq[:, step_idx],
        )

    np.testing.assert_allclose(carry.state, full.decoder_hidden, rtol=0.0, atol=1e-5)


@pytest.mark.jax
def test_teacher_carry_replay_matches_full_factorized_decode() -> None:
    """Lightweight carry replay must match full decode decoder_hidden export."""
    from src.jax.factored_decode_scan import (
        advance_scan_decode_carry,
        init_scan_decode_carry,
        scan_decode_step,
    )
    from src.jax.factored_sequence_scan import forward_factorized_encode
    from src.jax.policy import factorized_decode

    cfg = _train_cfg(architecture="planet_graph_transformer")
    cfg.model.max_moves_k = 3
    cfg.model.decoder_carry = True
    policy = build_jax_policy(cfg)
    batch = make_synthetic_turn_batch(2, cfg.task, key=jax.random.PRNGKey(3))
    params = policy.init(jax.random.PRNGKey(4), batch)
    encoder_out = forward_factorized_encode(params, policy, batch)
    source_seq = jnp.array([[0, 1, 0], [0, 0, 1]], dtype=jnp.int32)
    slot_seq = jnp.array([[0, 1, 2], [1, 0, 0]], dtype=jnp.int32)

    full = factorized_decode(
        params,
        policy,
        encoder_out,
        source_sequence=source_seq,
        target_slot_sequence=slot_seq,
        deterministic=True,
    )
    carry = init_scan_decode_carry(params, policy, encoder_out, cfg)
    for step_idx in range(cfg.model.max_moves_k):
        _, carry = scan_decode_step(
            params,
            policy,
            encoder_out,
            carry,
            deterministic=True,
        )
        carry = advance_scan_decode_carry(
            encoder_out,
            carry,
            source=source_seq[:, step_idx],
            target_slot=slot_seq[:, step_idx],
        )
    np.testing.assert_allclose(carry.state, full.decoder_hidden, rtol=0.0, atol=1e-5)


@pytest.mark.jax
def test_advance_scan_decode_carry_matches_decoder_advance() -> None:
    """Scan carry advance must match FactorizedTopKPointerDecoder.advance_carry_input."""
    from src.jax.factored_decode_scan import (
        advance_scan_decode_carry,
        init_scan_decode_carry,
    )
    from src.jax.factored_sequence_scan import forward_factorized_encode
    from src.jax.policy import factorized_decode_advance_carry

    cfg = _train_cfg(architecture="planet_graph_transformer")
    cfg.model.max_moves_k = 3
    cfg.model.decoder_carry = True
    policy = build_jax_policy(cfg)
    batch = make_synthetic_turn_batch(2, cfg.task, key=jax.random.PRNGKey(5))
    params = policy.init(jax.random.PRNGKey(6), batch)
    encoder_out = forward_factorized_encode(params, policy, batch)
    rng = jax.random.PRNGKey(7)
    source = jax.random.randint(rng, (2,), 0, MAX_PLANETS, dtype=jnp.int32)
    target_slot = jax.random.randint(
        jax.random.fold_in(rng, 1), (2,), 0, edge_k(cfg.task), dtype=jnp.int32
    )

    carry = init_scan_decode_carry(params, policy, encoder_out, cfg)
    scan_carry = advance_scan_decode_carry(
        encoder_out,
        carry,
        source=source,
        target_slot=target_slot,
    )
    decoder_carry = factorized_decode_advance_carry(
        params,
        policy,
        encoder_out,
        carry,
        source=source,
        target_slot=target_slot,
    )

    np.testing.assert_allclose(
        scan_carry.input_emb,
        decoder_carry.input_emb,
        rtol=0.0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        scan_carry.state,
        decoder_carry.state,
        rtol=0.0,
        atol=1e-5,
    )


def test_build_jax_policy_dispatches_factorized_transformer() -> None:
    cfg = _train_cfg(
        architecture="planet_graph_transformer", pointer_decoder="factorized_topk"
    )
    policy = build_jax_policy(cfg)
    assert policy.__class__.__name__ == "ComposableFactorizedPlanetPolicy"


def test_encoder_backbone_metadata_mapping() -> None:
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


def test_planet_self_attention_mask_excludes_inactive_planets() -> None:
    """Active queries must not attend to inactive planet keys (F1 regression guard)."""
    planet_mask = jnp.array([[True, True, False, False]], dtype=bool)
    attn = planet_self_attention_mask(planet_mask)
    assert bool(attn[0, 0, 1])
    assert not bool(attn[0, 0, 2])
    assert not bool(attn[0, 1, 2])


def test_inactive_planet_coords_do_not_change_active_attention_bias() -> None:
    """Spatial bias for active pairs ignores perturbations on inactive planet rows."""
    cfg = _train_cfg(architecture="planet_graph_transformer")
    batch = make_synthetic_turn_batch(1, cfg.task, key=jax.random.PRNGKey(0))
    active_mask = jnp.zeros((1, MAX_PLANETS), dtype=bool)
    active_mask = active_mask.at[0, :2].set(True)
    batch_base = batch._replace(planet_mask=active_mask)
    batch_perturbed = batch_base._replace(
        planet_features=batch.planet_features.at[0, 2:].set(999.0)
    )
    coords_base = planet_orbit_coords(batch_base.planet_features)
    coords_perturbed = planet_orbit_coords(batch_perturbed.planet_features)
    bias_base = planet_attention_mask_with_bias(
        batch_base.planet_mask, coords_base, spatial_attention_bias=True
    )
    bias_perturbed = planet_attention_mask_with_bias(
        batch_perturbed.planet_mask, coords_perturbed, spatial_attention_bias=True
    )
    np.testing.assert_allclose(
        bias_base[0, 0, 0, :2], bias_perturbed[0, 0, 0, :2], rtol=0.0, atol=1e-6
    )
