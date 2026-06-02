from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from src.config import TrainConfig
from src.jax.action_codec import (
    PlanetFlowPolicyOutput,
    planet_flow_action_log_prob_entropy,
    planet_flow_categorical_kl,
    planet_flow_invalid_bucket_count,
    sample_planet_flow_pressure_action,
)
from src.jax.ppo_update import ppo_update_jax
from src.jax.rollout.types import (
    FactorizedActionReplay,
    JaxTransitionBatch,
    PlanetFlowActionReplay,
)


def _policy_output() -> PlanetFlowPolicyOutput:
    return PlanetFlowPolicyOutput(
        target_demand_logits=jnp.array(
            [
                [
                    [0.0, 1.0, 2.0, -1.0, -2.0],
                    [5.0, 4.0, 3.0, 2.0, 1.0],
                    [-1.0, 0.0, 1.0, 2.0, 3.0],
                ]
            ],
            dtype=jnp.float32,
        ),
        value=jnp.array([0.0], dtype=jnp.float32),
    )


def test_planet_flow_deterministic_sample_excludes_unreachable_targets() -> None:
    output = PlanetFlowPolicyOutput(
        target_demand_logits=jnp.array(
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [10.0, 9.0, 8.0, 7.0, 6.0],
                    [-1.0, 0.0, 1.0, 2.0, 3.0],
                ]
            ],
            dtype=jnp.float32,
        ),
        value=jnp.array([0.0], dtype=jnp.float32),
    )
    target_mask = jnp.array([[True, False, True]])
    sample = sample_planet_flow_pressure_action(
        jax.random.PRNGKey(0),
        output,
        jnp.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=jnp.float32),
        target_mask,
        deterministic=True,
    )

    assert sample.target_bucket.tolist() == [[4, 0, 4]]
    assert sample.target_pressure.tolist() == [[1.0, 0.0, 1.0]]


def test_planet_flow_deterministic_sample_forces_inactive_targets_to_hold() -> None:
    output = _policy_output()
    target_mask = jnp.array([[True, False, True]])
    sample = sample_planet_flow_pressure_action(
        jax.random.PRNGKey(0),
        output,
        jnp.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=jnp.float32),
        target_mask,
        deterministic=True,
    )

    assert sample.target_bucket.tolist() == [[2, 0, 4]]
    assert sample.target_pressure.tolist() == [[0.5, 0.0, 1.0]]
    assert sample.target_mask.tolist() == [[True, False, True]]
    assert jnp.isfinite(sample.log_prob).all()
    assert jnp.isfinite(sample.entropy).all()


def test_planet_flow_replay_log_prob_matches_sampled_action() -> None:
    output = _policy_output()
    target_mask = jnp.array([[True, False, True]])
    sample = sample_planet_flow_pressure_action(
        jax.random.PRNGKey(1),
        output,
        jnp.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=jnp.float32),
        target_mask,
        deterministic=False,
    )

    replay_log_prob, replay_entropy = planet_flow_action_log_prob_entropy(
        output, sample.target_bucket, target_mask
    )

    assert float(replay_log_prob[0]) == pytest.approx(float(sample.log_prob[0]))
    assert float(replay_entropy[0]) == pytest.approx(float(sample.entropy[0]))


def test_planet_flow_kl_is_zero_for_identical_masked_distributions() -> None:
    output = _policy_output()
    target_mask = jnp.array([[True, False, True]])

    kl = planet_flow_categorical_kl(output, output, target_mask)

    assert float(kl[0]) == pytest.approx(0.0, abs=1e-7)


def test_planet_flow_invalid_bucket_count_ignores_inactive_targets() -> None:
    target_bucket = jnp.array([[4, 99, 6]], dtype=jnp.int32)
    target_mask = jnp.array([[True, False, True]])

    invalid = planet_flow_invalid_bucket_count(target_bucket, 5, target_mask)

    assert int(invalid) == 1


def _minimal_shared_batch_fields() -> dict[str, jax.Array]:
    zeros_2d = jnp.zeros((1, 1), dtype=jnp.float32)
    zeros_planets = jnp.zeros((1, 1, 3), dtype=jnp.float32)
    bool_planets = jnp.ones((1, 1, 3), dtype=bool)
    zeros_edges = jnp.zeros((1, 1, 3, 2), dtype=jnp.float32)
    bool_edges = jnp.ones((1, 1, 3, 2), dtype=bool)
    zeros_ids = jnp.zeros((1, 1, 3, 2), dtype=jnp.int32)
    zeros_sequence = jnp.zeros((1, 1, 2), dtype=jnp.int32)
    float_sequence = jnp.zeros((1, 1, 2), dtype=jnp.float32)
    return {
        "planet_features": zeros_planets,
        "planet_mask": bool_planets,
        "edge_features": zeros_edges,
        "edge_mask": bool_edges,
        "edge_src_ids": zeros_ids,
        "edge_tgt_ids": zeros_ids,
        "global_features": jnp.zeros((1, 1, 4), dtype=jnp.float32),
        "theta_ref": zeros_2d,
        "player_count": jnp.full((1, 1), 2, dtype=jnp.int32),
        "returns": zeros_2d,
        "advantages": zeros_2d,
        "factorized_replay": FactorizedActionReplay(
            ship_bucket_mask=jnp.ones((1, 1, 2, 3, 2, 5), dtype=bool),
            target_index=zeros_sequence,
            ship_bucket=zeros_sequence,
            log_prob=float_sequence,
            source_index=zeros_sequence,
            target_slot=zeros_sequence,
            stop_flag=zeros_sequence,
            step_mask=float_sequence,
        ),
    }


def _minimal_transition_batch() -> JaxTransitionBatch:
    fields = _minimal_shared_batch_fields()
    return JaxTransitionBatch(
        planet_features=fields["planet_features"],
        planet_mask=fields["planet_mask"],
        edge_features=fields["edge_features"],
        edge_mask=fields["edge_mask"],
        edge_src_ids=fields["edge_src_ids"],
        edge_tgt_ids=fields["edge_tgt_ids"],
        global_features=fields["global_features"],
        theta_ref=fields["theta_ref"],
        player_count=fields["player_count"],
        returns=fields["returns"],
        advantages=fields["advantages"],
        action_replay=fields["factorized_replay"],
    )


def _minimal_planet_flow_transition_batch() -> JaxTransitionBatch:
    batch = _minimal_transition_batch()
    return batch._replace(
        action_replay=PlanetFlowActionReplay(
            target_bucket=jnp.array([[[2, 0, 4]]], dtype=jnp.int32),
            target_pressure=jnp.array([[[0.5, 0.0, 1.0]]]),
            target_mask=jnp.array([[[True, False, True]]]),
            log_prob=jnp.zeros((1, 1), dtype=jnp.float32),
        )
    )


def test_transition_batch_stores_planet_flow_replay_variant() -> None:
    batch = _minimal_transition_batch()
    pressure_batch = _minimal_planet_flow_transition_batch()

    assert isinstance(batch.action_replay, FactorizedActionReplay)
    assert isinstance(pressure_batch.action_replay, PlanetFlowActionReplay)
    replay = pressure_batch.action_replay
    assert replay.target_bucket is not None
    assert replay.target_mask is not None


def test_ppo_update_rejects_factorized_replay_under_planet_flow_config() -> None:
    cfg = TrainConfig()
    cfg.model.pointer_decoder = "planet_flow_target_heatmap"
    batch = _minimal_transition_batch()

    with pytest.raises(ValueError, match="Factorized action replay requires"):
        ppo_update_jax(
            None,  # type: ignore[arg-type]
            None,
            batch,
            cfg,
        )


def test_ppo_update_rejects_planet_flow_replay_under_factorized_config() -> None:
    cfg = TrainConfig()
    cfg.model.pointer_decoder = "factorized_topk"
    batch = _minimal_planet_flow_transition_batch()

    with pytest.raises(
        ValueError, match="Planet Flow action replay requires a planet_flow"
    ):
        ppo_update_jax(
            None,  # type: ignore[arg-type]
            None,
            batch,
            cfg,
        )


def test_ppo_update_rejects_invalid_planet_flow_pressure_bucket() -> None:
    cfg = TrainConfig()
    cfg.model.pointer_decoder = "planet_flow_target_heatmap"
    replay = _minimal_planet_flow_transition_batch().action_replay
    assert isinstance(replay, PlanetFlowActionReplay)
    batch = _minimal_planet_flow_transition_batch()._replace(
        action_replay=replay._replace(
            target_bucket=jnp.array([[[99, 0, 4]]], dtype=jnp.int32),
        )
    )

    with pytest.raises(ValueError, match="out-of-range pressure bucket"):
        ppo_update_jax(
            None,  # type: ignore[arg-type]
            None,
            batch,
            cfg,
        )
