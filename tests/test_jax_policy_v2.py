import jax.numpy as jnp
import pytest

import jax
from src.config import TrainConfig
from src.features.registry_v2 import edge_k
from src.game.constants import MAX_PLANETS
from src.jax.policy import build_jax_policy, sample_actions
from src.jax.policy_v2 import (
    PlanetEdgeBackboneEncoder,
    build_gnn_pointer_v2_policy,
    edge_action_count,
    make_synthetic_turn_batch_v2,
)
from src.jax.train_state import (
    _find_encoder_input_dim,
    init_train_state,
    uses_v2_policy_batch,
    validate_policy_param_shapes,
)

BATCH_SIZE = 2
HIDDEN_SIZE = 64
MAX_MOVES_K = 3
SHIP_BUCKET_COUNT = 5


@pytest.fixture
def v2_cfg() -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.architecture = "gnn_pointer_v2"
    cfg.task.encoding_version = "v2"
    cfg.task.candidate_count = 4
    cfg.task.ship_bucket_count = SHIP_BUCKET_COUNT
    cfg.model.hidden_size = HIDDEN_SIZE
    cfg.model.max_moves_k = MAX_MOVES_K
    cfg.model.gnn_k_neighbors = 3
    cfg.model.gnn_message_passing_layers = 1
    return cfg


@pytest.fixture
def v2_batch(v2_cfg: TrainConfig):
    return make_synthetic_turn_batch_v2(
        BATCH_SIZE, v2_cfg.task, key=jax.random.PRNGKey(7)
    )


def test_build_gnn_pointer_v2_policy(v2_cfg: TrainConfig):
    policy = build_jax_policy(v2_cfg)
    assert policy.encoder_module.k_neighbors == v2_cfg.model.gnn_k_neighbors
    assert (
        policy.encoder_module.msg_passing_layers
        == v2_cfg.model.gnn_message_passing_layers
    )
    assert policy.edge_k == edge_k(v2_cfg.task)


def test_gnn_pointer_v2_forward_shapes(v2_cfg: TrainConfig, v2_batch):
    policy = build_gnn_pointer_v2_policy(v2_cfg)
    init_key, run_key = jax.random.split(jax.random.PRNGKey(0))
    params = policy.init(init_key, v2_batch)
    output = policy.apply(params, v2_batch, rng=run_key)

    edge_count = edge_action_count(v2_cfg.task)
    assert output.target_logits.shape == (BATCH_SIZE, MAX_MOVES_K, edge_count)
    assert output.ship_logits.shape == (
        BATCH_SIZE,
        MAX_MOVES_K,
        edge_count,
        SHIP_BUCKET_COUNT,
    )
    assert output.value.shape == (BATCH_SIZE,)
    assert output.decoded_target_sequence.shape == (BATCH_SIZE, MAX_MOVES_K)


def test_gnn_pointer_v2_sample_actions(v2_cfg: TrainConfig, v2_batch):
    policy = build_gnn_pointer_v2_policy(v2_cfg)
    init_key, run_key, sample_key = jax.random.split(jax.random.PRNGKey(1), 3)
    params = policy.init(init_key, v2_batch)
    output = policy.apply(params, v2_batch, rng=run_key)
    target_index, ship_bucket, log_prob, entropy = sample_actions(sample_key, output)

    assert target_index.shape == (BATCH_SIZE, MAX_MOVES_K)
    assert ship_bucket.shape == (BATCH_SIZE, MAX_MOVES_K)
    assert log_prob.shape == (BATCH_SIZE, MAX_MOVES_K)
    assert entropy.shape == (BATCH_SIZE, MAX_MOVES_K)
    assert jnp.all(target_index >= 0)
    assert jnp.all(target_index < edge_action_count(v2_cfg.task))


def test_init_train_state_v2_dummy_batch(v2_cfg: TrainConfig):
    policy = build_jax_policy(v2_cfg)
    train_state = init_train_state(jax.random.PRNGKey(2), policy, v2_cfg)
    validate_policy_param_shapes(train_state.params, v2_cfg.task)
    assert _find_encoder_input_dim(train_state.params["params"], "planet_enc") == 13


def test_uses_v2_policy_batch_by_architecture(v2_cfg: TrainConfig):
    assert uses_v2_policy_batch(v2_cfg) is True
    v2_cfg.task.encoding_version = "v1"
    assert uses_v2_policy_batch(v2_cfg) is True


def test_planet_edge_encoder_output_contract(v2_cfg: TrainConfig, v2_batch):
    encoder = PlanetEdgeBackboneEncoder(
        hidden_size=HIDDEN_SIZE,
        k_neighbors=3,
        msg_passing_layers=1,
        planet_feature_dim=13,
        edge_feature_dim=12,
        global_feature_dim=46,
        edge_k=edge_k(v2_cfg.task),
    )
    params = encoder.init(jax.random.PRNGKey(3), v2_batch)
    encoder_out = encoder.apply(params, v2_batch)

    k_slots = edge_k(v2_cfg.task)
    assert encoder_out.attended_edges.shape == (
        BATCH_SIZE,
        MAX_PLANETS * k_slots,
        HIDDEN_SIZE,
    )
    assert encoder_out.edge_action_mask.shape == (BATCH_SIZE, MAX_PLANETS * k_slots)
    assert encoder_out.context_query.shape == (BATCH_SIZE, HIDDEN_SIZE)
    assert encoder_out.value_input.shape[0] == BATCH_SIZE
