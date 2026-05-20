import pytest
import jax
import jax.numpy as jnp
import flax.linen as nn

# Import your composable system modules here
from src.jax_policy import (
    ComposablePlanetPolicy,
    MLPBackboneEncoder,
    TransformerBackboneEncoder,
    GNNBackboneEncoder,
    AutoregressivePointerDecoder,
    JaxPolicyOutput,
)

# Define standard test hyper-parameters matching your Orbit Wars environment setup
BATCH_SIZE = 2
MAX_PLANETS = 10
CANDIDATE_COUNT = 6
SHIP_BUCKET_COUNT = 5
MAX_MOVES_K = 3
HIDDEN_SIZE = 64


@pytest.fixture
def mock_inputs():
    """Generate static dummy JAX arrays matching the shapes from jax_features.py."""
    key = jax.random.PRNGKey(42)

    # Feature dimension spaces matching typical feature schemas
    self_dim = 32  # e.g., base features + context + temporal
    cand_dim = 28  # e.g., base target features + temporal
    global_dim = 45  # e.g., base global features + history deltas

    # Construct random input arrays
    k1, k2, k3 = jax.random.split(key, 3)
    self_features = jax.random.normal(k1, (BATCH_SIZE, MAX_PLANETS, self_dim))
    candidate_features = jax.random.normal(
        k2, (BATCH_SIZE, MAX_PLANETS, CANDIDATE_COUNT, cand_dim)
    )
    global_features = jax.random.normal(k3, (BATCH_SIZE, MAX_PLANETS, global_dim))

    # Ensure all elements are active initially via boolean masks
    candidate_mask = jnp.ones((BATCH_SIZE, MAX_PLANETS, CANDIDATE_COUNT), dtype=bool)

    return {
        "self_features": self_features,
        "candidate_features": candidate_features,
        "global_features": global_features,
        "candidate_mask": candidate_mask,
    }


@pytest.mark.parametrize("encoder_type", ["mlp", "transformer", "gnn"])
def test_composable_policy_shapes(encoder_type, mock_inputs):
    """Verify all pluggable encoder combinations emit correct policy/critic shapes."""
    key = jax.random.PRNGKey(0)
    init_key, run_key = jax.random.split(key)

    # Resolve requested pluggable module
    if encoder_type == "mlp":
        encoder = MLPBackboneEncoder(hidden_size=HIDDEN_SIZE)
    elif encoder_type == "transformer":
        encoder = TransformerBackboneEncoder(hidden_size=HIDDEN_SIZE, attention_heads=2)
    elif encoder_type == "gnn":
        encoder = GNNBackboneEncoder(
            hidden_size=HIDDEN_SIZE, k_neighbors=3, msg_passing_layers=1
        )

    decoder = AutoregressivePointerDecoder(
        ship_bucket_count=SHIP_BUCKET_COUNT,
        max_moves_k=MAX_MOVES_K,
        hidden_size=HIDDEN_SIZE,
    )

    # Build complete architecture container instance
    policy = ComposablePlanetPolicy(
        encoder_module=encoder, decoder_module=decoder, hidden_size=HIDDEN_SIZE
    )

    # Step 1: Verify initialization contract succeeds without errors
    params = policy.init(
        init_key,
        mock_inputs["self_features"],
        mock_inputs["candidate_features"],
        mock_inputs["global_features"],
        mock_inputs["candidate_mask"],
    )
    assert "params" in params

    # Step 2: Verify forward pass execution shapes
    output = policy.apply(
        params,
        mock_inputs["self_features"],
        mock_inputs["candidate_features"],
        mock_inputs["global_features"],
        mock_inputs["candidate_mask"],
        rng=run_key,
    )

    assert isinstance(output, JaxPolicyOutput)

    # Assert shape tracking targets hold precisely across sequence dimension K
    assert output.target_logits.shape == (BATCH_SIZE, MAX_MOVES_K, CANDIDATE_COUNT)
    assert output.ship_logits.shape == (
        BATCH_SIZE,
        MAX_MOVES_K,
        CANDIDATE_COUNT,
        SHIP_BUCKET_COUNT,
    )
    assert output.value.shape == (BATCH_SIZE,)


def test_deterministic_decoding_reproducibility(mock_inputs):
    """Ensure deterministic mode yields completely identical outputs without an RNG key."""
    encoder = MLPBackboneEncoder(hidden_size=HIDDEN_SIZE)
    decoder = AutoregressivePointerDecoder(
        ship_bucket_count=SHIP_BUCKET_COUNT,
        max_moves_k=MAX_MOVES_K,
        hidden_size=HIDDEN_SIZE,
    )
    policy = ComposablePlanetPolicy(
        encoder_module=encoder, decoder_module=decoder, hidden_size=HIDDEN_SIZE
    )

    params = policy.init(
        jax.random.PRNGKey(1),
        mock_inputs["self_features"],
        mock_inputs["candidate_features"],
        mock_inputs["global_features"],
        mock_inputs["candidate_mask"],
    )

    # Run twice with deterministic settings flag active
    out1 = policy.apply(params, **mock_inputs, deterministic=True)
    out2 = policy.apply(params, **mock_inputs, deterministic=True)

    # Ensure target choices and values are completely stable
    jnp.testing.assert_array_equal(out1.target_logits, out2.target_logits)
    jnp.testing.assert_array_equal(out1.ship_logits, out2.ship_logits)


def test_teacher_forcing_alignment(mock_inputs):
    """Verify passing a target trajectory bypasses stochastic generation steps cleanly."""
    encoder = MLPBackboneEncoder(hidden_size=HIDDEN_SIZE)
    decoder = AutoregressivePointerDecoder(
        ship_bucket_count=SHIP_BUCKET_COUNT,
        max_moves_k=MAX_MOVES_K,
        hidden_size=HIDDEN_SIZE,
    )
    policy = ComposablePlanetPolicy(
        encoder_module=encoder, decoder_module=decoder, hidden_size=HIDDEN_SIZE
    )

    params = policy.init(
        jax.random.PRNGKey(2),
        mock_inputs["self_features"],
        mock_inputs["candidate_features"],
        mock_inputs["global_features"],
        mock_inputs["candidate_mask"],
    )

    # Construct a static ground truth target sequence selection path
    # Shape must be (BATCH_SIZE, MAX_MOVES_K)
    mock_target_sequence = jnp.zeros((BATCH_SIZE, MAX_MOVES_K), dtype=jnp.int32)

    # Execute the PPO training configuration mode pass
    output = policy.apply(params, **mock_inputs, target_sequence=mock_target_sequence)

    # Confirm structural sizes align correctly back out to standard evaluation layouts
    assert output.target_logits.shape == (BATCH_SIZE, MAX_MOVES_K, CANDIDATE_COUNT)
