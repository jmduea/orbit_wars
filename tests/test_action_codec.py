import jax.numpy as jnp
import numpy as np

from src.jax.action_codec import (
    decode_flat_edge,
    flat_edge_index,
    noop_edge_index,
)


def test_flat_edge_index_roundtrip() -> None:
    k = 7
    for src_row in (0, 5, 59):
        for slot in (0, 3, k - 1):
            flat = int(flat_edge_index(jnp.int32(src_row), jnp.int32(slot), k))
            decoded_src, decoded_slot = decode_flat_edge(jnp.int32(flat), k)
            np.testing.assert_array_equal(np.asarray(decoded_src), src_row)
            np.testing.assert_array_equal(np.asarray(decoded_slot), slot)


def test_noop_edge_index() -> None:
    assert noop_edge_index(7, max_planets=60) == 420
