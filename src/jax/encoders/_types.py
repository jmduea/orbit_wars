from typing import NamedTuple

import jax


class EncoderOutput(NamedTuple):
    """Structural bridge between any encoder and any decoder.

    Fields
    ------
    attended_candidates: jax.Array
        Detailed per-planet representations
    context_query: jax.Array
        Aggregated global game state query
    value_input: jax.Array
        Combined state summary for critic head
    """

    attended_candidates: jax.Array
    context_query: jax.Array
    value_input: jax.Array
