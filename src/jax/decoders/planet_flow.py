from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

import jax
from src.jax.encoders.planet_encoder_common import PlanetEdgeEncoderOutput


class PlanetFlowTargetDemandHead(nn.Module):
    """Per-target pressure-bucket logits for the Planet Flow P0 proof slice."""

    pressure_bucket_count: int
    hidden_size: int = 128

    @nn.compact
    def __call__(self, encoder_out: PlanetEdgeEncoderOutput) -> jax.Array:
        context = jnp.broadcast_to(
            encoder_out.context_query[:, None, :],
            encoder_out.planet_states.shape[:-1]
            + (encoder_out.context_query.shape[-1],),
        )
        joint = jnp.concatenate([encoder_out.planet_states, context], axis=-1)
        hidden = nn.relu(nn.Dense(self.hidden_size, name="target_demand_dense")(joint))
        logits = nn.Dense(
            self.pressure_bucket_count,
            name="target_demand_out",
        )(hidden)
        return jnp.where(encoder_out.planet_mask[..., None], logits, 0.0)
