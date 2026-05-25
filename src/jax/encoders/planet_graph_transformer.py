"""Masked planet self-attention encoder for ``TurnBatch`` inputs."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from src.jax.encoders.planet_encoder_common import (
    PlanetEdgeEncoderOutput,
    finalize_planet_edge_encoder_output,
    fuse_source_target_edges,
    mlp,
    planet_attention_mask_with_bias,
    planet_orbit_coords,
)
from src.jax.encoders.remat import remat_if
from src.jax.features import TurnBatch


class PlanetTransformerBlock(nn.Module):
    """Single transformer block for ``PlanetGraphTransformerEncoder``."""

    hidden_size: int
    attention_heads: int
    layer_idx: int

    @nn.compact
    def __call__(
        self,
        current_planet_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        normed = nn.LayerNorm(name=f"planet_tx_norm_attn_{self.layer_idx}")(
            current_planet_states
        )
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.attention_heads,
            qkv_features=self.hidden_size,
            out_features=self.hidden_size,
            name=f"planet_tx_attn_{self.layer_idx}",
        )(
            inputs_q=normed,
            inputs_k=normed,
            inputs_v=normed,
            mask=attention_mask,
            deterministic=True,
        )
        current_planet_states = current_planet_states + attn_out

        normed = nn.LayerNorm(name=f"planet_tx_norm_ffn_{self.layer_idx}")(
            current_planet_states
        )
        ffn_out = mlp(
            normed,
            self.hidden_size,
            self.hidden_size,
            f"planet_tx_ffn_{self.layer_idx}",
        )
        return current_planet_states + ffn_out


class PlanetGraphTransformerEncoder(nn.Module):
    """Planet graph transformer with spatial attention bias and tgt-aware edge fusion."""

    hidden_size: int = 128
    attention_heads: int = 4
    planet_transformer_layers: int = 2
    spatial_attention_bias: bool = True
    planet_feature_dim: int = 13
    edge_feature_dim: int = 12
    global_feature_dim: int = 46
    edge_k: int = 3
    gradient_checkpointing: bool = False

    def setup(self) -> None:
        if self.planet_transformer_layers < 1:
            raise ValueError("planet_transformer_layers must be at least 1.")
        if self.edge_k < 0:
            raise ValueError("edge_k must be non-negative.")
        if self.hidden_size % self.attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by attention_heads for planet self-attention."
            )

    @nn.compact
    def __call__(self, batch: TurnBatch) -> PlanetEdgeEncoderOutput:
        planet_mask = batch.planet_mask.astype(bool)
        edge_mask = batch.edge_mask.astype(bool)

        planet_hidden = mlp(
            batch.planet_features,
            self.hidden_size,
            self.hidden_size,
            "planet_enc",
        )
        global_hidden = mlp(
            batch.global_features,
            self.hidden_size,
            self.hidden_size,
            "global_enc",
        )

        if self.edge_k == 0:
            edge_hidden = jnp.zeros(
                (
                    batch.planet_features.shape[0],
                    batch.planet_features.shape[1],
                    0,
                    self.hidden_size,
                ),
                dtype=jnp.float32,
            )
        else:
            edge_hidden = mlp(
                batch.edge_features,
                self.hidden_size,
                self.hidden_size,
                "edge_enc",
            )

        coords = planet_orbit_coords(batch.planet_features)
        attention_mask = planet_attention_mask_with_bias(
            planet_mask,
            coords,
            spatial_attention_bias=self.spatial_attention_bias,
        )

        current_planet_states = planet_hidden
        block_cls = remat_if(PlanetTransformerBlock, self.gradient_checkpointing)
        for layer_idx in range(self.planet_transformer_layers):
            current_planet_states = block_cls(
                hidden_size=self.hidden_size,
                attention_heads=self.attention_heads,
                layer_idx=layer_idx,
                name=f"planet_tx_block_{layer_idx}",
            )(current_planet_states, attention_mask)

        if self.edge_k > 0:
            edge_hidden = fuse_source_target_edges(
                current_planet_states,
                edge_hidden,
                batch,
                hidden_size=self.hidden_size,
            )

        return finalize_planet_edge_encoder_output(
            current_planet_states=current_planet_states,
            global_hidden=global_hidden,
            edge_hidden=edge_hidden,
            edge_mask=edge_mask,
            planet_mask=planet_mask,
            hidden_size=self.hidden_size,
            edge_k=self.edge_k,
        )
