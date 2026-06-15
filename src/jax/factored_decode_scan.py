"""O(K) factorized decoder scan helpers for rollout sampling and PPO replay."""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

import jax
from src.config import TrainConfig
from src.jax.decoder_carry import decoder_carry_enabled
from src.jax.decoders.factorized_topk_pointer import (
    FactorizedDecodeCarry,
    FactorizedStepLogits,
)
from src.jax.encoders.planet_encoder_common import PlanetEdgeEncoderOutput
from src.jax.policy import factorized_decode_init_carry, factorized_decode_step


def advance_scan_decode_carry(
    encoder_out: PlanetEdgeEncoderOutput,
    carry: FactorizedDecodeCarry,
    *,
    source: jax.Array,
    target_slot: jax.Array,
) -> FactorizedDecodeCarry:
    """Advance decoder input embedding from a committed (source, target_slot) launch."""

    batch_size = encoder_out.context_query.shape[0]
    edge_hidden = encoder_out.edge_hidden
    batch_indices = jnp.arange(batch_size, dtype=jnp.int32)
    chosen_edges = edge_hidden[batch_indices, source, :, :]
    chosen_edge_emb = jnp.take_along_axis(
        chosen_edges,
        target_slot[:, None, None],
        axis=1,
    ).squeeze(1)
    return carry._replace(input_emb=chosen_edge_emb)


def init_scan_decode_carry(
    params: dict,
    policy: nn.Module,
    encoder_out: PlanetEdgeEncoderOutput,
    cfg: TrainConfig,
    decoder_hidden_in: jax.Array | None = None,
) -> FactorizedDecodeCarry:
    """Initial decoder carry for one factorized K-step scan."""

    hidden = decoder_hidden_in if decoder_carry_enabled(cfg) else None
    return factorized_decode_init_carry(
        params,
        policy,
        encoder_out,
        decoder_hidden=hidden,
    )


def scan_decode_step(
    params: dict,
    policy: nn.Module,
    encoder_out: PlanetEdgeEncoderOutput,
    carry: FactorizedDecodeCarry,
    *,
    teacher_source: jax.Array | None = None,
    teacher_target_slot: jax.Array | None = None,
    rng: jax.Array | None = None,
    deterministic: bool = False,
) -> tuple[FactorizedStepLogits, FactorizedDecodeCarry]:
    """Run one autoregressive decoder step on cached encoder output."""

    return factorized_decode_step(
        params,
        policy,
        encoder_out,
        carry,
        teacher_source=teacher_source,
        teacher_target_slot=teacher_target_slot,
        rng=rng,
        deterministic=deterministic,
    )
