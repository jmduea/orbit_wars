"""Factorized source → target-slot pointer decoder for top-K edges."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

import jax
from src.jax.encoders.planet_encoder_common import PlanetEdgeEncoderOutput


class FactorizedTopKPointerDecoder(nn.Module):
    """Autoregressive factorized pointer: source planet, target slot, bucket, stop."""

    ship_bucket_count: int
    max_moves_k: int
    hidden_size: int = 128
    edge_k: int = 3

    @nn.compact
    def __call__(
        self,
        encoder_out: PlanetEdgeEncoderOutput,
        *,
        source_sequence: jax.Array | None = None,
        target_slot_sequence: jax.Array | None = None,
        rng: jax.Array | None = None,
        deterministic: bool = False,
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        batch_size = encoder_out.context_query.shape[0]
        planet_states = encoder_out.planet_states
        edge_hidden = encoder_out.edge_hidden
        edge_mask = encoder_out.edge_mask.astype(bool)
        planet_mask = encoder_out.planet_mask.astype(bool)
        source_mask = planet_mask & edge_mask.any(axis=-1)

        decoder_cell = nn.GRUCell(features=self.hidden_size, name="factorized_dec_gru")
        init_decoder_state = nn.Dense(
            self.hidden_size, name="factorized_init_dec_state"
        )(encoder_out.context_query)
        src_q_dense = nn.Dense(self.hidden_size, name="factorized_src_q")
        src_k_dense = nn.Dense(self.hidden_size, name="factorized_src_k")
        tgt_q_dense = nn.Dense(self.hidden_size, name="factorized_tgt_q")
        tgt_k_dense = nn.Dense(self.hidden_size, name="factorized_tgt_k")
        stop_dense = nn.Dense(1, name="factorized_stop")
        ship_dense = nn.Dense(self.hidden_size, name="factorized_ship_dense")
        ship_out = nn.Dense(self.ship_bucket_count, name="factorized_ship_out")

        start_token = self.param(
            "factorized_start_token",
            nn.initializers.zeros,
            (self.hidden_size,),
        )
        current_input_emb = jnp.broadcast_to(
            start_token[None, :], (batch_size, self.hidden_size)
        )

        illegal_logit = jnp.finfo(jnp.float32).min
        scale = jnp.sqrt(jnp.asarray(self.hidden_size, dtype=jnp.float32))
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)

        (
            all_source_logits,
            all_target_logits,
            all_stop_logits,
            all_ship_logits,
            all_sources,
            all_target_slots,
            all_stops,
        ) = ([], [], [], [], [], [], [])

        current_state = init_decoder_state
        current_rng = rng

        for step_idx in range(self.max_moves_k):
            current_state, _ = decoder_cell(current_state, current_input_emb)

            step_stop_logit = stop_dense(current_state).squeeze(-1)
            all_stop_logits.append(step_stop_logit)

            src_q = src_q_dense(current_state)[:, None, :]
            src_k = src_k_dense(planet_states)
            step_source_logits = jnp.einsum("b1h,bph->bp", src_q, src_k) / scale
            step_source_logits = jnp.where(
                source_mask, step_source_logits, illegal_logit
            )
            all_source_logits.append(step_source_logits)

            if source_sequence is not None:
                chosen_source = source_sequence[:, step_idx]
            elif deterministic or current_rng is None:
                chosen_source = jnp.argmax(step_source_logits, axis=-1)
            else:
                step_rng, current_rng = jax.random.split(current_rng)
                chosen_source = jax.random.categorical(
                    step_rng, step_source_logits, axis=-1
                )
            all_sources.append(chosen_source)

            chosen_edges = edge_hidden[batch_indices, chosen_source, :, :]
            chosen_edge_mask = edge_mask[batch_indices, chosen_source, :]

            tgt_q = tgt_q_dense(current_state)[:, None, :]
            tgt_k = tgt_k_dense(chosen_edges)
            step_target_logits = jnp.einsum("b1h,bkh->bk", tgt_q, tgt_k) / scale
            step_target_logits = jnp.where(
                chosen_edge_mask, step_target_logits, illegal_logit
            )
            all_target_logits.append(step_target_logits)

            expanded_state = jnp.broadcast_to(
                current_state[:, None, :], (*chosen_edges.shape[:2], self.hidden_size)
            )
            ship_input = jnp.concatenate([expanded_state, chosen_edges], axis=-1)
            step_ship_logits = ship_out(nn.relu(ship_dense(ship_input)))
            all_ship_logits.append(step_ship_logits)

            if target_slot_sequence is not None:
                chosen_target_slot = target_slot_sequence[:, step_idx]
            elif deterministic or current_rng is None:
                chosen_target_slot = jnp.argmax(step_target_logits, axis=-1)
            else:
                step_rng, current_rng = jax.random.split(current_rng)
                chosen_target_slot = jax.random.categorical(
                    step_rng, step_target_logits, axis=-1
                )
            all_target_slots.append(chosen_target_slot)

            stop_prob = jax.nn.sigmoid(step_stop_logit)
            if deterministic or current_rng is None:
                chosen_stop = (stop_prob >= 0.5).astype(jnp.int32)
            else:
                step_rng, current_rng = jax.random.split(current_rng)
                chosen_stop = jax.random.bernoulli(step_rng, stop_prob).astype(
                    jnp.int32
                )
            all_stops.append(chosen_stop)

            chosen_edge_emb = jnp.take_along_axis(
                chosen_edges,
                chosen_target_slot[:, None, None],
                axis=1,
            ).squeeze(1)
            current_input_emb = chosen_edge_emb

        return (
            jnp.stack(all_source_logits, axis=1),
            jnp.stack(all_target_logits, axis=1),
            jnp.stack(all_stop_logits, axis=1),
            jnp.stack(all_ship_logits, axis=1),
            jnp.stack(all_sources, axis=1),
            jnp.stack(all_target_slots, axis=1),
            jnp.stack(all_stops, axis=1),
        )
