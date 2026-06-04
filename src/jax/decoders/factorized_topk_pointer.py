"""Factorized source → target-slot pointer decoder for top-K edges."""

from __future__ import annotations

from typing import NamedTuple

import flax.linen as nn
import jax.numpy as jnp

import jax
from src.jax.decoder_carry import resolve_decoder_initial_state
from src.jax.encoders.planet_encoder_common import PlanetEdgeEncoderOutput


class FactorizedDecodeCarry(NamedTuple):
    """GRU state and edge embedding input for one autoregressive decoder step."""

    state: jax.Array
    input_emb: jax.Array


class FactorizedStepLogits(NamedTuple):
    """Logits for a single factorized sub-move (one column of the K-step sequence)."""

    source_logits: jax.Array
    target_logits: jax.Array
    stop_logits: jax.Array
    ship_logits: jax.Array


class FactorizedTopKPointerDecoder(nn.Module):
    """Autoregressive factorized pointer: source planet, target slot, bucket, stop."""

    ship_bucket_count: int
    max_moves_k: int
    ship_action_mode: str = "buckets"
    decoder_carry: bool = False
    hidden_size: int = 128
    edge_k: int = 3

    def setup(self) -> None:
        self.decoder_cell = nn.GRUCell(features=self.hidden_size, name="factorized_dec_gru")
        self.init_dec_dense = nn.Dense(self.hidden_size, name="factorized_init_dec_state")
        self.src_q_dense = nn.Dense(self.hidden_size, name="factorized_src_q")
        self.src_k_dense = nn.Dense(self.hidden_size, name="factorized_src_k")
        self.tgt_q_dense = nn.Dense(self.hidden_size, name="factorized_tgt_q")
        self.tgt_k_dense = nn.Dense(self.hidden_size, name="factorized_tgt_k")
        self.stop_dense = nn.Dense(1, name="factorized_stop")
        self.ship_dense = nn.Dense(self.hidden_size, name="factorized_ship_dense")
        ship_width = (
            1
            if self.ship_action_mode.strip().lower() == "continuous_fraction"
            else self.ship_bucket_count
        )
        self.ship_out = nn.Dense(ship_width, name="factorized_ship_out")
        self.start_token = self.param(
            "factorized_start_token",
            nn.initializers.zeros,
            (self.hidden_size,),
        )

    def _encoder_views(
        self, encoder_out: PlanetEdgeEncoderOutput
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        batch_size = encoder_out.context_query.shape[0]
        planet_states = encoder_out.planet_states
        edge_hidden = encoder_out.edge_hidden
        edge_mask = encoder_out.edge_mask.astype(bool)
        planet_mask = encoder_out.planet_mask.astype(bool)
        source_mask = planet_mask & edge_mask.any(axis=-1)
        illegal_logit = jnp.finfo(jnp.float32).min
        scale = jnp.sqrt(jnp.asarray(self.hidden_size, dtype=jnp.float32))
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)
        return (
            planet_states,
            edge_hidden,
            edge_mask,
            source_mask,
            illegal_logit,
            scale,
            batch_indices,
        )

    def init_carry(
        self,
        encoder_out: PlanetEdgeEncoderOutput,
        *,
        decoder_hidden_in: jax.Array | None = None,
    ) -> FactorizedDecodeCarry:
        batch_size = encoder_out.context_query.shape[0]
        init_decoder_state = self.init_dec_dense(encoder_out.context_query)
        state = resolve_decoder_initial_state(
            init_decoder_state,
            decoder_hidden_in,
            enabled=self.decoder_carry,
        )
        input_emb = jnp.broadcast_to(
            self.start_token[None, :], (batch_size, self.hidden_size)
        )
        return FactorizedDecodeCarry(state=state, input_emb=input_emb)

    def advance_carry_input(
        self,
        encoder_out: PlanetEdgeEncoderOutput,
        carry: FactorizedDecodeCarry,
        *,
        source: jax.Array,
        target_slot: jax.Array,
    ) -> FactorizedDecodeCarry:
        """Set next-step GRU input from a committed (source, target_slot) launch."""

        _, edge_hidden, _, _, _, _, batch_indices = self._encoder_views(encoder_out)
        chosen_edges = edge_hidden[batch_indices, source, :, :]
        chosen_edge_emb = jnp.take_along_axis(
            chosen_edges,
            target_slot[:, None, None],
            axis=1,
        ).squeeze(1)
        return carry._replace(input_emb=chosen_edge_emb)

    def step(
        self,
        encoder_out: PlanetEdgeEncoderOutput,
        carry: FactorizedDecodeCarry,
        *,
        teacher_source: jax.Array | None = None,
        teacher_target_slot: jax.Array | None = None,
        rng: jax.Array | None = None,
        deterministic: bool = False,
    ) -> tuple[FactorizedStepLogits, FactorizedDecodeCarry, jax.Array, jax.Array, jax.Array]:
        """Run one GRU decoder step and return logits plus teacher/chosen indices."""

        (
            planet_states,
            edge_hidden,
            edge_mask,
            source_mask,
            illegal_logit,
            scale,
            batch_indices,
        ) = self._encoder_views(encoder_out)

        current_state, _ = self.decoder_cell(carry.state, carry.input_emb)
        step_stop_logit = self.stop_dense(current_state).squeeze(-1)

        src_q = self.src_q_dense(current_state)[:, None, :]
        src_k = self.src_k_dense(planet_states)
        step_source_logits = jnp.einsum("b1h,bph->bp", src_q, src_k) / scale
        step_source_logits = jnp.where(source_mask, step_source_logits, illegal_logit)

        current_rng = rng
        if teacher_source is not None:
            chosen_source = teacher_source
        elif deterministic or current_rng is None:
            chosen_source = jnp.argmax(step_source_logits, axis=-1)
        else:
            step_rng, current_rng = jax.random.split(current_rng)
            chosen_source = jax.random.categorical(
                step_rng, step_source_logits, axis=-1
            )

        chosen_edges = edge_hidden[batch_indices, chosen_source, :, :]
        chosen_edge_mask = edge_mask[batch_indices, chosen_source, :]

        tgt_q = self.tgt_q_dense(current_state)[:, None, :]
        tgt_k = self.tgt_k_dense(chosen_edges)
        step_target_logits = jnp.einsum("b1h,bkh->bk", tgt_q, tgt_k) / scale
        step_target_logits = jnp.where(
            chosen_edge_mask, step_target_logits, illegal_logit
        )

        expanded_state = jnp.broadcast_to(
            current_state[:, None, :], (*chosen_edges.shape[:2], self.hidden_size)
        )
        ship_input = jnp.concatenate([expanded_state, chosen_edges], axis=-1)
        step_ship_logits = self.ship_out(nn.relu(self.ship_dense(ship_input)))

        if teacher_target_slot is not None:
            chosen_target_slot = teacher_target_slot
        elif deterministic or current_rng is None:
            chosen_target_slot = jnp.argmax(step_target_logits, axis=-1)
        else:
            step_rng, current_rng = jax.random.split(current_rng)
            chosen_target_slot = jax.random.categorical(
                step_rng, step_target_logits, axis=-1
            )

        stop_prob = jax.nn.sigmoid(step_stop_logit)
        if deterministic or current_rng is None:
            chosen_stop = (stop_prob >= 0.5).astype(jnp.int32)
        else:
            step_rng, current_rng = jax.random.split(current_rng)
            chosen_stop = jax.random.bernoulli(step_rng, stop_prob).astype(jnp.int32)

        step_logits = FactorizedStepLogits(
            source_logits=step_source_logits,
            target_logits=step_target_logits,
            stop_logits=step_stop_logit,
            ship_logits=step_ship_logits,
        )
        new_carry = FactorizedDecodeCarry(state=current_state, input_emb=carry.input_emb)
        return step_logits, new_carry, chosen_source, chosen_target_slot, chosen_stop

    def __call__(
        self,
        encoder_out: PlanetEdgeEncoderOutput,
        *,
        source_sequence: jax.Array | None = None,
        target_slot_sequence: jax.Array | None = None,
        decoder_hidden_in: jax.Array | None = None,
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
        jax.Array,
    ]:
        carry = self.init_carry(encoder_out, decoder_hidden_in=decoder_hidden_in)
        current_rng = rng

        (
            all_source_logits,
            all_target_logits,
            all_stop_logits,
            all_ship_logits,
            all_sources,
            all_target_slots,
            all_stops,
        ) = ([], [], [], [], [], [], [])

        for step_idx in range(self.max_moves_k):
            teacher_source = (
                source_sequence[:, step_idx] if source_sequence is not None else None
            )
            teacher_target = (
                target_slot_sequence[:, step_idx]
                if target_slot_sequence is not None
                else None
            )
            step_logits, carry, chosen_source, chosen_target_slot, _chosen_stop = (
                self.step(
                    encoder_out,
                    carry,
                    teacher_source=teacher_source,
                    teacher_target_slot=teacher_target,
                    rng=current_rng,
                    deterministic=deterministic,
                )
            )
            if current_rng is not None and not deterministic:
                current_rng, _ = jax.random.split(current_rng)

            all_source_logits.append(step_logits.source_logits)
            all_target_logits.append(step_logits.target_logits)
            all_stop_logits.append(step_logits.stop_logits)
            all_ship_logits.append(step_logits.ship_logits)
            all_sources.append(chosen_source)
            all_target_slots.append(chosen_target_slot)
            all_stops.append(_chosen_stop)

            carry = self.advance_carry_input(
                encoder_out,
                carry,
                source=chosen_source,
                target_slot=chosen_target_slot,
            )

        return (
            jnp.stack(all_source_logits, axis=1),
            jnp.stack(all_target_logits, axis=1),
            jnp.stack(all_stop_logits, axis=1),
            jnp.stack(all_ship_logits, axis=1),
            jnp.stack(all_sources, axis=1),
            jnp.stack(all_target_slots, axis=1),
            jnp.stack(all_stops, axis=1),
            carry.state,
        )
