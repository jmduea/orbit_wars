"""V2 policy stack: planet/edge GNN encoder + joint edge pointer decoder."""

from __future__ import annotations

from typing import NamedTuple

import flax.linen as nn
import jax.numpy as jnp

import jax
from src.config import TrainConfig
from src.features.registry_v2 import (
    edge_feature_dim,
    edge_k,
    global_v2_feature_dim,
    planet_feature_dim,
)
from src.game.constants import MAX_PLANETS
from src.jax.features_v2 import JaxTurnBatchV2
from src.jax.policy import (
    AutoregressivePointerDecoder,
    EncoderOutput,
    JaxPolicyOutput,
    SharedValueHead,
    build_value_head,
    masked_mean,
    mlp,
)


class EncoderOutputV2(NamedTuple):
    """Policy encoder contract for ``JaxTurnBatchV2`` inputs."""

    attended_edges: jax.Array
    edge_action_mask: jax.Array
    context_query: jax.Array
    value_input: jax.Array


class PlanetEdgeBackboneEncoder(nn.Module):
    """Planet GNN with top-K edge message passing on v2 turn batches."""

    hidden_size: int = 128
    k_neighbors: int = 5
    msg_passing_layers: int = 2
    planet_feature_dim: int = 13
    edge_feature_dim: int = 12
    global_feature_dim: int = 46
    edge_k: int = 3

    def setup(self) -> None:
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be at least 1.")
        if self.msg_passing_layers < 1:
            raise ValueError("msg_passing_layers must be at least 1.")
        if self.edge_k < 0:
            raise ValueError("edge_k must be non-negative.")

    @nn.compact
    def __call__(self, batch: JaxTurnBatchV2) -> EncoderOutputV2:
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
                (batch.planet_features.shape[0], MAX_PLANETS, 0, self.hidden_size),
                dtype=jnp.float32,
            )
        else:
            edge_hidden = mlp(
                batch.edge_features,
                self.hidden_size,
                self.hidden_size,
                "edge_enc",
            )

        orbit_radius = batch.planet_features[..., 1]
        orbit_angle = batch.planet_features[..., 2] * jnp.pi
        coords = jnp.stack(
            [
                orbit_radius * jnp.cos(orbit_angle),
                orbit_radius * jnp.sin(orbit_angle),
            ],
            axis=-1,
        )

        num_planets = planet_hidden.shape[-2]
        diffs = coords[:, :, None, :] - coords[:, None, :, :]
        dist_matrix = jnp.sum(diffs**2, axis=-1)
        neighbor_count = min(self.k_neighbors, num_planets)
        _, topk_indices = jax.lax.top_k(-dist_matrix, k=neighbor_count)
        adj_matrix = jnp.sum(jax.nn.one_hot(topk_indices, num_planets), axis=-2).astype(
            bool
        )
        final_adj_mask = adj_matrix & (
            planet_mask[:, :, None] & planet_mask[:, None, :]
        )

        current_planet_states = planet_hidden
        for layer_idx in range(self.msg_passing_layers):
            msg_proj = nn.Dense(self.hidden_size, name=f"planet_msg_proj_{layer_idx}")(
                current_planet_states
            )
            masked_messages = jnp.where(
                final_adj_mask[..., None], msg_proj[:, None, :, :], 0.0
            )
            aggregated_messages = jnp.sum(masked_messages, axis=2)
            combined_planet_input = jnp.concatenate(
                [current_planet_states, aggregated_messages], axis=-1
            )
            current_planet_states = mlp(
                combined_planet_input,
                self.hidden_size,
                self.hidden_size,
                f"planet_gnn_layer_{layer_idx}",
            )
            current_planet_states = nn.LayerNorm(name=f"planet_gnn_norm_{layer_idx}")(
                planet_hidden + current_planet_states
            )

        if self.edge_k > 0:
            src_planet = jnp.broadcast_to(
                current_planet_states[:, :, None, :],
                (*edge_hidden.shape[:3], self.hidden_size),
            )
            edge_input = jnp.concatenate([src_planet, edge_hidden], axis=-1)
            edge_hidden = mlp(
                edge_input, self.hidden_size, self.hidden_size, "edge_fuse"
            )
            edge_hidden = nn.LayerNorm(name="edge_fuse_norm")(edge_hidden)
            batch_size = edge_hidden.shape[0]
            attended_edges = edge_hidden.reshape(
                batch_size, MAX_PLANETS * self.edge_k, -1
            )
            edge_action_mask = edge_mask.reshape(batch_size, MAX_PLANETS * self.edge_k)
        else:
            batch_size = planet_hidden.shape[0]
            attended_edges = jnp.zeros(
                (batch_size, 0, self.hidden_size), dtype=jnp.float32
            )
            edge_action_mask = jnp.zeros((batch_size, 0), dtype=bool)

        pooled_planets = masked_mean(current_planet_states, planet_mask)
        pooled_edges = masked_mean(attended_edges, edge_action_mask)
        context_query = mlp(
            jnp.concatenate([global_hidden, pooled_planets], axis=-1),
            self.hidden_size,
            self.hidden_size,
            "context_query",
        )
        value_input = jnp.concatenate(
            [context_query, global_hidden, pooled_planets, pooled_edges], axis=-1
        )

        return EncoderOutputV2(
            attended_edges=attended_edges,
            edge_action_mask=edge_action_mask,
            context_query=context_query,
            value_input=value_input,
        )


class ComposablePlanetPolicyV2(nn.Module):
    """Encoder/decoder wrapper for v2 planet-edge batches."""

    encoder_module: nn.Module
    decoder_module: nn.Module
    value_head_module: nn.Module | None = None
    hidden_size: int = 128
    edge_k: int = 3

    @nn.compact
    def __call__(
        self,
        batch: JaxTurnBatchV2,
        player_count: jax.Array | None = None,
        target_sequence: jax.Array | None = None,
        rng: jax.Array | None = None,
        deterministic: bool = False,
    ) -> JaxPolicyOutput:
        encoder_out = self.encoder_module(batch)
        batch_size = encoder_out.attended_edges.shape[0]
        noop_embedding = self.param(
            "noop_edge_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.hidden_size),
        )
        noop_embedding = jnp.broadcast_to(
            noop_embedding, (batch_size, 1, self.hidden_size)
        )
        attended_candidates = jnp.concatenate(
            [encoder_out.attended_edges, noop_embedding], axis=1
        )
        noop_mask = jnp.ones((batch_size, 1), dtype=bool)
        action_mask = jnp.concatenate([encoder_out.edge_action_mask, noop_mask], axis=1)

        decoder_encoder = EncoderOutput(
            attended_candidates=attended_candidates,
            context_query=encoder_out.context_query,
            value_input=encoder_out.value_input,
        )
        target_logits, ship_logits, decoded_target_sequence = self.decoder_module(
            decoder_encoder,
            action_mask,
            target_sequence=target_sequence,
            rng=rng,
            deterministic=deterministic,
        )

        value_head_module = self.value_head_module
        if value_head_module is None:
            value_head_module = SharedValueHead(hidden_size=self.hidden_size)
        value = value_head_module(encoder_out.value_input, player_count=player_count)

        return JaxPolicyOutput(
            target_logits=target_logits,
            ship_logits=ship_logits,
            value=value,
            decoded_target_sequence=decoded_target_sequence,
        )


def edge_action_count(task_cfg) -> int:
    """Flat edge logits including the always-legal NO_OP slot."""

    return MAX_PLANETS * edge_k(task_cfg) + 1


def build_gnn_pointer_v2_policy(cfg: TrainConfig) -> ComposablePlanetPolicyV2:
    """Construct the v2 GNN pointer policy for ``JaxTurnBatchV2`` inputs."""

    hidden = cfg.model.hidden_size
    k_slots = edge_k(cfg.task)
    return ComposablePlanetPolicyV2(
        encoder_module=PlanetEdgeBackboneEncoder(
            hidden_size=hidden,
            k_neighbors=cfg.model.gnn_k_neighbors,
            msg_passing_layers=cfg.model.gnn_message_passing_layers,
            planet_feature_dim=planet_feature_dim(cfg.task),
            edge_feature_dim=edge_feature_dim(cfg.task),
            global_feature_dim=global_v2_feature_dim(cfg.task),
            edge_k=k_slots,
        ),
        decoder_module=AutoregressivePointerDecoder(
            ship_bucket_count=cfg.task.ship_bucket_count,
            max_moves_k=cfg.model.max_moves_k,
            hidden_size=hidden,
        ),
        value_head_module=build_value_head(cfg),
        hidden_size=hidden,
        edge_k=k_slots,
    )


def make_synthetic_turn_batch_v2(
    batch_size: int,
    task_cfg,
    *,
    key: jax.Array | None = None,
) -> JaxTurnBatchV2:
    """Build a random ``JaxTurnBatchV2`` for policy smoke tests."""

    if key is None:
        key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    k_slots = edge_k(task_cfg)
    planet_dim = planet_feature_dim(task_cfg)
    global_dim = global_v2_feature_dim(task_cfg)
    edge_dim = edge_feature_dim(task_cfg)

    planet_features = jax.random.normal(
        k1, (batch_size, MAX_PLANETS, planet_dim), dtype=jnp.float32
    )
    planet_mask = jnp.ones((batch_size, MAX_PLANETS), dtype=bool)
    edge_features = jax.random.normal(
        k2, (batch_size, MAX_PLANETS, k_slots, edge_dim), dtype=jnp.float32
    )
    edge_mask = jnp.ones((batch_size, MAX_PLANETS, k_slots), dtype=bool)
    edge_src_ids = jnp.broadcast_to(
        jnp.arange(MAX_PLANETS, dtype=jnp.int32)[None, :], (batch_size, MAX_PLANETS)
    )
    edge_tgt_ids = jnp.broadcast_to(
        jnp.arange(k_slots, dtype=jnp.int32)[None, None, :],
        (batch_size, MAX_PLANETS, k_slots),
    )
    global_features = jax.random.normal(k3, (batch_size, global_dim), dtype=jnp.float32)
    theta_ref = jax.random.uniform(k4, (batch_size,), dtype=jnp.float32)

    if k_slots == 0:
        edge_features = jnp.zeros(
            (batch_size, MAX_PLANETS, 0, edge_dim), dtype=jnp.float32
        )
        edge_mask = jnp.zeros((batch_size, MAX_PLANETS, 0), dtype=bool)
        edge_tgt_ids = jnp.zeros((batch_size, MAX_PLANETS, 0), dtype=jnp.int32)

    return JaxTurnBatchV2(
        planet_features=planet_features,
        planet_mask=planet_mask,
        edge_features=edge_features,
        edge_mask=edge_mask,
        edge_src_ids=edge_src_ids,
        edge_tgt_ids=edge_tgt_ids,
        global_features=global_features,
        theta_ref=theta_ref,
    )
