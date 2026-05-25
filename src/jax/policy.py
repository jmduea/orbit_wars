from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

import jax
from src.config import TrainConfig
from src.features.registry import (
    edge_feature_dim,
    edge_k,
    global_feature_dim,
    planet_feature_dim,
)
from src.game.constants import MAX_PLANETS
from src.jax.action_codec import (
    FactoredPolicyOutput,
    JaxPolicyOutput,
    action_log_prob_and_entropy,
    ensure_policy_sequence,
)
from src.jax.decoders.factorized_topk_pointer import FactorizedTopKPointerDecoder
from src.jax.encoders import EncoderOutput
from src.jax.encoders.planet_encoder_common import (
    PlanetEdgeEncoderOutput,
    finalize_planet_edge_encoder_output,
    fuse_source_target_edges,
    mlp,
    planet_orbit_coords,
)
from src.jax.encoders.planet_graph_transformer import PlanetGraphTransformerEncoder
from src.jax.encoders.remat import remat_if
from src.jax.features import TurnBatch

# --- Decoders ---


class FeedForwardActionDecoder(nn.Module):
    """Single-step target and ship-bucket decoder for non-autoregressive policies."""

    ship_bucket_count: int
    hidden_size: int = 128

    @nn.compact
    def __call__(
        self,
        encoder_out: EncoderOutput,
        candidate_mask: jax.Array,
        target_sequence: jax.Array | None = None,
        rng: jax.Array | None = None,
        deterministic: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del target_sequence, rng, deterministic

        expanded_context = jnp.broadcast_to(
            encoder_out.context_query[:, None, :],
            encoder_out.attended_candidates.shape[:-1]
            + (encoder_out.context_query.shape[-1],),
        )
        joint = jnp.concatenate(
            [expanded_context, encoder_out.attended_candidates], axis=-1
        )
        target_hidden = nn.relu(nn.Dense(self.hidden_size, name="target_dense")(joint))
        target_logits = nn.Dense(1, name="target_out")(target_hidden).squeeze(-1)
        target_logits = jnp.where(
            candidate_mask, target_logits, jnp.finfo(jnp.float32).min
        )
        ship_hidden = nn.relu(nn.Dense(self.hidden_size, name="ship_dense")(joint))
        ship_logits = nn.Dense(self.ship_bucket_count, name="ship_out")(ship_hidden)
        decoded_target_sequence = jnp.full(
            (target_logits.shape[0], 1), -1, dtype=jnp.int32
        )
        return (
            target_logits[:, None, :],
            ship_logits[:, None, :, :],
            decoded_target_sequence,
        )


class AutoregressivePointerDecoder(nn.Module):
    """K-Step sequential pointer network decoding strategy.

    Sequentially steps through up to K decisions per game turn, using a GRU cell
    to track execution state history across successive pointer actions.
    """

    ship_bucket_count: int
    max_moves_k: int
    hidden_size: int = 128

    @nn.compact
    def __call__(
        self,
        encoder_out: EncoderOutput,
        candidate_mask: jax.Array,
        target_sequence: jax.Array | None = None,
        rng: jax.Array | None = None,
        deterministic: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        batch_size = encoder_out.context_query.shape[0]

        decoder_cell = nn.GRUCell(features=self.hidden_size, name="dec_gru")
        query_dense = nn.Dense(self.hidden_size, name="ptr_q")
        key_dense = nn.Dense(self.hidden_size, name="ptr_k")
        ship_dense = nn.Dense(self.hidden_size, name="ship_dense_step")
        ship_out = nn.Dense(self.ship_bucket_count, name="ship_out_step")

        init_decoder_state = nn.Dense(self.hidden_size, name="init_dec_state")(
            encoder_out.context_query
        )

        start_token = self.param(
            "start_token", nn.initializers.zeros, (self.hidden_size,)
        )
        current_input_emb = jnp.broadcast_to(
            start_token[None, :], (batch_size, self.hidden_size)
        )

        all_target_logits, all_ship_logits, all_chosen_targets = [], [], []
        current_state = init_decoder_state
        current_rng = rng

        for step_idx in range(self.max_moves_k):
            current_state, _ = decoder_cell(current_state, current_input_emb)

            q = query_dense(current_state)[:, None, :]
            k = key_dense(encoder_out.attended_candidates)

            step_target_logits = jnp.einsum("b1h,bch->bc", q, k) / jnp.sqrt(
                self.hidden_size
            )
            step_target_logits = jnp.where(
                candidate_mask, step_target_logits, jnp.finfo(jnp.float32).min
            )
            all_target_logits.append(step_target_logits)

            expanded_state = jnp.broadcast_to(
                current_state[:, None, :], encoder_out.attended_candidates.shape
            )
            ship_input = jnp.concatenate(
                [expanded_state, encoder_out.attended_candidates], axis=-1
            )
            step_ship_logits = ship_out(nn.relu(ship_dense(ship_input)))
            all_ship_logits.append(step_ship_logits)

            if target_sequence is not None:
                chosen_target = target_sequence[:, step_idx]
            elif deterministic or current_rng is None:
                chosen_target = jnp.argmax(step_target_logits, axis=-1)
            else:
                step_rng, current_rng = jax.random.split(current_rng)
                chosen_target = jax.random.categorical(
                    step_rng, step_target_logits, axis=-1
                )
            all_chosen_targets.append(chosen_target)

            current_input_emb = jnp.take_along_axis(
                encoder_out.attended_candidates, chosen_target[:, None, None], axis=1
            ).squeeze(1)

        return (
            jnp.stack(all_target_logits, axis=1),
            jnp.stack(all_ship_logits, axis=1),
            jnp.stack(all_chosen_targets, axis=1),
        )


class SharedValueHead(nn.Module):
    """Single critic head shared across all training formats."""

    hidden_size: int = 128

    @nn.compact
    def __call__(
        self,
        value_input: jax.Array,
        player_count: jax.Array | None = None,
    ) -> jax.Array:
        del player_count
        value_hidden = nn.relu(
            nn.Dense(self.hidden_size, name="shared_value_dense")(value_input)
        )
        return nn.Dense(1, name="shared_value_out")(value_hidden).squeeze(-1)


class FormatRoutedValueHead(nn.Module):
    """Select between dedicated 2p and 4p critic heads per batch row."""

    hidden_size: int = 128

    @nn.compact
    def __call__(
        self,
        value_input: jax.Array,
        player_count: jax.Array | None = None,
    ) -> jax.Array:
        if player_count is None:
            raise ValueError(
                "FormatRoutedValueHead requires an explicit player_count array."
            )

        player_count = jnp.asarray(player_count, dtype=jnp.int32)
        if player_count.ndim == 0:
            player_count = jnp.full(
                (value_input.shape[0],), player_count, dtype=jnp.int32
            )
        else:
            player_count = player_count.reshape((value_input.shape[0],))

        two_player_hidden = nn.relu(
            nn.Dense(self.hidden_size, name="two_player_value_dense")(value_input)
        )
        two_player_value = nn.Dense(1, name="two_player_value_out")(
            two_player_hidden
        ).squeeze(-1)

        four_player_hidden = nn.relu(
            nn.Dense(self.hidden_size, name="four_player_value_dense")(value_input)
        )
        four_player_value = nn.Dense(1, name="four_player_value_out")(
            four_player_hidden
        ).squeeze(-1)

        return jnp.where(player_count == 2, two_player_value, four_player_value)


# --- Helper Functions ---


def safe_attention_mask(candidate_mask: jax.Array) -> jax.Array:
    """Ensure attention has at least one unmasked key for every batch row."""

    has_valid_key = candidate_mask.any(axis=-1, keepdims=True)
    return jnp.where(has_valid_key, candidate_mask, jnp.ones_like(candidate_mask))


def build_value_head(cfg: TrainConfig) -> nn.Module:
    """Construct the configured critic head module."""

    normalized_value_head = cfg.model.value_head.strip().lower()
    if normalized_value_head == "shared":
        return SharedValueHead(hidden_size=cfg.model.hidden_size)
    if normalized_value_head == "format_routed":
        return FormatRoutedValueHead(hidden_size=cfg.model.hidden_size)
    raise ValueError(
        f"Unsupported value head '{cfg.model.value_head}'. Expected 'shared' or 'format_routed'."
    )


# --- Planet-Edge Policy Encoders ---


class PlanetGnnMessageLayer(nn.Module):
    """Single GNN message-passing block for ``PlanetEdgeBackboneEncoder``."""

    hidden_size: int
    layer_idx: int

    @nn.compact
    def __call__(
        self,
        current_planet_states: jax.Array,
        final_adj_mask: jax.Array,
    ) -> jax.Array:
        residual = current_planet_states
        msg_proj = nn.Dense(self.hidden_size, name=f"planet_msg_proj_{self.layer_idx}")(
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
            f"planet_gnn_layer_{self.layer_idx}",
        )
        return nn.LayerNorm(name=f"planet_gnn_norm_{self.layer_idx}")(
            residual + current_planet_states
        )


class PlanetEdgeBackboneEncoder(nn.Module):
    """Planet GNN with top-K edge message passing on v2 turn batches."""

    hidden_size: int = 128
    k_neighbors: int = 5
    msg_passing_layers: int = 2
    planet_feature_dim: int = 13
    edge_feature_dim: int = 18
    global_feature_dim: int = 46
    edge_k: int = 3
    gradient_checkpointing: bool = False

    def setup(self) -> None:
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be at least 1.")
        if self.msg_passing_layers < 1:
            raise ValueError("msg_passing_layers must be at least 1.")
        if self.edge_k < 0:
            raise ValueError("edge_k must be non-negative.")

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

        coords = planet_orbit_coords(batch.planet_features)

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
        layer_cls = remat_if(PlanetGnnMessageLayer, self.gradient_checkpointing)
        for layer_idx in range(self.msg_passing_layers):
            current_planet_states = layer_cls(
                hidden_size=self.hidden_size,
                layer_idx=layer_idx,
                name=f"planet_gnn_block_{layer_idx}",
            )(current_planet_states, final_adj_mask)

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


class ComposablePlanetPolicy(nn.Module):
    """Encoder/decoder wrapper for v2 planet-edge batches."""

    encoder_module: nn.Module
    decoder_module: nn.Module
    value_head_module: nn.Module | None = None
    hidden_size: int = 128
    edge_k: int = 3

    @nn.compact
    def __call__(
        self,
        batch: TurnBatch,
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


class ComposableFactorizedPlanetPolicy(nn.Module):
    """Encoder + factorized top-K pointer decoder for v2 planet-edge batches."""

    encoder_module: nn.Module
    decoder_module: FactorizedTopKPointerDecoder
    value_head_module: nn.Module | None = None
    hidden_size: int = 128

    @nn.compact
    def __call__(
        self,
        batch: TurnBatch,
        player_count: jax.Array | None = None,
        source_sequence: jax.Array | None = None,
        target_slot_sequence: jax.Array | None = None,
        rng: jax.Array | None = None,
        deterministic: bool = False,
    ) -> FactoredPolicyOutput:
        encoder_out = self.encoder_module(batch)
        (
            source_logits,
            target_logits,
            stop_logits,
            ship_logits,
            decoded_source_sequence,
            decoded_target_slot_sequence,
            decoded_stop_sequence,
        ) = self.decoder_module(
            encoder_out,
            source_sequence=source_sequence,
            target_slot_sequence=target_slot_sequence,
            rng=rng,
            deterministic=deterministic,
        )

        value_head_module = self.value_head_module
        if value_head_module is None:
            value_head_module = SharedValueHead(hidden_size=self.hidden_size)
        value = value_head_module(encoder_out.value_input, player_count=player_count)

        return FactoredPolicyOutput(
            source_logits=source_logits,
            target_logits=target_logits,
            stop_logits=stop_logits,
            ship_logits=ship_logits,
            value=value,
            decoded_source_sequence=decoded_source_sequence,
            decoded_target_slot_sequence=decoded_target_slot_sequence,
            decoded_stop_sequence=decoded_stop_sequence,
        )


def edge_action_count(task_cfg) -> int:
    """Flat edge logits including the always-legal NO_OP slot."""

    return MAX_PLANETS * edge_k(task_cfg) + 1


def build_pointer_decoder(cfg: TrainConfig) -> nn.Module:
    """Construct the configured pointer decoder module."""

    from src.artifacts.checkpoint_compat import (
        POINTER_DECODER_FACTORIZED_TOPK,
        pointer_decoder_for_model,
    )

    hidden = cfg.model.hidden_size
    if pointer_decoder_for_model(cfg.model) == POINTER_DECODER_FACTORIZED_TOPK:
        return FactorizedTopKPointerDecoder(
            ship_bucket_count=cfg.task.ship_bucket_count,
            max_moves_k=cfg.model.max_moves_k,
            hidden_size=hidden,
            edge_k=edge_k(cfg.task),
        )
    return AutoregressivePointerDecoder(
        ship_bucket_count=cfg.task.ship_bucket_count,
        max_moves_k=cfg.model.max_moves_k,
        hidden_size=hidden,
    )


def _is_factorized_pointer(cfg: TrainConfig) -> bool:
    from src.artifacts.checkpoint_compat import (
        POINTER_DECODER_FACTORIZED_TOPK,
        pointer_decoder_for_model,
    )

    return pointer_decoder_for_model(cfg.model) == POINTER_DECODER_FACTORIZED_TOPK


def build_jax_policy(cfg: TrainConfig) -> nn.Module:
    """Construct the planet-edge policy for the configured architecture."""
    normalized_architecture = cfg.model.architecture.strip().lower()
    if normalized_architecture in {"gnn_pointer", "gnn_pointer_v2"}:
        return build_gnn_pointer_policy(cfg)
    if normalized_architecture == "planet_graph_transformer":
        return build_planet_graph_transformer_policy(cfg)
    raise ValueError(
        f"Unsupported JAX model architecture '{cfg.model.architecture}'. "
        "Expected 'gnn_pointer' or 'planet_graph_transformer'."
    )


def _build_planet_edge_policy_shell(
    cfg: TrainConfig,
    *,
    encoder_module: nn.Module,
) -> nn.Module:
    """Wrap a planet-edge encoder with the configured pointer decoder shell."""

    hidden = cfg.model.hidden_size
    k_slots = edge_k(cfg.task)
    decoder_module = build_pointer_decoder(cfg)
    value_head_module = build_value_head(cfg)
    if _is_factorized_pointer(cfg):
        return ComposableFactorizedPlanetPolicy(
            encoder_module=encoder_module,
            decoder_module=decoder_module,
            value_head_module=value_head_module,
            hidden_size=hidden,
        )
    return ComposablePlanetPolicy(
        encoder_module=encoder_module,
        decoder_module=decoder_module,
        value_head_module=value_head_module,
        hidden_size=hidden,
        edge_k=k_slots,
    )


def build_planet_graph_transformer_policy(cfg: TrainConfig) -> nn.Module:
    """Construct the planet graph transformer policy (M2 encoder)."""

    hidden = cfg.model.hidden_size
    k_slots = edge_k(cfg.task)
    if hidden % cfg.model.attention_heads != 0:
        raise ValueError(
            "model.hidden_size must be divisible by model.attention_heads for "
            "planet_graph_transformer."
        )
    encoder_module = PlanetGraphTransformerEncoder(
        hidden_size=hidden,
        attention_heads=cfg.model.attention_heads,
        planet_transformer_layers=cfg.model.planet_transformer_layers,
        spatial_attention_bias=cfg.model.spatial_attention_bias,
        planet_feature_dim=planet_feature_dim(cfg.task),
        edge_feature_dim=edge_feature_dim(cfg.task),
        global_feature_dim=global_feature_dim(cfg.task),
        edge_k=k_slots,
        gradient_checkpointing=cfg.training.enable_gradient_checkpointing,
    )
    return _build_planet_edge_policy_shell(cfg, encoder_module=encoder_module)


def build_gnn_pointer_policy(cfg: TrainConfig) -> nn.Module:
    """Construct the v2 GNN pointer policy for ``TurnBatch`` inputs."""

    hidden = cfg.model.hidden_size
    k_slots = edge_k(cfg.task)
    encoder_module = PlanetEdgeBackboneEncoder(
        hidden_size=hidden,
        k_neighbors=cfg.model.gnn_k_neighbors,
        msg_passing_layers=cfg.model.gnn_message_passing_layers,
        planet_feature_dim=planet_feature_dim(cfg.task),
        edge_feature_dim=edge_feature_dim(cfg.task),
        global_feature_dim=global_feature_dim(cfg.task),
        edge_k=k_slots,
        gradient_checkpointing=cfg.training.enable_gradient_checkpointing,
    )
    return _build_planet_edge_policy_shell(cfg, encoder_module=encoder_module)


def make_synthetic_turn_batch(
    batch_size: int,
    task_cfg,
    *,
    key: jax.Array | None = None,
) -> TurnBatch:
    """Build a random ``TurnBatch`` for policy smoke tests."""

    if key is None:
        key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    k_slots = edge_k(task_cfg)
    planet_dim = planet_feature_dim(task_cfg)
    global_dim = global_feature_dim(task_cfg)
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

    return TurnBatch(
        planet_features=planet_features,
        planet_mask=planet_mask,
        edge_features=edge_features,
        edge_mask=edge_mask,
        edge_src_ids=edge_src_ids,
        edge_tgt_ids=edge_tgt_ids,
        global_features=global_features,
        theta_ref=theta_ref,
    )


def sample_actions(
    key: jax.Array,
    output: JaxPolicyOutput,
    *,
    deterministic: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sample or greedily select target and ship-bucket actions.

    Returns ``(target_index, ship_bucket, log_prob, entropy)`` for each flattened
    decision row in ``output``.
    """

    key_target, key_ship = jax.random.split(key)
    target_logits = ensure_policy_sequence(output.target_logits)
    ship_logits = ensure_policy_sequence(output.ship_logits)
    decoded_targets = output.decoded_target_sequence
    if deterministic:
        target_index = jnp.argmax(target_logits, axis=-1)
    else:
        sampled_target = jax.random.categorical(key_target, target_logits, axis=-1)
        target_index = jnp.where(decoded_targets >= 0, decoded_targets, sampled_target)
    selected_ship_logits = jnp.take_along_axis(
        ship_logits,
        target_index[..., None, None].repeat(ship_logits.shape[-1], axis=-1),
        axis=2,
    ).squeeze(axis=2)
    ship_bucket = jnp.where(
        deterministic,
        jnp.argmax(selected_ship_logits, axis=-1),
        jax.random.categorical(key_ship, selected_ship_logits, axis=-1),
    )
    log_prob, entropy = action_log_prob_and_entropy(output, target_index, ship_bucket)
    return target_index, ship_bucket, log_prob, entropy


def first_policy_step(output: JaxPolicyOutput) -> JaxPolicyOutput:
    """Project a K-step policy output to the executable first move."""

    if output.target_logits.ndim == 3:
        return JaxPolicyOutput(
            target_logits=output.target_logits[:, 0, :],
            ship_logits=output.ship_logits[:, 0, :, :],
            value=output.value,
            decoded_target_sequence=output.decoded_target_sequence[:, :1],
        )
    return output
