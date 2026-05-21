from __future__ import annotations

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from src import TrainConfig
from src.feature_registry import candidate_feature_schema


# --- Contracts ---
class JaxPolicyOutput(NamedTuple):
    """Unified policy output structure.

    Fields
    ------
    target_logits: jax.Array
        Shape: (batch, sequence_k, candidates)
    ship_logits: jax.Array
        Shape: (batch, sequence_k, candidates, ship_buckets)
    value: jax.Array
        Shape: (batch,)
    decoded_target_sequence: jax.Array
        Shape: (batch, sequence_k). Target path used by autoregressive decoders,
        or -1 for decoders whose steps can be sampled from logits directly.
    """

    target_logits: jax.Array
    ship_logits: jax.Array
    value: jax.Array
    decoded_target_sequence: jax.Array


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

# --- Encoders ---
class MLPBackboneEncoder(nn.Module):
    """Lightweight, fast MLP feature extractor."""

    hidden_size: int = 128

    @nn.compact
    def __call__(
        self,
        self_features: jax.Array,
        candidate_features: jax.Array,
        global_features: jax.Array,
        candidate_mask: jax.Array,
    ) -> EncoderOutput:
        self_hidden = mlp(
            self_features, self.hidden_size, self.hidden_size, "self_encoder"
        )
        global_hidden = mlp(
            global_features, self.hidden_size, self.hidden_size, "global_encoder"
        )
        candidate_hidden = mlp(
            candidate_features, self.hidden_size, self.hidden_size, "candidate_encoder"
        )

        pooled_candidates = masked_mean(candidate_hidden, candidate_mask)
        context_query = jnp.concatenate([self_hidden, global_hidden], axis=-1)
        value_input = jnp.concatenate([context_query, pooled_candidates], axis=-1)

        return EncoderOutput(
            attended_candidates=candidate_hidden,
            context_query=context_query,
            value_input=value_input,
        )


class TransformerBackboneEncoder(nn.Module):
    """Attention-based graph feature extractor."""

    hidden_size: int = 128
    attention_heads: int = 4
    enable_gradient_checkpointing: bool = False

    def setup(self) -> None:
        """Validate attention hyperparameters before parameter initialization."""

        if self.hidden_size % self.attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"attention_heads ({self.attention_heads})."
            )

    @nn.compact
    def __call__(
        self,
        self_features: jax.Array,
        candidate_features: jax.Array,
        global_features: jax.Array,
        candidate_mask: jax.Array,
    ) -> EncoderOutput:
        batch_size = self_features.shape[0]
        safe_mask = safe_attention_mask(candidate_mask)

        self_hidden = mlp(self_features, self.hidden_size, self.hidden_size, "self_enc")
        global_hidden = mlp(
            global_features, self.hidden_size, self.hidden_size, "global_enc"
        )
        candidate_hidden = mlp(
            candidate_features, self.hidden_size, self.hidden_size, "candidate_enc"
        )

        attention_module = (
            nn.remat(nn.MultiHeadDotProductAttention)
            if self.enable_gradient_checkpointing
            else nn.MultiHeadDotProductAttention
        )

        attended_candidates = attention_module(
            num_heads=self.attention_heads,
            qkv_features=self.hidden_size,
            out_features=self.hidden_size,
            name="mp_attn",
        )(
            candidate_hidden,
            candidate_hidden,
            mask=jnp.broadcast_to(
                safe_mask[:, None, None, :],
                (
                    batch_size,
                    self.attention_heads,
                    candidate_hidden.shape[1],
                    candidate_hidden.shape[1],
                ),
            ),
        )
        attended_candidates = nn.LayerNorm(name="cand_norm")(
            candidate_hidden + attended_candidates
        )

        context_query = mlp(
            jnp.concatenate([self_hidden, global_hidden], axis=-1),
            self.hidden_size,
            self.hidden_size,
            "context_query",
        )[:, None, :]
        attended_context = attention_module(
            num_heads=self.attention_heads,
            qkv_features=self.hidden_size,
            out_features=self.hidden_size,
            name="context_attention",
        )(
            context_query,
            attended_candidates,
            mask=jnp.broadcast_to(
                safe_mask[:, None, None, :],
                (batch_size, self.attention_heads, 1, candidate_hidden.shape[1]),
            ),
        )
        attended_context = nn.LayerNorm(name="context_norm")(
            context_query + attended_context
        ).squeeze(axis=1)

        pooled_candidates = masked_mean(attended_candidates, candidate_mask)
        value_input = jnp.concatenate(
            [self_hidden, global_hidden, attended_context, pooled_candidates], axis=-1
        )

        return EncoderOutput(
            attended_candidates=attended_candidates,
            context_query=attended_context,
            value_input=value_input,
        )


class GNNBackboneEncoder(nn.Module):
    """Graph Neural Network feature extractor using K-Nearest Neighbor message passing.

    This encoder treats the Orbit Wars map as a geometric network graph, allowing
    planets to exchange state information with their closest neighbors.
    """

    hidden_size: int = 128
    k_neighbors: int = 5
    msg_passing_layers: int = 2
    target_coords_span: tuple[int, int] = (4, 6)

    def setup(self) -> None:
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be at least 1.")
        if self.msg_passing_layers < 1:
            raise ValueError("msg_passing_layers must be at least 1.")

    @nn.compact
    def __call__(
        self,
        self_features: jax.Array,
        candidate_features: jax.Array,
        global_features: jax.Array,
        candidate_mask: jax.Array,
    ) -> EncoderOutput:
        num_planets = candidate_features.shape[1]

        # 1. Encode base entity representations
        self_hidden = mlp(self_features, self.hidden_size, self.hidden_size, "self_enc")
        global_hidden = mlp(
            global_features, self.hidden_size, self.hidden_size, "global_enc"
        )
        candidate_hidden = mlp(
            candidate_features, self.hidden_size, self.hidden_size, "candidate_enc"
        )

        # 2. Extract current-frame normalized spatial coordinates
        coords = candidate_features[..., slice(*self.target_coords_span)]

        # 3. Pairwise Euclidean distance tracking via implicit broadcasting
        diffs = coords[:, :, None, :] - coords[:, None, :, :]
        dist_matrix = jnp.sum(diffs**2, axis=-1)

        # 4. Extract neighbor connections and mask out dead padded elements
        neighbor_count = min(self.k_neighbors, num_planets)
        _, topk_indices = jax.lax.top_k(-dist_matrix, k=neighbor_count)
        adj_matrix = jnp.sum(jax.nn.one_hot(topk_indices, num_planets), axis=-2).astype(
            bool
        )

        final_adj_mask = adj_matrix & (
            candidate_mask[:, :, None] & candidate_mask[:, None, :]
        )

        # 5. Message Passing Execution Loop
        current_node_states = candidate_hidden
        for layer_idx in range(self.msg_passing_layers):
            msg_proj = nn.Dense(self.hidden_size, name=f"msg_proj_{layer_idx}")(
                current_node_states
            )
            masked_messages = jnp.where(
                final_adj_mask[..., None], msg_proj[:, None, :, :], 0.0
            )
            aggregated_messages = jnp.sum(masked_messages, axis=2)

            combined_node_input = jnp.concatenate(
                [current_node_states, aggregated_messages], axis=-1
            )
            current_node_states = mlp(
                combined_node_input,
                self.hidden_size,
                self.hidden_size,
                f"gnn_layer_{layer_idx}",
            )
            current_node_states = nn.LayerNorm(name=f"gnn_norm_{layer_idx}")(
                candidate_hidden + current_node_states
            )

        # 6. Contract Assembly
        pooled_candidates = masked_mean(current_node_states, candidate_mask)
        context_query = jnp.concatenate([self_hidden, global_hidden], axis=-1)
        value_input = jnp.concatenate([context_query, pooled_candidates], axis=-1)

        return EncoderOutput(
            attended_candidates=current_node_states,
            context_query=context_query,
            value_input=value_input,
        )


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


# --- Composable Policy Wrapper ---
class ComposablePlanetPolicy(nn.Module):
    """The master framework container that unifies injected backbones and decoders."""

    encoder_module: nn.Module
    decoder_module: nn.Module
    value_head_module: nn.Module | None = None
    hidden_size: int = 128

    @nn.compact
    def __call__(
        self,
        self_features: jax.Array,
        candidate_features: jax.Array,
        global_features: jax.Array,
        candidate_mask: jax.Array,
        player_count: jax.Array | None = None,
        target_sequence: jax.Array | None = None,
        rng: jax.Array | None = None,
        deterministic: bool = False,
    ) -> JaxPolicyOutput:

        # Route raw arrays through selected encoder backbone to produce structured intermediate representations
        encoder_out = self.encoder_module(
            self_features, candidate_features, global_features, candidate_mask
        )

        # Route encoder outputs through selected decoder head to produce final action logits
        target_logits, ship_logits, decoded_target_sequence = self.decoder_module(
            encoder_out,
            candidate_mask,
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


class JaxPlanetPolicy(nn.Module):
    """Composable MLP policy/value network for fixed-shape JAX decisions."""

    candidate_count: int
    ship_bucket_count: int
    value_head_module: nn.Module | None = None
    hidden_size: int = 128

    @nn.compact
    def __call__(
        self,
        self_features: jax.Array,
        candidate_features: jax.Array,
        global_features: jax.Array,
        candidate_mask: jax.Array,
        player_count: jax.Array | None = None,
        target_sequence: jax.Array | None = None,
        rng: jax.Array | None = None,
        deterministic: bool = False,
    ) -> JaxPolicyOutput:
        del self.candidate_count

        return ComposablePlanetPolicy(
            encoder_module=MLPBackboneEncoder(hidden_size=self.hidden_size),
            decoder_module=FeedForwardActionDecoder(
                ship_bucket_count=self.ship_bucket_count, hidden_size=self.hidden_size
            ),
            value_head_module=self.value_head_module,
            hidden_size=self.hidden_size,
        )(
            self_features,
            candidate_features,
            global_features,
            candidate_mask,
            player_count=player_count,
            target_sequence=target_sequence,
            rng=rng,
            deterministic=deterministic,
        )


class JaxAttentionPlanetPolicy(nn.Module):
    """Composable attention/transformer policy for fixed-shape JAX batches."""

    candidate_count: int
    ship_bucket_count: int
    value_head_module: nn.Module | None = None
    hidden_size: int = 128
    attention_heads: int = 4
    enable_gradient_checkpointing: bool = False

    def setup(self) -> None:
        """Validate attention hyperparameters before parameter initialization."""

        if self.hidden_size % self.attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"attention_heads ({self.attention_heads})."
            )

    @nn.compact
    def __call__(
        self,
        self_features: jax.Array,
        candidate_features: jax.Array,
        global_features: jax.Array,
        candidate_mask: jax.Array,
        player_count: jax.Array | None = None,
        target_sequence: jax.Array | None = None,
        rng: jax.Array | None = None,
        deterministic: bool = False,
    ) -> JaxPolicyOutput:
        del self.candidate_count

        return ComposablePlanetPolicy(
            encoder_module=TransformerBackboneEncoder(
                hidden_size=self.hidden_size,
                attention_heads=self.attention_heads,
                enable_gradient_checkpointing=self.enable_gradient_checkpointing,
            ),
            decoder_module=FeedForwardActionDecoder(
                ship_bucket_count=self.ship_bucket_count, hidden_size=self.hidden_size
            ),
            value_head_module=self.value_head_module,
            hidden_size=self.hidden_size,
        )(
            self_features,
            candidate_features,
            global_features,
            candidate_mask.astype(bool),
            player_count=player_count,
            target_sequence=target_sequence,
            rng=rng,
            deterministic=deterministic,
        )

# --- Helper Functions ---

def mlp(x: jax.Array, hidden_size: int, output_size: int, name: str) -> jax.Array:
    """Apply a named two-layer ReLU MLP block."""

    x = nn.Dense(hidden_size, name=f"{name}_0")(x)
    x = nn.relu(x)
    x = nn.Dense(output_size, name=f"{name}_1")(x)
    return nn.relu(x)


def safe_attention_mask(candidate_mask: jax.Array) -> jax.Array:
    """Ensure attention has at least one unmasked key for every batch row."""

    has_valid_key = candidate_mask.any(axis=-1, keepdims=True)
    return jnp.where(has_valid_key, candidate_mask, jnp.ones_like(candidate_mask))


def masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    """Average candidate embeddings while ignoring masked candidate slots."""

    weights = mask.astype(values.dtype)[..., None]
    total = (values * weights).sum(axis=1)
    count = jnp.maximum(weights.sum(axis=1), 1.0)
    return total / count


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

def build_jax_policy(
    cfg: TrainConfig,
) -> nn.Module:
    """Construct a JAX policy module for the requested architecture.

    ``architecture='transformer'`` is accepted as an alias for the attention
    implementation.
    """
    hidden = cfg.model.hidden_size
    buckets = cfg.task.ship_bucket_count
    attention_heads = cfg.model.attention_heads
    k_steps = cfg.model.max_moves_k
    value_head_module = build_value_head(cfg)
    target_coords_slice = candidate_feature_schema(cfg.task).slice("target_coords")
    normalized_architecture = cfg.model.architecture.strip().lower()
    if normalized_architecture == "mlp":
        return JaxPlanetPolicy(
            candidate_count=cfg.task.candidate_count,
            ship_bucket_count=cfg.task.ship_bucket_count,
            value_head_module=value_head_module,
            hidden_size=cfg.model.hidden_size,
        )
    elif normalized_architecture in {"attention", "transformer"}:
        return JaxAttentionPlanetPolicy(
            candidate_count=cfg.task.candidate_count,
            ship_bucket_count=cfg.task.ship_bucket_count,
            value_head_module=value_head_module,
            hidden_size=cfg.model.hidden_size,
            attention_heads=cfg.model.attention_heads,
            enable_gradient_checkpointing=cfg.training.enable_gradient_checkpointing,
        )
    elif normalized_architecture == "mlp_pointer":
        return ComposablePlanetPolicy(
            encoder_module=MLPBackboneEncoder(hidden_size=hidden),
            decoder_module=AutoregressivePointerDecoder(
                ship_bucket_count=buckets, max_moves_k=k_steps, hidden_size=hidden
            ),
            value_head_module=value_head_module,
            hidden_size=hidden,
        )
    elif normalized_architecture == "transformer_pointer":
        return ComposablePlanetPolicy(
            encoder_module=TransformerBackboneEncoder(
                hidden_size=hidden,
                attention_heads=attention_heads,
                enable_gradient_checkpointing=cfg.training.enable_gradient_checkpointing,
            ),
            decoder_module=AutoregressivePointerDecoder(
                ship_bucket_count=buckets, max_moves_k=k_steps, hidden_size=hidden
            ),
            value_head_module=value_head_module,
            hidden_size=hidden,
        )
    elif normalized_architecture == "gnn_pointer":
        return ComposablePlanetPolicy(
            encoder_module=GNNBackboneEncoder(
                hidden_size=hidden,
                k_neighbors=cfg.model.gnn_k_neighbors,
                msg_passing_layers=cfg.model.gnn_message_passing_layers,
                target_coords_span=(target_coords_slice.start, target_coords_slice.stop),
            ),
            decoder_module=AutoregressivePointerDecoder(
                ship_bucket_count=buckets, max_moves_k=k_steps, hidden_size=hidden
            ),
            value_head_module=value_head_module,
            hidden_size=hidden,
        )
    else:
        raise ValueError(
            f"Unsupported JAX model architecture '{cfg.model.architecture}'. Expected 'mlp', "
            "'attention', 'transformer', 'mlp_pointer', 'transformer_pointer', or 'gnn_pointer'."
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


def action_log_prob_and_entropy(
    output: JaxPolicyOutput,
    target_index: jax.Array,
    ship_bucket: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute joint log-probability and entropy for target/bucket actions."""

    squeeze_sequence = target_index.ndim == 1
    target_logits = ensure_policy_sequence(output.target_logits)
    ship_logits = ensure_policy_sequence(output.ship_logits)
    target_index = ensure_action_sequence(target_index)
    ship_bucket = ensure_action_sequence(ship_bucket)
    target_log_probs = jax.nn.log_softmax(target_logits, axis=-1)
    target_probs = jax.nn.softmax(target_logits, axis=-1)
    target_lp = jnp.take_along_axis(
        target_log_probs, target_index[..., None], axis=-1
    ).squeeze(-1)
    selected_ship_logits = jnp.take_along_axis(
        ship_logits,
        target_index[..., None, None].repeat(ship_logits.shape[-1], axis=-1),
        axis=2,
    ).squeeze(axis=2)
    ship_log_probs = jax.nn.log_softmax(selected_ship_logits, axis=-1)
    ship_probs = jax.nn.softmax(selected_ship_logits, axis=-1)
    ship_lp = jnp.take_along_axis(
        ship_log_probs, ship_bucket[..., None], axis=-1
    ).squeeze(-1)
    target_entropy = -(target_probs * target_log_probs).sum(axis=-1)
    ship_entropy = -(ship_probs * ship_log_probs).sum(axis=-1)
    log_prob = target_lp + ship_lp
    entropy = target_entropy + ship_entropy
    if squeeze_sequence:
        return log_prob[:, 0], entropy[:, 0]
    return log_prob, entropy


def ensure_policy_sequence(value: jax.Array) -> jax.Array:
    """Represent policy logits with an explicit sequence axis."""

    if value.ndim == 2:
        return value[:, None, :]
    if value.ndim == 3:
        return value
    return value


def ensure_action_sequence(value: jax.Array) -> jax.Array:
    """Represent sampled action ids with an explicit sequence axis."""

    if value.ndim == 1:
        return value[:, None]
    return value


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
