from __future__ import annotations

from typing import NamedTuple

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
    PlanetFlowPolicyOutput,
    action_log_prob_and_entropy,
    ensure_policy_sequence,
)
from src.jax.decoders.planet_flow import PlanetFlowTargetDemandHead
from src.jax.decoders.factorized_topk_pointer import FactorizedTopKPointerDecoder
from src.jax.distributional_value import (
    expected_value_from_logits,
    value_support,
)
from src.jax.encoders import EncoderOutput
from src.jax.encoders.planet_encoder_common import PlanetEdgeEncoderOutput
from src.jax.encoders.planet_graph_transformer import PlanetGraphTransformerEncoder
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


class ValueHeadOutput(NamedTuple):
    """Scalar bootstrap value plus optional distributional logits."""

    value: jax.Array
    value_logits: jax.Array | None = None


class SharedValueHead(nn.Module):
    """Single critic head shared across all training formats."""

    hidden_size: int = 128

    @nn.compact
    def __call__(
        self,
        value_input: jax.Array,
        player_count: jax.Array | None = None,
    ) -> ValueHeadOutput:
        del player_count
        value_hidden = nn.relu(
            nn.Dense(self.hidden_size, name="shared_value_dense")(value_input)
        )
        value = nn.Dense(1, name="shared_value_out")(value_hidden).squeeze(-1)
        return ValueHeadOutput(value=value, value_logits=None)


class FormatRoutedValueHead(nn.Module):
    """Select between dedicated 2p and 4p critic heads per batch row."""

    hidden_size: int = 128

    @nn.compact
    def __call__(
        self,
        value_input: jax.Array,
        player_count: jax.Array | None = None,
    ) -> ValueHeadOutput:
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

        value = jnp.where(player_count == 2, two_player_value, four_player_value)
        return ValueHeadOutput(value=value, value_logits=None)


class CategoricalValueHead(nn.Module):
    """C51-style categorical return distribution with scalar expected value."""

    hidden_size: int = 128
    value_bins: int = 51
    value_max: float = 1.0

    @nn.compact
    def __call__(
        self,
        value_input: jax.Array,
        player_count: jax.Array | None = None,
    ) -> ValueHeadOutput:
        del player_count
        if self.value_bins < 2:
            raise ValueError("CategoricalValueHead requires value_bins >= 2.")
        if self.value_max <= 0.0:
            raise ValueError("CategoricalValueHead requires value_max > 0.")
        value_hidden = nn.relu(
            nn.Dense(self.hidden_size, name="distributional_value_dense")(value_input)
        )
        value_logits = nn.Dense(self.value_bins, name="distributional_value_out")(
            value_hidden
        )
        support = value_support(self.value_bins, self.value_max)
        value = expected_value_from_logits(value_logits, support)
        return ValueHeadOutput(value=value, value_logits=value_logits)


# --- Helper Functions ---


def safe_attention_mask(candidate_mask: jax.Array) -> jax.Array:
    """Ensure attention has at least one unmasked key for every batch row."""

    has_valid_key = candidate_mask.any(axis=-1, keepdims=True)
    return jnp.where(has_valid_key, candidate_mask, jnp.ones_like(candidate_mask))


def build_value_head(cfg: TrainConfig) -> nn.Module:
    """Construct the configured critic head module."""

    normalized_value_head = cfg.model.value_head.strip().lower()
    hidden = cfg.model.hidden_size
    if normalized_value_head == "shared":
        return SharedValueHead(hidden_size=hidden)
    if normalized_value_head == "format_routed":
        return FormatRoutedValueHead(hidden_size=hidden)
    if normalized_value_head == "distributional":
        return CategoricalValueHead(
            hidden_size=hidden,
            value_bins=cfg.model.value_bins,
            value_max=cfg.model.value_max,
        )
    raise ValueError(
        "Unsupported value head "
        f"'{cfg.model.value_head}'. Expected 'shared', 'format_routed', or "
        "'distributional'."
    )


def is_distributional_value_head(cfg: TrainConfig) -> bool:
    """Return True when the configured critic uses categorical return logits."""

    return cfg.model.value_head.strip().lower() == "distributional"


class ComposableFactorizedPlanetPolicy(nn.Module):
    """Encoder + factorized top-K pointer decoder for v2 planet-edge batches."""

    encoder_module: nn.Module
    decoder_module: FactorizedTopKPointerDecoder
    value_head_module: nn.Module | None = None
    decoder_carry: bool = False
    hidden_size: int = 128

    def _resolve_value_head(self) -> nn.Module:
        value_head_module = self.value_head_module
        if value_head_module is None:
            value_head_module = SharedValueHead(hidden_size=self.hidden_size)
        return value_head_module

    def encode(self, batch: TurnBatch) -> PlanetEdgeEncoderOutput:
        """Run the planet graph encoder once per fixed ``TurnBatch``."""

        return self.encoder_module(batch)

    def critic(
        self,
        encoder_out: PlanetEdgeEncoderOutput,
        player_count: jax.Array | None = None,
    ) -> ValueHeadOutput:
        """Bootstrap value from cached encoder output."""

        return self._resolve_value_head()(
            encoder_out.value_input, player_count=player_count
        )

    def decode(
        self,
        encoder_out: PlanetEdgeEncoderOutput,
        *,
        player_count: jax.Array | None = None,
        source_sequence: jax.Array | None = None,
        target_slot_sequence: jax.Array | None = None,
        decoder_hidden: jax.Array | None = None,
        rng: jax.Array | None = None,
        deterministic: bool = False,
        include_value: bool = True,
    ) -> FactoredPolicyOutput:
        """Decoder-only forward on cached ``encoder_out``."""

        (
            source_logits,
            target_logits,
            stop_logits,
            ship_logits,
            decoded_source_sequence,
            decoded_target_slot_sequence,
            decoded_stop_sequence,
            decoder_hidden_out,
        ) = self.decoder_module(
            encoder_out,
            source_sequence=source_sequence,
            target_slot_sequence=target_slot_sequence,
            decoder_hidden_in=decoder_hidden,
            rng=rng,
            deterministic=deterministic,
        )

        if include_value:
            value_out = self.critic(encoder_out, player_count=player_count)
            value = value_out.value
            value_logits = value_out.value_logits
        else:
            batch_size = encoder_out.context_query.shape[0]
            value = jnp.zeros((batch_size,), dtype=jnp.float32)
            value_logits = None

        return FactoredPolicyOutput(
            source_logits=source_logits,
            target_logits=target_logits,
            stop_logits=stop_logits,
            ship_logits=ship_logits,
            value=value,
            decoded_source_sequence=decoded_source_sequence,
            decoded_target_slot_sequence=decoded_target_slot_sequence,
            decoded_stop_sequence=decoded_stop_sequence,
            value_logits=value_logits,
            decoder_hidden=decoder_hidden_out if self.decoder_carry else None,
        )

    @nn.compact
    def __call__(
        self,
        batch: TurnBatch,
        player_count: jax.Array | None = None,
        source_sequence: jax.Array | None = None,
        target_slot_sequence: jax.Array | None = None,
        decoder_hidden: jax.Array | None = None,
        rng: jax.Array | None = None,
        deterministic: bool = False,
    ) -> FactoredPolicyOutput:
        encoder_out = self.encode(batch)
        return self.decode(
            encoder_out,
            player_count=player_count,
            source_sequence=source_sequence,
            target_slot_sequence=target_slot_sequence,
            decoder_hidden=decoder_hidden,
            rng=rng,
            deterministic=deterministic,
            include_value=True,
        )


class ComposablePlanetFlowPolicy(nn.Module):
    """Encoder + target-demand pressure head for Planet Flow P0."""

    encoder_module: nn.Module
    demand_head_module: PlanetFlowTargetDemandHead
    value_head_module: nn.Module | None = None
    hidden_size: int = 128

    def _resolve_value_head(self) -> nn.Module:
        value_head_module = self.value_head_module
        if value_head_module is None:
            value_head_module = SharedValueHead(hidden_size=self.hidden_size)
        return value_head_module

    def encode(self, batch: TurnBatch) -> PlanetEdgeEncoderOutput:
        """Run the planet graph encoder once per fixed ``TurnBatch``."""

        return self.encoder_module(batch)

    def critic(
        self,
        encoder_out: PlanetEdgeEncoderOutput,
        player_count: jax.Array | None = None,
    ) -> ValueHeadOutput:
        """Bootstrap value from cached encoder output."""

        return self._resolve_value_head()(
            encoder_out.value_input, player_count=player_count
        )

    def decode(
        self,
        encoder_out: PlanetEdgeEncoderOutput,
        *,
        player_count: jax.Array | None = None,
        include_value: bool = True,
    ) -> PlanetFlowPolicyOutput:
        """Decoder-only forward on cached ``encoder_out``."""

        target_demand_logits = self.demand_head_module(encoder_out)
        if include_value:
            value_out = self.critic(encoder_out, player_count=player_count)
            value = value_out.value
            value_logits = value_out.value_logits
        else:
            batch_size = encoder_out.context_query.shape[0]
            value = jnp.zeros((batch_size,), dtype=jnp.float32)
            value_logits = None
        return PlanetFlowPolicyOutput(
            target_demand_logits=target_demand_logits,
            value=value,
            value_logits=value_logits,
        )

    @nn.compact
    def __call__(
        self,
        batch: TurnBatch,
        player_count: jax.Array | None = None,
        rng: jax.Array | None = None,
        deterministic: bool = False,
    ) -> PlanetFlowPolicyOutput:
        del rng, deterministic
        encoder_out = self.encode(batch)
        return self.decode(
            encoder_out,
            player_count=player_count,
            include_value=True,
        )


def factorized_encode(
    params: dict,
    policy: nn.Module,
    batch: TurnBatch,
) -> PlanetEdgeEncoderOutput:
    """Encode ``TurnBatch`` without running the decoder."""

    return policy.apply(params, batch, method=ComposableFactorizedPlanetPolicy.encode)


def factorized_critic(
    params: dict,
    policy: nn.Module,
    encoder_out: PlanetEdgeEncoderOutput,
    *,
    player_count: jax.Array | None = None,
) -> ValueHeadOutput:
    """Run the critic head on cached encoder output."""

    return policy.apply(
        params,
        encoder_out,
        player_count=player_count,
        method=ComposableFactorizedPlanetPolicy.critic,
    )


def factorized_decode(
    params: dict,
    policy: nn.Module,
    encoder_out: PlanetEdgeEncoderOutput,
    *,
    player_count: jax.Array | None = None,
    source_sequence: jax.Array | None = None,
    target_slot_sequence: jax.Array | None = None,
    decoder_hidden: jax.Array | None = None,
    rng: jax.Array | None = None,
    deterministic: bool = False,
    include_value: bool = False,
) -> FactoredPolicyOutput:
    """Decoder-only forward on cached encoder output."""

    return policy.apply(
        params,
        encoder_out,
        player_count=player_count,
        source_sequence=source_sequence,
        target_slot_sequence=target_slot_sequence,
        decoder_hidden=decoder_hidden,
        rng=rng,
        deterministic=deterministic,
        include_value=include_value,
        method=ComposableFactorizedPlanetPolicy.decode,
    )


def edge_action_count(task_cfg) -> int:
    """Flat edge logits including the always-legal NO_OP slot."""

    return MAX_PLANETS * edge_k(task_cfg) + 1


def build_pointer_decoder(cfg: TrainConfig) -> FactorizedTopKPointerDecoder:
    """Construct the factorized top-K pointer decoder module."""

    from src.artifacts.checkpoint_compat import (
        POINTER_DECODER_FACTORIZED_TOPK,
        pointer_decoder_for_model,
    )

    pointer_decoder = pointer_decoder_for_model(cfg.model)
    if pointer_decoder != POINTER_DECODER_FACTORIZED_TOPK:
        raise ValueError(
            "build_pointer_decoder only supports factorized decoders, got "
            f"{pointer_decoder!r}."
        )
    hidden = cfg.model.hidden_size
    return FactorizedTopKPointerDecoder(
        ship_bucket_count=cfg.task.ship_bucket_count,
        ship_action_mode=cfg.task.ship_action_mode,
        decoder_carry=cfg.model.decoder_carry,
        max_moves_k=cfg.model.max_moves_k,
        hidden_size=hidden,
        edge_k=edge_k(cfg.task),
    )


_PLANET_GRAPH_TRANSFORMER_ARCHITECTURES = frozenset(
    {
        "planet_graph_transformer",
        "planet_graph_transformer_small",
    }
)


def build_jax_policy(cfg: TrainConfig) -> nn.Module:
    """Construct the planet-edge policy for the configured architecture."""
    normalized_architecture = cfg.model.architecture.strip().lower()
    if normalized_architecture in _PLANET_GRAPH_TRANSFORMER_ARCHITECTURES:
        return build_planet_graph_transformer_policy(cfg)
    raise ValueError(
        f"Unsupported JAX model architecture '{cfg.model.architecture}'. "
        "Expected one of: "
        + ", ".join(sorted(_PLANET_GRAPH_TRANSFORMER_ARCHITECTURES))
        + "."
    )


def build_planet_graph_transformer_policy(cfg: TrainConfig) -> nn.Module:
    """Construct the planet graph transformer policy (M2 encoder)."""

    from src.artifacts.checkpoint_compat import (
        POINTER_DECODER_FACTORIZED_TOPK,
        POINTER_DECODER_PLANET_FLOW_TARGET_HEATMAP,
        pointer_decoder_for_model,
    )

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
    value_head_module = build_value_head(cfg)
    pointer_decoder = pointer_decoder_for_model(cfg.model)
    if pointer_decoder == POINTER_DECODER_FACTORIZED_TOPK:
        decoder_module = build_pointer_decoder(cfg)
        return ComposableFactorizedPlanetPolicy(
            encoder_module=encoder_module,
            decoder_module=decoder_module,
            value_head_module=value_head_module,
            hidden_size=hidden,
            decoder_carry=cfg.model.decoder_carry,
        )
    if pointer_decoder == POINTER_DECODER_PLANET_FLOW_TARGET_HEATMAP:
        return ComposablePlanetFlowPolicy(
            encoder_module=encoder_module,
            demand_head_module=PlanetFlowTargetDemandHead(
                pressure_bucket_count=len(
                    cfg.model.planet_flow.pressure_bucket_values
                ),
                hidden_size=hidden,
            ),
            value_head_module=value_head_module,
            hidden_size=hidden,
        )
    raise ValueError(f"Unsupported pointer_decoder={pointer_decoder!r}.")


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
            value_logits=output.value_logits,
            decoder_hidden=output.decoder_hidden,
        )
    return output
