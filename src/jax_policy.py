from __future__ import annotations

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp


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
    """

    target_logits: jax.Array
    ship_logits: jax.Array
    value: jax.Array


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


class JaxPlanetPolicy(nn.Module):
    """MLP policy/value network for fixed-shape JAX Orbit Wars decisions.

    The module consumes flattened decision rows and returns one target
    distribution over candidates, one ship-bucket distribution per candidate,
    and one scalar value estimate per row.
    """

    candidate_count: int
    ship_bucket_count: int
    hidden_size: int = 128

    @nn.compact
    def __call__(
        self,
        self_features: jax.Array,
        candidate_features: jax.Array,
        global_features: jax.Array,
        candidate_mask: jax.Array,
    ) -> JaxPolicyOutput:
        self_hidden = mlp(
            self_features, self.hidden_size, self.hidden_size, "self_encoder"
        )
        global_hidden = mlp(
            global_features, self.hidden_size, self.hidden_size, "global_encoder"
        )
        candidate_hidden = mlp(
            candidate_features, self.hidden_size, self.hidden_size, "candidate_encoder"
        )
        expanded_self = jnp.broadcast_to(
            self_hidden[:, None, :], candidate_hidden.shape
        )
        expanded_global = jnp.broadcast_to(
            global_hidden[:, None, :], candidate_hidden.shape
        )
        joint = jnp.concatenate(
            [expanded_self, expanded_global, candidate_hidden], axis=-1
        )
        target_hidden = nn.relu(nn.Dense(self.hidden_size, name="target_dense")(joint))
        target_logits = nn.Dense(1, name="target_out")(target_hidden).squeeze(-1)
        target_logits = jnp.where(
            candidate_mask, target_logits, jnp.finfo(jnp.float32).min
        )
        ship_hidden = nn.relu(nn.Dense(self.hidden_size, name="ship_dense")(joint))
        ship_logits = nn.Dense(self.ship_bucket_count, name="ship_out")(ship_hidden)
        pooled = candidate_hidden.mean(axis=1)
        value_input = jnp.concatenate([self_hidden, global_hidden, pooled], axis=-1)
        value_hidden = nn.relu(
            nn.Dense(self.hidden_size, name="value_dense")(value_input)
        )
        value = nn.Dense(1, name="value_out")(value_hidden).squeeze(-1)
        return JaxPolicyOutput(
            target_logits=target_logits, ship_logits=ship_logits, value=value
        )


class JaxAttentionPlanetPolicy(nn.Module):
    """Flax attention/transformer policy matching the Torch attention interface."""

    candidate_count: int
    ship_bucket_count: int
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
    ) -> JaxPolicyOutput:
        candidate_mask = candidate_mask.astype(bool)
        safe_mask = safe_attention_mask(candidate_mask)
        self_hidden = mlp(
            self_features, self.hidden_size, self.hidden_size, "self_encoder"
        )
        global_hidden = mlp(
            global_features, self.hidden_size, self.hidden_size, "global_encoder"
        )
        candidate_hidden = mlp(
            candidate_features, self.hidden_size, self.hidden_size, "candidate_encoder"
        )

        self_attention_mask = safe_mask[:, None, None, :]
        attention_module = (
            nn.remat(nn.MultiHeadDotProductAttention)
            if self.enable_gradient_checkpointing
            else nn.MultiHeadDotProductAttention
        )
        attended_candidates = attention_module(
            num_heads=self.attention_heads,
            qkv_features=self.hidden_size,
            out_features=self.hidden_size,
            name="candidate_attention",
        )(
            candidate_hidden,
            candidate_hidden,
            mask=jnp.broadcast_to(
                self_attention_mask,
                (
                    candidate_hidden.shape[0],
                    self.attention_heads,
                    self.candidate_count,
                    self.candidate_count,
                ),
            ),
        )
        attended_candidates = nn.LayerNorm(name="target_norm")(
            candidate_hidden + attended_candidates
        )

        context_query = mlp(
            jnp.concatenate([self_hidden, global_hidden], axis=-1),
            self.hidden_size,
            self.hidden_size,
            "context_query",
        )[:, None, :]
        context_attention_mask = jnp.broadcast_to(
            safe_mask[:, None, None, :],
            (candidate_hidden.shape[0], self.attention_heads, 1, self.candidate_count),
        )
        attended_context = attention_module(
            num_heads=self.attention_heads,
            qkv_features=self.hidden_size,
            out_features=self.hidden_size,
            name="context_attention",
        )(
            context_query,
            attended_candidates,
            mask=context_attention_mask,
        )
        attended_context = nn.LayerNorm(name="context_norm")(
            context_query + attended_context
        ).squeeze(axis=1)

        expanded_self = jnp.broadcast_to(
            self_hidden[:, None, :], attended_candidates.shape
        )
        expanded_global = jnp.broadcast_to(
            global_hidden[:, None, :], attended_candidates.shape
        )
        expanded_context = jnp.broadcast_to(
            attended_context[:, None, :], attended_candidates.shape
        )
        target_input = jnp.concatenate(
            [expanded_self, expanded_global, expanded_context, attended_candidates],
            axis=-1,
        )
        target_hidden = nn.relu(
            nn.Dense(self.hidden_size, name="target_dense")(target_input)
        )
        target_logits = nn.Dense(1, name="target_out")(target_hidden).squeeze(-1)
        target_logits = jnp.where(
            candidate_mask, target_logits, jnp.finfo(jnp.float32).min
        )

        pooled_candidates = masked_mean(attended_candidates, candidate_mask)
        ship_hidden = nn.relu(
            nn.Dense(self.hidden_size, name="ship_dense")(target_input)
        )
        ship_logits = nn.Dense(self.ship_bucket_count, name="ship_out")(ship_hidden)
        value_input = jnp.concatenate(
            [self_hidden, global_hidden, attended_context, pooled_candidates], axis=-1
        )
        value_hidden = nn.relu(
            nn.Dense(self.hidden_size, name="value_dense")(value_input)
        )
        value = nn.Dense(1, name="value_out")(value_hidden).squeeze(-1)
        return JaxPolicyOutput(
            target_logits=target_logits, ship_logits=ship_logits, value=value
        )


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


def build_jax_policy(
    *,
    candidate_count: int,
    ship_bucket_count: int,
    hidden_size: int = 128,
    architecture: str = "mlp",
    attention_heads: int = 4,
    enable_gradient_checkpointing: bool = False,
) -> nn.Module:
    """Construct a JAX policy module for the requested architecture.

    ``architecture='transformer'`` is accepted as an alias for the attention
    implementation to match the Torch policy builder.
    """

    normalized_architecture = architecture.strip().lower()
    if normalized_architecture == "mlp":
        return JaxPlanetPolicy(
            candidate_count=candidate_count,
            ship_bucket_count=ship_bucket_count,
            hidden_size=hidden_size,
        )
    if normalized_architecture in {"attention", "transformer"}:
        return JaxAttentionPlanetPolicy(
            candidate_count=candidate_count,
            ship_bucket_count=ship_bucket_count,
            hidden_size=hidden_size,
            attention_heads=attention_heads,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
        )
    raise ValueError(
        f"Unsupported JAX model architecture '{architecture}'. Expected 'mlp', "
        "'attention', or 'transformer'."
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
    target_index = jnp.where(
        deterministic,
        jnp.argmax(output.target_logits, axis=-1),
        jax.random.categorical(key_target, output.target_logits, axis=-1),
    )
    selected_ship_logits = jnp.take_along_axis(
        output.ship_logits,
        target_index[:, None, None].repeat(output.ship_logits.shape[-1], axis=-1),
        axis=1,
    ).squeeze(axis=1)
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

    target_log_probs = jax.nn.log_softmax(output.target_logits, axis=-1)
    target_probs = jax.nn.softmax(output.target_logits, axis=-1)
    target_lp = jnp.take_along_axis(
        target_log_probs, target_index[:, None], axis=-1
    ).squeeze(-1)
    selected_ship_logits = jnp.take_along_axis(
        output.ship_logits,
        target_index[:, None, None].repeat(output.ship_logits.shape[-1], axis=-1),
        axis=1,
    ).squeeze(axis=1)
    ship_log_probs = jax.nn.log_softmax(selected_ship_logits, axis=-1)
    ship_probs = jax.nn.softmax(selected_ship_logits, axis=-1)
    ship_lp = jnp.take_along_axis(
        ship_log_probs, ship_bucket[:, None], axis=-1
    ).squeeze(-1)
    target_entropy = -(target_probs * target_log_probs).sum(axis=-1)
    ship_entropy = -(ship_probs * ship_log_probs).sum(axis=-1)
    return target_lp + ship_lp, target_entropy + ship_entropy
