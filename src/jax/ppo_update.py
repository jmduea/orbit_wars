from __future__ import annotations

import optax

import jax
import jax.numpy as jnp

from src.config import TrainConfig
from src.features.registry import (
    candidate_feature_dim,
    global_feature_dim,
    self_feature_dim,
)
from src.game.trajectory_shield import mask_policy_output_for_shield

from .features import JaxTurnBatch
from .policy import action_log_prob_and_entropy
from .rollout.types import JaxTransitionBatch, JaxTrainState


def flatten_batch(
    batch: JaxTurnBatch,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Flatten environment/source dimensions into policy decision rows."""

    return (
        batch.self_features.reshape(-1, batch.self_features.shape[-1]),
        batch.candidate_features.reshape(
            -1, batch.candidate_features.shape[-2], batch.candidate_features.shape[-1]
        ),
        batch.global_features.reshape(-1, batch.global_features.shape[-1]),
        batch.candidate_mask.reshape(-1, batch.candidate_mask.shape[-1]),
        batch.decision_mask.reshape(-1),
    )


def concatenate_transition_batches(
    batches: tuple[JaxTransitionBatch, ...] | list[JaxTransitionBatch],
) -> JaxTransitionBatch:
    """Concatenate compatible rollout batches along the environment axis.

    Mixed-format JAX training uses one compiled collector per static player
    count. The resulting transition tensors share rollout and feature shapes,
    so PPO can consume a single larger batch by joining their independent
    environment axes.
    """

    if not batches:
        raise ValueError("At least one transition batch is required.")
    if len(batches) == 1:
        return batches[0]
    reference_shape = batches[0].self_features.shape
    for batch in batches[1:]:
        if (
            batch.self_features.shape[0] != reference_shape[0]
            or batch.self_features.shape[2:] != reference_shape[2:]
        ):
            raise ValueError(
                "Transition batches must share rollout and feature dimensions "
                "to concatenate along the environment axis."
            )
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *batches)


def discounted_returns(rewards: jax.Array, done: jax.Array, gamma: float) -> jax.Array:
    """Compute discounted returns over rollout time with terminal resets."""

    def step(carry, item):
        reward, terminal = item
        carry = reward + gamma * carry * (1.0 - terminal.astype(jnp.float32))
        return carry, carry

    _, out = jax.lax.scan(
        step, jnp.zeros_like(rewards[-1]), (rewards, done), reverse=True
    )
    return out


def ppo_update_jax(
    train_state: JaxTrainState,
    policy: object,
    batch: JaxTransitionBatch,
    cfg: TrainConfig,
) -> tuple[JaxTrainState, dict[str, jax.Array]]:
    """Apply one PPO epoch using memory-bounded minibatches.

    Rollouts can be large when benchmarking long attention-policy runs. Running
    the policy over every rollout row in a single XLA program forces the GPU to
    materialize attention intermediates for the entire rollout at once. Instead,
    flatten once, pad to static memory chunks, and scan sequential optimizer
    steps over those chunks. The chunk size honors large configured minibatches
    and the ``training.update_chunk_rows_min``/``training.update_chunk_rows_max``
    limits so rollouts can trade memory pressure for throughput.
    """

    sequence_k = batch.target_index.shape[-1]
    mask = batch.decision_mask.reshape(-1, sequence_k).astype(jnp.float32)
    self_features = batch.self_features.reshape(-1, self_feature_dim(cfg.task))
    candidate_features = batch.candidate_features.reshape(
        -1, cfg.task.candidate_count, candidate_feature_dim(cfg.task)
    )
    global_features = batch.global_features.reshape(-1, global_feature_dim(cfg.task))
    candidate_mask = batch.candidate_mask.reshape(-1, cfg.task.candidate_count)
    player_count = batch.player_count.reshape(-1)
    ship_bucket_mask = batch.ship_bucket_mask.reshape(
        -1, sequence_k, cfg.task.candidate_count, cfg.task.ship_bucket_count
    )
    target = batch.target_index.reshape(-1, sequence_k)
    bucket = batch.ship_bucket.reshape(-1, sequence_k)
    old_log_prob = batch.log_prob.reshape(-1, sequence_k)
    returns = batch.returns.reshape(-1, sequence_k)
    advantages = batch.advantages.reshape(-1, sequence_k)
    advantage_mean = masked_mean(advantages, mask)
    advantages = (advantages - advantage_mean) / jnp.sqrt(
        masked_mean((advantages - advantage_mean) ** 2, mask) + 1e-8
    )

    total_rows = mask.shape[0]
    min_chunk_rows = int(cfg.training.update_chunk_rows_min)
    max_chunk_rows = (
        int(cfg.training.update_chunk_rows_max)
        if cfg.training.update_chunk_rows_max is not None
        else total_rows
    )
    chunk_target = max(int(cfg.training.minibatch_size), min_chunk_rows)
    minibatch_size = min(max(chunk_target, 1), max_chunk_rows, total_rows)
    minibatch_count = (total_rows + minibatch_size - 1) // minibatch_size
    minibatches = {
        "mask": _reshape_minibatches(mask, minibatch_count, minibatch_size, 0.0),
        "self_features": _reshape_minibatches(
            self_features, minibatch_count, minibatch_size, 0.0
        ),
        "candidate_features": _reshape_minibatches(
            candidate_features, minibatch_count, minibatch_size, 0.0
        ),
        "global_features": _reshape_minibatches(
            global_features, minibatch_count, minibatch_size, 0.0
        ),
        "candidate_mask": _reshape_minibatches(
            candidate_mask, minibatch_count, minibatch_size, False
        ),
        "player_count": _reshape_minibatches(
            player_count, minibatch_count, minibatch_size, 0
        ),
        "ship_bucket_mask": _reshape_minibatches(
            ship_bucket_mask, minibatch_count, minibatch_size, False
        ),
        "target": _reshape_minibatches(target, minibatch_count, minibatch_size, 0),
        "bucket": _reshape_minibatches(bucket, minibatch_count, minibatch_size, 0),
        "old_log_prob": _reshape_minibatches(
            old_log_prob, minibatch_count, minibatch_size, 0.0
        ),
        "returns": _reshape_minibatches(returns, minibatch_count, minibatch_size, 0.0),
        "advantages": _reshape_minibatches(
            advantages, minibatch_count, minibatch_size, 0.0
        ),
    }

    def update_minibatch(carry, minibatch):
        params, opt_state = carry

        def loss_fn(params):
            output = policy.apply(
                params,
                minibatch["self_features"],
                minibatch["candidate_features"],
                minibatch["global_features"],
                minibatch["candidate_mask"],
                player_count=minibatch["player_count"],
                target_sequence=minibatch["target"],
            )
            output = mask_policy_output_for_shield(
                output,
                minibatch["candidate_mask"],
                cfg.task.ship_bucket_count,
                minibatch["ship_bucket_mask"],
            )
            new_log_prob, entropy = action_log_prob_and_entropy(
                output, minibatch["target"], minibatch["bucket"]
            )
            approx_kl = masked_mean(
                minibatch["old_log_prob"] - new_log_prob,
                minibatch["mask"],
            )
            ratio = jnp.exp(new_log_prob - minibatch["old_log_prob"])
            clipped_ratio = jnp.clip(
                ratio, 1.0 - cfg.training.clip_coef, 1.0 + cfg.training.clip_coef
            )
            policy_objective = jnp.minimum(
                minibatch["advantages"] * ratio,
                minibatch["advantages"] * clipped_ratio,
            )
            value_error = (minibatch["returns"] - output.value[:, None]) ** 2
            policy_loss = -masked_mean(policy_objective, minibatch["mask"])
            value_loss = 0.5 * masked_mean(value_error, minibatch["mask"])
            entropy_loss = masked_mean(entropy, minibatch["mask"])
            loss = (
                policy_loss
                + cfg.training.vf_coef * value_loss
                - cfg.training.ent_coef * entropy_loss
            )
            metrics = {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy_loss,
                "approx_kl": approx_kl,
                "loss": loss,
                "sample_count": minibatch["mask"].sum(),
            }
            for player_count in (2, 4):
                suffix = f"{player_count}p"
                format_mask = minibatch["mask"] * (
                    minibatch["player_count"][:, None] == player_count
                ).astype(jnp.float32)
                format_policy_loss = -masked_mean(policy_objective, format_mask)
                format_value_loss = 0.5 * masked_mean(value_error, format_mask)
                format_entropy = masked_mean(entropy, format_mask)
                format_approx_kl = masked_mean(
                    minibatch["old_log_prob"] - new_log_prob,
                    format_mask,
                )
                format_total_loss = (
                    format_policy_loss
                    + cfg.training.vf_coef * format_value_loss
                    - cfg.training.ent_coef * format_entropy
                )
                metrics[f"policy_loss_{suffix}"] = format_policy_loss
                metrics[f"value_loss_{suffix}"] = format_value_loss
                metrics[f"entropy_{suffix}"] = format_entropy
                metrics[f"approx_kl_{suffix}"] = format_approx_kl
                metrics[f"total_loss_{suffix}"] = format_total_loss
                metrics[f"loss_sample_count_{suffix}"] = format_mask.sum()
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = train_state.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics = dict(metrics)
        metrics["total_loss"] = loss
        return (params, opt_state), metrics

    (params, opt_state), metrics_by_minibatch = jax.lax.scan(
        update_minibatch, (train_state.params, train_state.opt_state), minibatches
    )
    format_metric_names = frozenset(
        f"{metric_name}_{suffix}"
        for suffix in ("2p", "4p")
        for metric_name in (
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "total_loss",
        )
    )
    format_sample_names = frozenset(
        f"loss_sample_count_{suffix}" for suffix in ("2p", "4p")
    )
    metric_weights = jnp.where(metrics_by_minibatch["sample_count"] > 0.0, 1.0, 0.0)
    metric_denominator = jnp.maximum(metric_weights.sum(), 1.0)
    metrics = {
        name: (values * metric_weights).sum() / metric_denominator
        for name, values in metrics_by_minibatch.items()
        if name not in {"sample_count", *format_metric_names, *format_sample_names}
    }
    for suffix in ("2p", "4p"):
        sample_name = f"loss_sample_count_{suffix}"
        sample_counts = metrics_by_minibatch[sample_name]
        sample_denominator = jnp.maximum(sample_counts.sum(), 1.0)
        metrics[sample_name] = sample_counts.sum()
        for metric_name in (
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "total_loss",
        ):
            name = f"{metric_name}_{suffix}"
            metrics[name] = (
                metrics_by_minibatch[name] * sample_counts
            ).sum() / sample_denominator
    metrics["minibatches"] = jnp.array(minibatch_count, dtype=jnp.float32)
    return (
        JaxTrainState(
            params=params, opt_state=opt_state, optimizer=train_state.optimizer
        ),
        metrics,
    )


def _reshape_minibatches(
    value: jax.Array,
    minibatch_count: int,
    minibatch_size: int,
    padding_value: float | int | bool,
) -> jax.Array:
    """Pad and reshape a flat leading axis into static minibatches."""

    padded_rows = minibatch_count * minibatch_size
    pad_rows = padded_rows - value.shape[0]
    pad_width = [(0, pad_rows)] + [(0, 0)] * (value.ndim - 1)
    padded = jnp.pad(value, pad_width, constant_values=padding_value)
    return padded.reshape((minibatch_count, minibatch_size) + value.shape[1:])


def masked_mean(x: jax.Array, mask: jax.Array) -> jax.Array:
    """Return the mean of ``x`` over entries where ``mask`` is non-zero."""

    return (x * mask).sum() / jnp.maximum(mask.sum(), 1.0)
