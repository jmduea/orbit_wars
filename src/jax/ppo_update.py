from __future__ import annotations

import jax.numpy as jnp
import optax

import jax
from src.artifacts.checkpoint_compat import is_factorized_pointer_decoder
from src.config import TrainConfig
from src.game.trajectory_shield import (
    mask_policy_output_for_shield_v2,
)
from src.jax.action_codec import (
    action_log_prob_and_entropy,
    factored_action_log_prob_with_shield,
)
from src.jax.distributional_value import (
    categorical_value_cross_entropy,
    value_support,
)
from src.jax.features import TurnBatch
from src.jax.policy import edge_action_count, is_distributional_value_head
from src.jax.rollout.types import JaxTrainState, JaxTransitionBatch
from src.jax.ship_action import is_continuous_ship_mode


def concatenate_transition_batches(
    batches: tuple[JaxTransitionBatch, ...] | list[JaxTransitionBatch],
) -> JaxTransitionBatch:
    """Concatenate compatible rollout batches along the environment axis."""
    if not batches:
        raise ValueError("At least one transition batch is required.")
    if len(batches) == 1:
        return batches[0]
    reference_shape = batches[0].planet_features.shape
    for batch in batches[1:]:
        if (
            batch.planet_features.shape[0] != reference_shape[0]
            or batch.planet_features.shape[2:] != reference_shape[2:]
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


def gae_returns_and_advantages(
    rewards: jax.Array,
    values: jax.Array,
    done: jax.Array,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute GAE returns and advantages along the rollout time axis."""

    if gae_lambda == 1.0:
        returns = discounted_returns(rewards, done, gamma)
        return returns, returns - values

    next_values = jnp.concatenate([values[1:], jnp.zeros_like(values[:1])], axis=0)
    not_done = 1.0 - done.astype(jnp.float32)

    def step(carry, item):
        reward, value, next_value, active = item
        delta = reward + gamma * next_value * active - value
        gae = delta + gamma * gae_lambda * active * carry
        return gae, gae

    _, advantages = jax.lax.scan(
        step,
        jnp.zeros_like(values[-1]),
        (rewards, values, next_values, not_done),
        reverse=True,
    )
    returns = advantages + values
    return returns, advantages


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

    # jnp.where avoids NaN * 0.0 = NaN from padded or fully masked action paths.
    safe_x = jnp.where(mask > 0, x, 0.0)
    return safe_x.sum() / jnp.maximum(mask.sum(), 1.0)


def _flatten_transition_to_turn_batch(batch: JaxTransitionBatch) -> TurnBatch:
    env_rows = batch.target_index.shape[0] * batch.target_index.shape[1]
    return TurnBatch(
        planet_features=batch.planet_features.reshape(
            env_rows, *batch.planet_features.shape[2:]
        ),
        planet_mask=batch.planet_mask.reshape(env_rows, batch.planet_mask.shape[-1]),
        edge_features=batch.edge_features.reshape(
            env_rows, *batch.edge_features.shape[2:]
        ),
        edge_mask=batch.edge_mask.reshape(env_rows, *batch.edge_mask.shape[2:]),
        edge_src_ids=batch.edge_src_ids.reshape(env_rows, batch.edge_src_ids.shape[-1]),
        edge_tgt_ids=batch.edge_tgt_ids.reshape(
            env_rows, *batch.edge_tgt_ids.shape[2:]
        ),
        global_features=batch.global_features.reshape(
            env_rows, batch.global_features.shape[-1]
        ),
        theta_ref=batch.theta_ref.reshape(env_rows),
    )


def _aggregate_ppo_metrics(
    metrics_by_minibatch: dict[str, jax.Array],
    minibatch_count: int,
) -> dict[str, jax.Array]:
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
        name: jnp.where(metric_weights > 0, values, 0.0).sum() / metric_denominator
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
                jnp.where(sample_counts > 0, metrics_by_minibatch[name], 0.0)
                * sample_counts
            ).sum() / sample_denominator
    metrics["minibatches"] = jnp.array(minibatch_count, dtype=jnp.float32)
    return metrics


def _value_loss_per_step(
    cfg: TrainConfig,
    value: jax.Array,
    value_logits: jax.Array | None,
    returns: jax.Array,
) -> jax.Array:
    """Return per-step critic loss with shape matching ``returns``."""

    if is_distributional_value_head(cfg):
        if value_logits is None:
            raise ValueError(
                "Distributional value head requires value_logits in policy output."
            )
        support = value_support(cfg.model.value_bins, cfg.model.value_max)
        batch_size, sequence_k = returns.shape
        bins = value_logits.shape[-1]
        expanded_logits = jnp.broadcast_to(
            value_logits[:, None, :],
            (batch_size, sequence_k, bins),
        ).reshape(-1, bins)
        flat_ce = categorical_value_cross_entropy(
            expanded_logits,
            returns.reshape(-1),
            support,
        )
        return flat_ce.reshape(batch_size, sequence_k)
    return 0.5 * (returns - value[:, None]) ** 2


def _ppo_update_joint_flat_jax(
    train_state: JaxTrainState,
    policy: object,
    batch: JaxTransitionBatch,
    cfg: TrainConfig,
) -> tuple[JaxTrainState, dict[str, jax.Array]]:
    sequence_k = batch.target_index.shape[-1]
    edge_count = edge_action_count(cfg.task)
    env_rows = batch.target_index.shape[0] * batch.target_index.shape[1]
    mask = jnp.ones((env_rows, sequence_k), dtype=jnp.float32)
    turn_batch = _flatten_transition_to_turn_batch(batch)
    player_count = batch.player_count.reshape(env_rows)
    ship_bucket_mask = batch.ship_bucket_mask.reshape(
        env_rows, sequence_k, edge_count, cfg.task.ship_bucket_count
    )
    target = batch.target_index.reshape(env_rows, sequence_k)
    bucket = batch.ship_bucket.reshape(env_rows, sequence_k)
    old_log_prob = batch.log_prob.reshape(env_rows, sequence_k)
    returns = batch.returns.reshape(env_rows, sequence_k)
    advantages = batch.advantages.reshape(env_rows, sequence_k)
    advantage_mean = masked_mean(advantages, mask)
    advantages = (advantages - advantage_mean) / jnp.sqrt(
        masked_mean((advantages - advantage_mean) ** 2, mask) + 1e-8
    )
    continuous = is_continuous_ship_mode(cfg)
    ship_fraction = None
    if continuous and batch.ship_fraction is not None:
        ship_fraction = batch.ship_fraction.reshape(env_rows, sequence_k)
    decoder_hidden = None
    if cfg.model.decoder_carry and batch.decoder_hidden is not None:
        decoder_hidden = batch.decoder_hidden.reshape(env_rows, cfg.model.hidden_size)

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

    edge_action_mask = jnp.concatenate(
        [
            turn_batch.edge_mask.reshape(env_rows, edge_count - 1),
            jnp.ones((env_rows, 1), dtype=bool),
        ],
        axis=1,
    )
    minibatches = {
        "mask": _reshape_minibatches(mask, minibatch_count, minibatch_size, 0.0),
        "planet_features": _reshape_minibatches(
            turn_batch.planet_features, minibatch_count, minibatch_size, 0.0
        ),
        "planet_mask": _reshape_minibatches(
            turn_batch.planet_mask, minibatch_count, minibatch_size, False
        ),
        "edge_features": _reshape_minibatches(
            turn_batch.edge_features, minibatch_count, minibatch_size, 0.0
        ),
        "edge_mask": _reshape_minibatches(
            turn_batch.edge_mask, minibatch_count, minibatch_size, False
        ),
        "edge_src_ids": _reshape_minibatches(
            turn_batch.edge_src_ids, minibatch_count, minibatch_size, 0
        ),
        "edge_tgt_ids": _reshape_minibatches(
            turn_batch.edge_tgt_ids, minibatch_count, minibatch_size, 0
        ),
        "global_features": _reshape_minibatches(
            turn_batch.global_features, minibatch_count, minibatch_size, 0.0
        ),
        "theta_ref": _reshape_minibatches(
            turn_batch.theta_ref, minibatch_count, minibatch_size, 0.0
        ),
        "edge_action_mask": _reshape_minibatches(
            edge_action_mask, minibatch_count, minibatch_size, False
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
    if continuous and ship_fraction is not None:
        minibatches["ship_fraction"] = _reshape_minibatches(
            ship_fraction, minibatch_count, minibatch_size, 0.0
        )
    if cfg.model.decoder_carry and decoder_hidden is not None:
        minibatches["decoder_hidden"] = _reshape_minibatches(
            decoder_hidden, minibatch_count, minibatch_size, 0.0
        )

    def update_minibatch(carry, minibatch):
        params, opt_state = carry
        mb_batch = TurnBatch(
            planet_features=minibatch["planet_features"],
            planet_mask=minibatch["planet_mask"],
            edge_features=minibatch["edge_features"],
            edge_mask=minibatch["edge_mask"],
            edge_src_ids=minibatch["edge_src_ids"],
            edge_tgt_ids=minibatch["edge_tgt_ids"],
            global_features=minibatch["global_features"],
            theta_ref=minibatch["theta_ref"],
        )

        def loss_fn(params):
            apply_kwargs = {
                "player_count": minibatch["player_count"],
                "target_sequence": minibatch["target"],
            }
            if cfg.model.decoder_carry and "decoder_hidden" in minibatch:
                apply_kwargs["decoder_hidden"] = minibatch["decoder_hidden"]
            output = policy.apply(params, mb_batch, **apply_kwargs)
            output = mask_policy_output_for_shield_v2(
                output,
                minibatch["edge_action_mask"],
                cfg.task.ship_bucket_count,
                minibatch["ship_bucket_mask"],
            )
            fraction_arg = (
                minibatch["ship_fraction"]
                if continuous and "ship_fraction" in minibatch
                else None
            )
            new_log_prob, entropy = action_log_prob_and_entropy(
                output,
                minibatch["target"],
                minibatch["bucket"],
                ship_fraction=fraction_arg,
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
            value_error = _value_loss_per_step(
                cfg,
                output.value,
                output.value_logits,
                minibatch["returns"],
            )
            policy_loss = -masked_mean(policy_objective, minibatch["mask"])
            value_loss = masked_mean(value_error, minibatch["mask"])
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
            for format_player_count in (2, 4):
                suffix = f"{format_player_count}p"
                format_mask = minibatch["mask"] * (
                    minibatch["player_count"][:, None] == format_player_count
                ).astype(jnp.float32)
                format_policy_loss = -masked_mean(policy_objective, format_mask)
                format_value_loss = masked_mean(value_error, format_mask)
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

        (_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = train_state.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics = dict(metrics)
        metrics["total_loss"] = _loss
        return (params, opt_state), metrics

    (params, opt_state), metrics_by_minibatch = jax.lax.scan(
        update_minibatch, (train_state.params, train_state.opt_state), minibatches
    )
    metrics = _aggregate_ppo_metrics(metrics_by_minibatch, minibatch_count)
    return (
        JaxTrainState(
            params=params, opt_state=opt_state, optimizer=train_state.optimizer
        ),
        metrics,
    )


def _ppo_update_factorized_jax(
    train_state: JaxTrainState,
    policy: object,
    batch: JaxTransitionBatch,
    cfg: TrainConfig,
) -> tuple[JaxTrainState, dict[str, jax.Array]]:
    from src.features.registry import edge_k
    from src.game.constants import MAX_PLANETS

    sequence_k = batch.source_index.shape[-1]
    k = edge_k(cfg.task)
    env_rows = batch.source_index.shape[0] * batch.source_index.shape[1]
    mask = batch.step_mask.reshape(env_rows, sequence_k)
    turn_batch = _flatten_transition_to_turn_batch(batch)
    player_count = batch.player_count.reshape(env_rows)
    ship_bucket_mask = batch.ship_bucket_mask.reshape(
        env_rows, sequence_k, MAX_PLANETS, k, cfg.task.ship_bucket_count
    )
    source = batch.source_index.reshape(env_rows, sequence_k)
    target_slot = batch.target_slot.reshape(env_rows, sequence_k)
    bucket = batch.ship_bucket.reshape(env_rows, sequence_k)
    stop_flag = batch.stop_flag.reshape(env_rows, sequence_k)
    old_log_prob = batch.log_prob.reshape(env_rows, sequence_k)
    returns = batch.returns.reshape(env_rows, sequence_k)
    advantages = batch.advantages.reshape(env_rows, sequence_k)
    advantage_mean = masked_mean(advantages, mask)
    advantages = (advantages - advantage_mean) / jnp.sqrt(
        masked_mean((advantages - advantage_mean) ** 2, mask) + 1e-8
    )
    continuous = is_continuous_ship_mode(cfg)
    ship_fraction = None
    if continuous and batch.ship_fraction is not None:
        ship_fraction = batch.ship_fraction.reshape(env_rows, sequence_k)
    decoder_hidden = None
    if cfg.model.decoder_carry and batch.decoder_hidden is not None:
        decoder_hidden = batch.decoder_hidden.reshape(env_rows, cfg.model.hidden_size)

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
    pad_rows = minibatch_count * minibatch_size - total_rows

    source_mask = (
        ship_bucket_mask[..., 1:].any(axis=-1)
        & batch.edge_mask.reshape(env_rows, 1, MAX_PLANETS, k)
    ).any(axis=-1)
    minibatches = {
        "mask": _reshape_minibatches(mask, minibatch_count, minibatch_size, 0.0),
        "planet_features": _reshape_minibatches(
            turn_batch.planet_features, minibatch_count, minibatch_size, 0.0
        ),
        "planet_mask": _reshape_minibatches(
            turn_batch.planet_mask, minibatch_count, minibatch_size, False
        ),
        "edge_features": _reshape_minibatches(
            turn_batch.edge_features, minibatch_count, minibatch_size, 0.0
        ),
        "edge_mask": _reshape_minibatches(
            turn_batch.edge_mask, minibatch_count, minibatch_size, False
        ),
        "edge_src_ids": _reshape_minibatches(
            turn_batch.edge_src_ids, minibatch_count, minibatch_size, 0
        ),
        "edge_tgt_ids": _reshape_minibatches(
            turn_batch.edge_tgt_ids, minibatch_count, minibatch_size, 0
        ),
        "global_features": _reshape_minibatches(
            turn_batch.global_features, minibatch_count, minibatch_size, 0.0
        ),
        "theta_ref": _reshape_minibatches(
            turn_batch.theta_ref, minibatch_count, minibatch_size, 0.0
        ),
        "player_count": _reshape_minibatches(
            player_count, minibatch_count, minibatch_size, 0
        ),
        "ship_bucket_mask": _reshape_minibatches(
            ship_bucket_mask, minibatch_count, minibatch_size, False
        ),
        "source": _reshape_minibatches(source, minibatch_count, minibatch_size, 0),
        "target_slot": _reshape_minibatches(
            target_slot, minibatch_count, minibatch_size, 0
        ),
        "bucket": _reshape_minibatches(bucket, minibatch_count, minibatch_size, 0),
        "stop_flag": _reshape_minibatches(
            stop_flag, minibatch_count, minibatch_size, 0
        ),
        "old_log_prob": _reshape_minibatches(
            old_log_prob, minibatch_count, minibatch_size, 0.0
        ),
        "returns": _reshape_minibatches(returns, minibatch_count, minibatch_size, 0.0),
        "advantages": _reshape_minibatches(
            advantages, minibatch_count, minibatch_size, 0.0
        ),
    }
    if continuous and ship_fraction is not None:
        minibatches["ship_fraction"] = _reshape_minibatches(
            ship_fraction, minibatch_count, minibatch_size, 0.0
        )
    if cfg.model.decoder_carry and decoder_hidden is not None:
        minibatches["decoder_hidden"] = _reshape_minibatches(
            decoder_hidden, minibatch_count, minibatch_size, 0.0
        )

    def update_minibatch(carry, minibatch):
        params, opt_state = carry
        mb_batch = TurnBatch(
            planet_features=minibatch["planet_features"],
            planet_mask=minibatch["planet_mask"],
            edge_features=minibatch["edge_features"],
            edge_mask=minibatch["edge_mask"],
            edge_src_ids=minibatch["edge_src_ids"],
            edge_tgt_ids=minibatch["edge_tgt_ids"],
            global_features=minibatch["global_features"],
            theta_ref=minibatch["theta_ref"],
        )

        def loss_fn(params):
            apply_kwargs = {
                "player_count": minibatch["player_count"],
                "source_sequence": minibatch["source"],
                "target_slot_sequence": minibatch["target_slot"],
            }
            if cfg.model.decoder_carry and "decoder_hidden" in minibatch:
                apply_kwargs["decoder_hidden"] = minibatch["decoder_hidden"]
            output = policy.apply(params, mb_batch, **apply_kwargs)
            has_real_bucket = minibatch["ship_bucket_mask"][..., 1:].any(axis=-1)
            source_mask = (mb_batch.edge_mask[:, None, :, :] & has_real_bucket).any(
                axis=-1
            )
            fraction_arg = (
                minibatch["ship_fraction"]
                if continuous and "ship_fraction" in minibatch
                else None
            )
            new_log_prob, entropy, stop_entropy, move_entropy = (
                factored_action_log_prob_with_shield(
                    output,
                    minibatch["source"],
                    minibatch["target_slot"],
                    minibatch["bucket"],
                    minibatch["stop_flag"],
                    minibatch["mask"],
                    source_mask,
                    minibatch["ship_bucket_mask"],
                    ship_fraction=fraction_arg,
                )
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
            value_error = _value_loss_per_step(
                cfg,
                output.value,
                output.value_logits,
                minibatch["returns"],
            )
            policy_loss = -masked_mean(policy_objective, minibatch["mask"])
            value_loss = masked_mean(value_error, minibatch["mask"])
            entropy_loss = masked_mean(entropy, minibatch["mask"])
            entropy_stop_loss = masked_mean(stop_entropy, minibatch["mask"])
            entropy_move_loss = masked_mean(move_entropy, minibatch["mask"])
            loss = (
                policy_loss
                + cfg.training.vf_coef * value_loss
                - cfg.training.ent_coef * entropy_loss
            )
            metrics = {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy_loss,
                "entropy_stop": entropy_stop_loss,
                "entropy_move": entropy_move_loss,
                "approx_kl": approx_kl,
                "loss": loss,
                "sample_count": minibatch["mask"].sum(),
            }
            for format_player_count in (2, 4):
                suffix = f"{format_player_count}p"
                format_mask = minibatch["mask"] * (
                    minibatch["player_count"][:, None] == format_player_count
                ).astype(jnp.float32)
                format_policy_loss = -masked_mean(policy_objective, format_mask)
                format_value_loss = masked_mean(value_error, format_mask)
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

        (_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = train_state.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics = dict(metrics)
        metrics["total_loss"] = _loss
        return (params, opt_state), metrics

    (params, opt_state), metrics_by_minibatch = jax.lax.scan(
        update_minibatch, (train_state.params, train_state.opt_state), minibatches
    )
    metrics = _aggregate_ppo_metrics(metrics_by_minibatch, minibatch_count)
    return (
        JaxTrainState(
            params=params, opt_state=opt_state, optimizer=train_state.optimizer
        ),
        metrics,
    )


def ppo_update_jax(
    train_state: JaxTrainState,
    policy: object,
    batch: JaxTransitionBatch,
    cfg: TrainConfig,
) -> tuple[JaxTrainState, dict[str, jax.Array]]:
    if is_factorized_pointer_decoder(cfg.model):
        return _ppo_update_factorized_jax(train_state, policy, batch, cfg)
    return _ppo_update_joint_flat_jax(train_state, policy, batch, cfg)
