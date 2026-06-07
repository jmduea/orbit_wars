from __future__ import annotations

import jax.numpy as jnp
import optax

import jax
from src.artifacts.checkpoint_compat import (
    is_factorized_pointer_decoder,
    is_planet_flow_pointer_decoder,
)
from src.config import TrainConfig
from src.jax.action_codec import (
    PlanetFlowPolicyOutput,
    planet_flow_action_log_prob_entropy,
    planet_flow_invalid_bucket_count,
)
from src.jax.array_ops import masked_mean
from src.jax.distributional_value import (
    sparse_categorical_value_cross_entropy,
    value_support,
)
from src.jax.factored_sequence_scan import (
    factored_logprob_parity_metrics,
    replay_factored_sequence_logprob,
    rollout_replay_parity_summary,
)
from src.jax.features import TurnBatch
from src.jax.policy import is_distributional_value_head
from src.jax.rollout.types import (
    FactorizedActionReplay,
    JaxTrainState,
    JaxTransitionBatch,
    PlanetFlowActionReplay,
    require_factorized_replay,
    require_planet_flow_replay,
    transition_env_rows,
)
from src.jax.ship_action import is_continuous_ship_mode
from src.telemetry.metric_registry import (
    prune_scalar_metrics,
    required_ppo_metric_names,
)

# Cap importance-ratio magnitude before exp; exp(20) allows ~485M and explodes
# the policy surrogate on negative-advantage samples when replay diverges.
_LOG_RATIO_CLIP = 10.0


def _clipped_policy_objective(
    advantages: jax.Array,
    ratio: jax.Array,
    clipped_ratio: jax.Array,
) -> jax.Array:
    """PPO clipped surrogate with sign-aware branch selection.

    ``jnp.minimum`` alone fails when ``advantages < 0`` and ``ratio > 1 + eps``:
    the unclipped term is more negative and wins, so loss grows without bound
    as replay/importance ratios drift from rollout log-probs.
    """

    unclipped = advantages * ratio
    clipped = advantages * clipped_ratio
    return jnp.where(
        advantages >= 0.0,
        jnp.minimum(unclipped, clipped),
        jnp.maximum(unclipped, clipped),
    )


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


def _flatten_state_scalars(value: jax.Array, env_rows: int) -> jax.Array:
    """Collapse legacy per-sequence broadcast returns/advantages to per-state scalars."""

    flat = value.reshape(-1)
    if value.ndim >= 3:
        return value.reshape(env_rows, -1)[:, 0]
    if value.ndim == 2 and value.shape[-1] > 1:
        return value.reshape(env_rows, -1)[:, 0]
    return flat[:env_rows]


def _actor_advantages_from_state(
    advantages: jax.Array, sequence_k: int
) -> jax.Array:
    """Broadcast per-state advantages for per-sub-action PPO objectives."""

    if advantages.ndim == 1:
        return jnp.broadcast_to(advantages[:, None], (advantages.shape[0], sequence_k))
    return advantages


def _flatten_transition_to_turn_batch(batch: JaxTransitionBatch) -> TurnBatch:
    env_rows = transition_env_rows(batch)
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
            "approx_kl_v2",
            "total_loss",
        )
    )
    format_sample_names = frozenset(
        f"loss_sample_count_{suffix}" for suffix in ("2p", "4p")
    )
    excluded_from_mean = frozenset(
        {
            "sample_count",
            "log_ratio_abs_max",
            *format_metric_names,
            *format_sample_names,
        }
    )
    metric_weights = jnp.where(metrics_by_minibatch["sample_count"] > 0.0, 1.0, 0.0)
    metric_denominator = jnp.maximum(metric_weights.sum(), 1.0)
    metrics = {
        name: jnp.where(metric_weights > 0, values, 0.0).sum() / metric_denominator
        for name, values in metrics_by_minibatch.items()
        if name not in excluded_from_mean
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
            "approx_kl_v2",
            "total_loss",
        ):
            name = f"{metric_name}_{suffix}"
            if name not in metrics_by_minibatch:
                metrics[name] = jnp.array(0.0, dtype=jnp.float32)
                continue
            metrics[name] = (
                jnp.where(sample_counts > 0, metrics_by_minibatch[name], 0.0)
                * sample_counts
            ).sum() / sample_denominator
    sample_count = metrics_by_minibatch["sample_count"]
    minibatch_axis = jnp.arange(sample_count.shape[0], dtype=jnp.int32)
    active = sample_count > 0
    last_active = jnp.maximum(
        jnp.max(jnp.where(active, minibatch_axis, jnp.int32(-1))),
        jnp.int32(0),
    )
    for name in (
        "approx_kl",
        "approx_kl_v2",
        "log_ratio_abs_mean",
        "importance_ratio_mean",
        "clip_fraction",
    ):
        if name in metrics_by_minibatch:
            values = metrics_by_minibatch[name]
            metrics[f"{name}_first_minibatch"] = values[0]
            metrics[f"{name}_last_minibatch"] = values[last_active]
    if "log_ratio_abs_max" in metrics_by_minibatch:
        metrics["log_ratio_abs_max_last_minibatch"] = metrics_by_minibatch[
            "log_ratio_abs_max"
        ][last_active]
    metrics["minibatches"] = jnp.array(minibatch_count, dtype=jnp.float32)
    return metrics


def _check_planet_flow_bucket_count(invalid_bucket_count: jax.Array) -> None:
    def raise_if_invalid(count) -> None:
        count_int = int(count)
        if count_int:
            raise ValueError(
                "Planet Flow PPO received out-of-range pressure bucket ids in active "
                f"targets: {count_int} invalid entries."
            )

    jax.debug.callback(raise_if_invalid, invalid_bucket_count)


def _value_loss_per_state(
    cfg: TrainConfig,
    value: jax.Array,
    value_logits: jax.Array | None,
    returns: jax.Array,
) -> jax.Array:
    """Return per-state critic loss with shape ``(batch,)``."""

    if returns.ndim > 1:
        returns = returns.reshape(returns.shape[0], -1)[:, 0]
    if is_distributional_value_head(cfg):
        if value_logits is None:
            raise ValueError(
                "Distributional value head requires value_logits in policy output."
            )
        support = value_support(cfg.model.value_bins, cfg.model.value_max)
        return sparse_categorical_value_cross_entropy(
            value_logits, returns, support
        )
    return 0.5 * (returns - value) ** 2


def _ppo_importance_sampling(
    old_log_prob: jax.Array,
    new_log_prob: jax.Array,
    mask: jax.Array,
    *,
    clip_coef: float,
) -> dict[str, jax.Array]:
    """Shared PPO ratio, KL, and clip diagnostics for factorized and Planet Flow."""
    log_ratio_raw = new_log_prob - old_log_prob
    approx_kl = masked_mean(old_log_prob - new_log_prob, mask)
    ratio_for_kl = jnp.exp(jnp.clip(log_ratio_raw, -_LOG_RATIO_CLIP, _LOG_RATIO_CLIP))
    approx_kl_v2 = masked_mean((ratio_for_kl - 1.0) - log_ratio_raw, mask)
    log_ratio = jnp.clip(log_ratio_raw, -_LOG_RATIO_CLIP, _LOG_RATIO_CLIP)
    ratio = jnp.exp(log_ratio)
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    return {
        "log_ratio_raw": log_ratio_raw,
        "approx_kl": approx_kl,
        "ratio_for_kl": ratio_for_kl,
        "approx_kl_v2": approx_kl_v2,
        "ratio": ratio,
        "clipped_ratio": clipped_ratio,
        "log_ratio_abs_mean": masked_mean(jnp.abs(log_ratio_raw), mask),
        "log_ratio_abs_max": jnp.max(
            jnp.where(mask > 0.0, jnp.abs(log_ratio_raw), 0.0)
        ),
        "importance_ratio_mean": masked_mean(ratio, mask),
        "clip_fraction": masked_mean(
            (jnp.abs(ratio - 1.0) > clip_coef).astype(jnp.float32),
            mask,
        ),
    }


def _ppo_surrogate_losses(
    cfg: TrainConfig,
    *,
    policy_objective: jax.Array,
    value_error: jax.Array,
    entropy: jax.Array,
    mask: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    policy_loss = -masked_mean(policy_objective, mask)
    value_loss = masked_mean(value_error, mask)
    entropy_loss = masked_mean(entropy, mask)
    loss = (
        policy_loss
        + cfg.training.vf_coef * value_loss
        - cfg.training.ent_coef * entropy_loss
    )
    return loss, {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_loss,
    }


def _finalize_ppo_epoch_scan(
    train_state: JaxTrainState,
    minibatches: dict[str, jax.Array],
    update_minibatch,
    cfg: TrainConfig,
    minibatch_count: int,
    *,
    extra_metrics: dict[str, jax.Array] | None = None,
) -> tuple[JaxTrainState, dict[str, jax.Array]]:
    (params, opt_state), metrics_by_minibatch = jax.lax.scan(
        update_minibatch, (train_state.params, train_state.opt_state), minibatches
    )
    metrics = _aggregate_ppo_metrics(metrics_by_minibatch, minibatch_count)
    if extra_metrics:
        metrics.update(extra_metrics)
    allowed = frozenset(required_ppo_metric_names(cfg, tuple(metrics.keys())))
    metrics = prune_scalar_metrics(metrics, allowed)
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
    replay = require_factorized_replay(batch)

    from src.features.registry import edge_k
    from src.game.constants import MAX_PLANETS

    sequence_k = replay.source_index.shape[-1]
    k = edge_k(cfg.task)
    env_rows = transition_env_rows(batch)
    mask = replay.step_mask.reshape(env_rows, sequence_k)
    turn_batch = _flatten_transition_to_turn_batch(batch)
    player_count = batch.player_count.reshape(env_rows)
    ship_bucket_mask = replay.ship_bucket_mask.reshape(
        env_rows, sequence_k, MAX_PLANETS, k, cfg.task.ship_bucket_count
    )
    source = replay.source_index.reshape(env_rows, sequence_k)
    target_slot = replay.target_slot.reshape(env_rows, sequence_k)
    bucket = replay.ship_bucket.reshape(env_rows, sequence_k)
    stop_flag = replay.stop_flag.reshape(env_rows, sequence_k)
    old_log_prob = replay.log_prob.reshape(env_rows, sequence_k)
    returns_state = _flatten_state_scalars(batch.returns, env_rows)
    advantages_state = _flatten_state_scalars(batch.advantages, env_rows)
    advantages_actor = _actor_advantages_from_state(advantages_state, sequence_k)
    advantage_mean = jnp.mean(advantages_state)
    advantages_state = (advantages_state - advantage_mean) / jnp.sqrt(
        jnp.mean((advantages_state - advantage_mean) ** 2) + 1e-8
    )
    advantages_actor = _actor_advantages_from_state(advantages_state, sequence_k)
    continuous = is_continuous_ship_mode(cfg)
    ship_fraction = None
    if continuous and replay.ship_fraction is not None:
        ship_fraction = replay.ship_fraction.reshape(env_rows, sequence_k)
    decoder_hidden = None
    if cfg.model.decoder_carry and replay.decoder_hidden is not None:
        decoder_hidden = replay.decoder_hidden.reshape(env_rows, cfg.model.hidden_size)
    initial_planet_ships = None
    if batch.initial_planet_ships is not None:
        initial_planet_ships = batch.initial_planet_ships.reshape(
            env_rows, batch.initial_planet_ships.shape[-1]
        )

    total_rows = mask.shape[0]
    minibatch_size = min(max(int(cfg.training.update_chunk_rows), 1), total_rows)
    minibatch_count = (total_rows + minibatch_size - 1) // minibatch_size
    pad_rows = minibatch_count * minibatch_size - total_rows

    source_mask = ship_bucket_mask[..., 1:].any(axis=(-2, -1))
    launch_active = mask * (1.0 - stop_flag)
    per_step_bucket_counts = ship_bucket_mask.sum(axis=(-3, -2, -1))
    debug_group_enabled = bool(
        getattr(
            getattr(getattr(cfg, "telemetry", None), "metric_groups", None),
            "debug",
            False,
        )
    ) or bool(cfg.training.debug_replay_parity)
    debug_metrics: dict[str, jax.Array] = {}
    if debug_group_enabled:
        debug_metrics = {
        "debug_step_mask_sum": mask.sum(),
        "debug_old_log_prob_finite": jnp.all(jnp.isfinite(old_log_prob)).astype(
            jnp.float32
        ),
        "debug_returns_finite": jnp.all(jnp.isfinite(returns_state)).astype(
            jnp.float32
        ),
        "debug_advantages_finite": jnp.all(jnp.isfinite(advantages_state)).astype(
            jnp.float32
        ),
        "debug_ship_bucket_mask_any_min": per_step_bucket_counts.min(),
        "debug_ship_bucket_mask_all_false": (per_step_bucket_counts == 0)
        .sum()
        .astype(jnp.float32),
        "debug_source_mask_all_false": (source_mask.sum(axis=-1) == 0)
        .sum()
        .astype(jnp.float32),
        "debug_active_launch_all_false_bucket": (
            launch_active * (per_step_bucket_counts == 0)
        )
        .sum()
        .astype(jnp.float32),
    }
    parity_batch = TurnBatch(
        planet_features=turn_batch.planet_features,
        planet_mask=turn_batch.planet_mask,
        edge_features=turn_batch.edge_features,
        edge_mask=turn_batch.edge_mask,
        edge_src_ids=turn_batch.edge_src_ids,
        edge_tgt_ids=turn_batch.edge_tgt_ids,
        global_features=turn_batch.global_features,
        theta_ref=turn_batch.theta_ref,
    )
    if cfg.training.debug_replay_parity:
        parity_fraction = ship_fraction
        parity_hidden = decoder_hidden
        debug_metrics.update(
            factored_logprob_parity_metrics(
                train_state.params,
                policy,
                parity_batch,
                cfg,
                player_count=player_count,
                source_index=source,
                target_slot=target_slot,
                ship_bucket=bucket,
                stop_flag=stop_flag,
                step_mask=mask,
                ship_bucket_mask=ship_bucket_mask,
                old_log_prob=old_log_prob,
                ship_fraction=parity_fraction,
                decoder_hidden=parity_hidden,
                initial_remaining_ships=initial_planet_ships,
                advantages=advantages_actor,
            )
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
        "returns": _reshape_minibatches(
            returns_state, minibatch_count, minibatch_size, 0.0
        ),
        "advantages": _reshape_minibatches(
            advantages_actor, minibatch_count, minibatch_size, 0.0
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
    if initial_planet_ships is not None:
        minibatches["initial_planet_ships"] = _reshape_minibatches(
            initial_planet_ships, minibatch_count, minibatch_size, 0.0
        )

    first_mb_end = min(minibatch_size, total_rows)
    parity_batch = TurnBatch(
        planet_features=turn_batch.planet_features[:first_mb_end],
        planet_mask=turn_batch.planet_mask[:first_mb_end],
        edge_features=turn_batch.edge_features[:first_mb_end],
        edge_mask=turn_batch.edge_mask[:first_mb_end],
        edge_src_ids=turn_batch.edge_src_ids[:first_mb_end],
        edge_tgt_ids=turn_batch.edge_tgt_ids[:first_mb_end],
        global_features=turn_batch.global_features[:first_mb_end],
        theta_ref=turn_batch.theta_ref[:first_mb_end],
    )
    parity_metrics: dict[str, jax.Array] = {}
    if debug_group_enabled:
        parity_metrics = rollout_replay_parity_summary(
            train_state.params,
            policy,
            parity_batch,
            cfg,
            player_count=player_count[:first_mb_end],
            source_index=source[:first_mb_end],
            target_slot=target_slot[:first_mb_end],
            ship_bucket=bucket[:first_mb_end],
            stop_flag=stop_flag[:first_mb_end],
            step_mask=mask[:first_mb_end],
            ship_bucket_mask=ship_bucket_mask[:first_mb_end],
            old_log_prob=old_log_prob[:first_mb_end],
            ship_fraction=(
                ship_fraction[:first_mb_end] if ship_fraction is not None else None
            ),
            decoder_hidden=(
                decoder_hidden[:first_mb_end] if decoder_hidden is not None else None
            ),
            initial_remaining_ships=(
                initial_planet_ships[:first_mb_end]
                if initial_planet_ships is not None
                else None
            ),
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
            fraction_arg = (
                minibatch["ship_fraction"]
                if continuous and "ship_fraction" in minibatch
                else None
            )
            decoder_hidden_arg = (
                minibatch["decoder_hidden"]
                if cfg.model.decoder_carry and "decoder_hidden" in minibatch
                else None
            )
            initial_ships_arg = (
                minibatch["initial_planet_ships"]
                if "initial_planet_ships" in minibatch
                else None
            )
            replay = replay_factored_sequence_logprob(
                params,
                policy,
                mb_batch,
                cfg,
                player_count=minibatch["player_count"],
                source_index=minibatch["source"],
                target_slot=minibatch["target_slot"],
                ship_bucket=minibatch["bucket"],
                stop_flag=minibatch["stop_flag"],
                step_mask=minibatch["mask"],
                ship_bucket_mask=minibatch["ship_bucket_mask"],
                ship_fraction=fraction_arg,
                decoder_hidden=decoder_hidden_arg,
                initial_remaining_ships=initial_ships_arg,
            )
            new_log_prob = replay.log_prob
            entropy = replay.entropy
            stop_entropy = replay.stop_entropy
            move_entropy = replay.move_entropy
            sampling = _ppo_importance_sampling(
                minibatch["old_log_prob"],
                new_log_prob,
                minibatch["mask"],
                clip_coef=cfg.training.clip_coef,
            )
            log_ratio_raw = sampling["log_ratio_raw"]
            approx_kl = sampling["approx_kl"]
            ratio_for_kl = sampling["ratio_for_kl"]
            approx_kl_v2 = sampling["approx_kl_v2"]
            ratio = sampling["ratio"]
            clipped_ratio = sampling["clipped_ratio"]
            log_ratio_abs_mean = sampling["log_ratio_abs_mean"]
            log_ratio_abs_max = sampling["log_ratio_abs_max"]
            importance_ratio_mean = sampling["importance_ratio_mean"]
            clip_fraction = sampling["clip_fraction"]
            policy_objective = _clipped_policy_objective(
                minibatch["advantages"],
                ratio,
                clipped_ratio,
            )
            value = replay.value
            value_logits = replay.value_logits
            if value is None:
                raise ValueError("Factorized replay must return critic outputs.")
            value_error = _value_loss_per_state(
                cfg,
                value,
                value_logits,
                minibatch["returns"],
            )
            policy_loss = -masked_mean(policy_objective, minibatch["mask"])
            value_state_mask = (minibatch["mask"].sum(axis=-1) > 0.0).astype(
                jnp.float32
            )
            value_loss = masked_mean(value_error, value_state_mask)
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
                "approx_kl_v2": approx_kl_v2,
                "log_ratio_abs_mean": log_ratio_abs_mean,
                "log_ratio_abs_max": log_ratio_abs_max,
                "importance_ratio_mean": importance_ratio_mean,
                "clip_fraction": clip_fraction,
                "loss": loss,
                "sample_count": minibatch["mask"].sum(),
            }
            for format_player_count in (2, 4):
                suffix = f"{format_player_count}p"
                format_mask = minibatch["mask"] * (
                    minibatch["player_count"][:, None] == format_player_count
                ).astype(jnp.float32)
                state_mask = (
                    minibatch["player_count"] == format_player_count
                ).astype(jnp.float32)
                format_policy_loss = -masked_mean(policy_objective, format_mask)
                format_value_loss = masked_mean(value_error, state_mask)
                format_entropy = masked_mean(entropy, format_mask)
                format_approx_kl = masked_mean(
                    minibatch["old_log_prob"] - new_log_prob,
                    format_mask,
                )
                format_approx_kl_v2 = masked_mean(
                    (ratio_for_kl - 1.0) - log_ratio_raw,
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
                metrics[f"approx_kl_v2_{suffix}"] = format_approx_kl_v2
                metrics[f"total_loss_{suffix}"] = format_total_loss
                metrics[f"loss_sample_count_{suffix}"] = format_mask.sum()
            return loss, metrics

        (_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = train_state.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics = dict(metrics)
        metrics["total_loss"] = _loss
        return (params, opt_state), metrics

    extra_metrics = dict(parity_metrics)
    extra_metrics.update(debug_metrics)
    return _finalize_ppo_epoch_scan(
        train_state,
        minibatches,
        update_minibatch,
        cfg,
        minibatch_count,
        extra_metrics=extra_metrics,
    )


def _ppo_update_planet_flow_jax(
    train_state: JaxTrainState,
    policy: object,
    batch: JaxTransitionBatch,
    cfg: TrainConfig,
) -> tuple[JaxTrainState, dict[str, jax.Array]]:
    replay = require_planet_flow_replay(batch)

    invalid_bucket_count = planet_flow_invalid_bucket_count(
        replay.target_bucket,
        len(cfg.model.planet_flow.pressure_bucket_values),
        replay.target_mask,
    )
    _check_planet_flow_bucket_count(invalid_bucket_count)

    env_rows = transition_env_rows(batch)
    turn_batch = _flatten_transition_to_turn_batch(batch)
    player_count = batch.player_count.reshape(env_rows)
    target_bucket = replay.target_bucket.reshape(env_rows, -1)
    target_mask = replay.target_mask.reshape(env_rows, -1)
    old_log_prob = replay.log_prob.reshape(env_rows)
    returns_state = _flatten_state_scalars(batch.returns, env_rows)
    advantages_state = _flatten_state_scalars(batch.advantages, env_rows)
    advantage_mean = jnp.mean(advantages_state)
    advantages_state = (advantages_state - advantage_mean) / jnp.sqrt(
        jnp.mean((advantages_state - advantage_mean) ** 2) + 1e-8
    )

    total_rows = target_bucket.shape[0]
    minibatch_size = min(max(int(cfg.training.update_chunk_rows), 1), total_rows)
    minibatch_count = (total_rows + minibatch_size - 1) // minibatch_size
    pad_rows = minibatch_count * minibatch_size - total_rows
    row_mask = jnp.pad(
        jnp.ones((total_rows,), dtype=jnp.float32),
        (0, pad_rows),
        constant_values=0.0,
    ).reshape(minibatch_count, minibatch_size)

    minibatches = {
        "row_mask": row_mask,
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
        "target_bucket": _reshape_minibatches(
            target_bucket, minibatch_count, minibatch_size, 0
        ),
        "target_mask": _reshape_minibatches(
            target_mask, minibatch_count, minibatch_size, False
        ),
        "old_log_prob": _reshape_minibatches(
            old_log_prob, minibatch_count, minibatch_size, 0.0
        ),
        "returns": _reshape_minibatches(
            returns_state, minibatch_count, minibatch_size, 0.0
        ),
        "advantages": _reshape_minibatches(
            advantages_state, minibatch_count, minibatch_size, 0.0
        ),
    }

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
            output = policy.apply(
                params, mb_batch, player_count=minibatch["player_count"]
            )
            if not isinstance(output, PlanetFlowPolicyOutput):
                raise TypeError(
                    "planet_flow_target_heatmap policy must return "
                    "PlanetFlowPolicyOutput."
                )
            new_log_prob, entropy = planet_flow_action_log_prob_entropy(
                output,
                minibatch["target_bucket"],
                minibatch["target_mask"],
            )
            sampling = _ppo_importance_sampling(
                minibatch["old_log_prob"],
                new_log_prob,
                minibatch["row_mask"],
                clip_coef=cfg.training.clip_coef,
            )
            log_ratio_raw = sampling["log_ratio_raw"]
            approx_kl = sampling["approx_kl"]
            ratio_for_kl = sampling["ratio_for_kl"]
            approx_kl_v2 = sampling["approx_kl_v2"]
            ratio = sampling["ratio"]
            clipped_ratio = sampling["clipped_ratio"]
            policy_objective = _clipped_policy_objective(
                minibatch["advantages"],
                ratio,
                clipped_ratio,
            )
            value_error = _value_loss_per_state(
                cfg,
                output.value,
                output.value_logits,
                minibatch["returns"],
            )
            loss, core_metrics = _ppo_surrogate_losses(
                cfg,
                policy_objective=policy_objective,
                value_error=value_error,
                entropy=entropy,
                mask=minibatch["row_mask"],
            )
            metrics = {
                **core_metrics,
                "approx_kl": approx_kl,
                "approx_kl_v2": approx_kl_v2,
                "log_ratio_abs_mean": sampling["log_ratio_abs_mean"],
                "log_ratio_abs_max": sampling["log_ratio_abs_max"],
                "importance_ratio_mean": sampling["importance_ratio_mean"],
                "clip_fraction": sampling["clip_fraction"],
                "loss": loss,
                "sample_count": minibatch["row_mask"].sum(),
            }
            for format_player_count in (2, 4):
                suffix = f"{format_player_count}p"
                format_mask = minibatch["row_mask"] * (
                    minibatch["player_count"] == format_player_count
                ).astype(jnp.float32)
                format_policy_loss = -masked_mean(policy_objective, format_mask)
                format_value_loss = masked_mean(value_error, format_mask)
                format_entropy = masked_mean(entropy, format_mask)
                format_approx_kl = masked_mean(
                    minibatch["old_log_prob"] - new_log_prob,
                    format_mask,
                )
                format_approx_kl_v2 = masked_mean(
                    (ratio_for_kl - 1.0) - log_ratio_raw,
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
                metrics[f"approx_kl_v2_{suffix}"] = format_approx_kl_v2
                metrics[f"total_loss_{suffix}"] = format_total_loss
                metrics[f"loss_sample_count_{suffix}"] = format_mask.sum()
            return loss, metrics

        (_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = train_state.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics = dict(metrics)
        metrics["total_loss"] = _loss
        return (params, opt_state), metrics

    return _finalize_ppo_epoch_scan(
        train_state,
        minibatches,
        update_minibatch,
        cfg,
        minibatch_count,
    )


def ppo_update_jax(
    train_state: JaxTrainState,
    policy: object,
    batch: JaxTransitionBatch,
    cfg: TrainConfig,
) -> tuple[JaxTrainState, dict[str, jax.Array]]:
    if isinstance(batch.action_replay, FactorizedActionReplay):
        if not is_factorized_pointer_decoder(cfg.model):
            raise ValueError(
                "Factorized action replay requires a factorized pointer_decoder config."
            )
        return _ppo_update_factorized_jax(train_state, policy, batch, cfg)
    if isinstance(batch.action_replay, PlanetFlowActionReplay):
        if not is_planet_flow_pointer_decoder(cfg.model):
            raise ValueError(
                "Planet Flow action replay requires a planet_flow pointer_decoder config."
            )
        return _ppo_update_planet_flow_jax(train_state, policy, batch, cfg)
    raise ValueError("Transition batch action_replay variant is unsupported.")
