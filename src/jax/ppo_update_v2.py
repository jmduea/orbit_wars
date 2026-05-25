from __future__ import annotations

import jax.numpy as jnp
import optax

import jax
from src.config import TrainConfig
from src.game.trajectory_shield import mask_policy_output_for_shield_v2
from src.jax.features_v2 import JaxTurnBatchV2
from src.jax.policy import action_log_prob_and_entropy
from src.jax.policy_v2 import edge_action_count
from src.jax.ppo_update import _reshape_minibatches, masked_mean
from src.jax.rollout.types import JaxTrainState, JaxTransitionBatchV2


def _flatten_transition_to_turn_batch(batch: JaxTransitionBatchV2) -> JaxTurnBatchV2:
    env_rows = batch.target_index.shape[0] * batch.target_index.shape[1]
    return JaxTurnBatchV2(
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


def ppo_update_jax_v2(
    train_state: JaxTrainState,
    policy: object,
    batch: JaxTransitionBatchV2,
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

    def update_minibatch(carry, minibatch):
        params, opt_state = carry
        mb_batch = JaxTurnBatchV2(
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
                params,
                mb_batch,
                player_count=minibatch["player_count"],
                target_sequence=minibatch["target"],
            )
            output = mask_policy_output_for_shield_v2(
                output,
                minibatch["edge_action_mask"],
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
                "total_loss": loss,
                "total_loss_2p": loss,
                "total_loss_4p": jnp.array(0.0, dtype=jnp.float32),
                "loss_sample_count_2p": minibatch["mask"].sum(),
                "loss_sample_count_4p": jnp.array(0.0, dtype=jnp.float32),
            }
            return loss, metrics

        (_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = train_state.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), metrics

    (params, opt_state), metrics_by_minibatch = jax.lax.scan(
        update_minibatch, (train_state.params, train_state.opt_state), minibatches
    )
    metric_weights = jnp.where(metrics_by_minibatch["sample_count"] > 0.0, 1.0, 0.0)
    metric_denominator = jnp.maximum(metric_weights.sum(), 1.0)
    metrics = {
        name: (values * metric_weights).sum() / metric_denominator
        for name, values in metrics_by_minibatch.items()
        if name
        not in {
            "sample_count",
            "total_loss_2p",
            "total_loss_4p",
            "loss_sample_count_4p",
        }
    }
    metrics["minibatches"] = jnp.array(minibatch_count, dtype=jnp.float32)
    metrics["total_loss_2p"] = metrics["total_loss"]
    metrics["loss_sample_count_2p"] = metrics_by_minibatch["loss_sample_count_2p"].sum()
    metrics["loss_sample_count_4p"] = jnp.array(0.0, dtype=jnp.float32)
    return (
        JaxTrainState(
            params=params, opt_state=opt_state, optimizer=train_state.optimizer
        ),
        metrics,
    )
