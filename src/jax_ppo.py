from __future__ import annotations

import flax
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from .config import TrainConfig
from .features import candidate_feature_dim, global_feature_dim, self_feature_dim
from .jax_env import JaxAction, JaxEnvState, batched_reset, batched_step
from .jax_features import JaxTurnBatch, encode_turn
from .jax_policy import JaxPlanetPolicy, action_log_prob_and_entropy, sample_actions


class JaxTransitionBatch(NamedTuple):
    self_features: jax.Array
    candidate_features: jax.Array
    global_features: jax.Array
    candidate_mask: jax.Array
    decision_mask: jax.Array
    target_index: jax.Array
    ship_bucket: jax.Array
    log_prob: jax.Array
    returns: jax.Array
    advantages: jax.Array


@flax.struct.dataclass
class JaxTrainState:
    params: dict
    opt_state: optax.OptState
    optimizer: optax.GradientTransformation = flax.struct.field(pytree_node=False)


def init_train_state(
    key: jax.Array, policy: JaxPlanetPolicy, cfg: TrainConfig
) -> JaxTrainState:
    dummy_self = jnp.zeros((1, self_feature_dim()), dtype=jnp.float32)
    dummy_candidate = jnp.zeros(
        (1, cfg.env.candidate_count, candidate_feature_dim()), dtype=jnp.float32
    )
    dummy_global = jnp.zeros((1, global_feature_dim()), dtype=jnp.float32)
    dummy_mask = jnp.ones((1, cfg.env.candidate_count), dtype=bool)
    params = policy.init(key, dummy_self, dummy_candidate, dummy_global, dummy_mask)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.ppo.max_grad_norm), optax.adam(cfg.ppo.lr)
    )
    return JaxTrainState(
        params=params, opt_state=optimizer.init(params), optimizer=optimizer
    )


def flatten_batch(
    batch: JaxTurnBatch,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    return (
        batch.self_features.reshape(-1, self_feature_dim()),
        batch.candidate_features.reshape(
            -1, batch.candidate_features.shape[-2], candidate_feature_dim()
        ),
        batch.global_features.reshape(-1, global_feature_dim()),
        batch.candidate_mask.reshape(-1, batch.candidate_mask.shape[-1]),
        batch.decision_mask.reshape(-1),
    )


def ship_count_for_bucket_jax(
    available_ships: jax.Array, bucket: jax.Array, bucket_count: int
) -> jax.Array:
    fraction = jnp.where(
        bucket <= 0, 0.0, bucket.astype(jnp.float32) / float(max(bucket_count - 1, 1))
    )
    ships = jnp.ceil(available_ships * fraction)
    ships = jnp.minimum(available_ships, jnp.maximum(1.0, ships))
    return jnp.where((available_ships <= 0.0) | (fraction <= 0.0), 0.0, ships)


def build_action_from_batch(
    batch: JaxTurnBatch,
    target_index: jax.Array,
    ship_bucket: jax.Array,
    cfg: TrainConfig,
) -> JaxAction:
    env_count = batch.self_features.shape[0]
    planet_count = batch.self_features.shape[1]
    target_index = target_index.reshape(env_count, planet_count)
    ship_bucket = ship_bucket.reshape(env_count, planet_count)
    chosen_mask = jnp.take_along_axis(
        batch.candidate_mask, target_index[..., None], axis=-1
    ).squeeze(-1)
    chosen_angle = jnp.take_along_axis(
        batch.target_angles, target_index[..., None], axis=-1
    ).squeeze(-1)
    ships = ship_count_for_bucket_jax(
        batch.source_ships, ship_bucket, cfg.env.ship_bucket_count
    )
    valid = (
        batch.decision_mask
        & chosen_mask
        & (target_index > 0)
        & (ship_bucket > 0)
        & (ships > 0.0)
    )
    pad = cfg.env.max_fleets - planet_count
    source_id = jnp.pad(batch.source_ids, ((0, 0), (0, pad)), constant_values=-1)
    angle = jnp.pad(chosen_angle, ((0, 0), (0, pad)), constant_values=0.0)
    ships = jnp.pad(ships, ((0, 0), (0, pad)), constant_values=0.0)
    valid = jnp.pad(valid, ((0, 0), (0, pad)), constant_values=False)
    return JaxAction(source_id=source_id, angle=angle, ships=ships, valid=valid)


def collect_rollout_jax(
    key: jax.Array,
    env_state: JaxEnvState,
    turn_batch: JaxTurnBatch,
    train_state: JaxTrainState,
    policy: JaxPlanetPolicy,
    cfg: TrainConfig,
) -> tuple[
    jax.Array, JaxEnvState, JaxTurnBatch, JaxTransitionBatch, dict[str, jax.Array]
]:
    def scan_step(carry, _):
        key, state, batch = carry
        key, learner_key, opp_key, reset_key = jax.random.split(key, 4)
        flat_self, flat_candidate, flat_global, flat_mask, flat_decision = (
            flatten_batch(batch)
        )
        output = policy.apply(
            train_state.params, flat_self, flat_candidate, flat_global, flat_mask
        )
        target, bucket, log_prob, _entropy = sample_actions(
            learner_key, output, deterministic=False
        )
        learner_action = build_action_from_batch(batch, target, bucket, cfg)

        opp_game = state.game._replace(
            player=jnp.ones_like(state.game.step, dtype=jnp.int32)
        )
        opp_batch = jax.vmap(lambda game: encode_turn(game, cfg.env))(opp_game)
        opp_flat = flatten_batch(opp_batch)
        opp_output = policy.apply(
            train_state.params, opp_flat[0], opp_flat[1], opp_flat[2], opp_flat[3]
        )
        opp_target, opp_bucket, _opp_lp, _ = sample_actions(
            opp_key, opp_output, deterministic=cfg.self_play_deterministic
        )
        opponent_action = build_action_from_batch(
            opp_batch, opp_target, opp_bucket, cfg
        )

        next_state, result = batched_step(
            state, learner_action, opponent_action, cfg.env
        )
        reset_keys = jax.random.split(reset_key, batch.self_features.shape[0])
        reset_states, reset_batches = batched_reset(reset_keys, cfg.env)

        def maybe_reset(new, old):
            cond = result.done.reshape(result.done.shape + (1,) * (old.ndim - 1))
            return jnp.where(cond, new, old)

        next_state = jax.tree.map(maybe_reset, reset_states, next_state)
        next_batch = jax.tree.map(maybe_reset, reset_batches, result.batch)
        transition = {
            "self_features": batch.self_features,
            "candidate_features": batch.candidate_features,
            "global_features": batch.global_features,
            "candidate_mask": batch.candidate_mask,
            "decision_mask": flat_decision.reshape(batch.decision_mask.shape),
            "target_index": target.reshape(batch.decision_mask.shape),
            "ship_bucket": bucket.reshape(batch.decision_mask.shape),
            "log_prob": log_prob.reshape(batch.decision_mask.shape),
            "value": output.value.reshape(batch.decision_mask.shape),
            "reward": result.reward,
            "done": result.done,
        }
        return (key, next_state, next_batch), transition

    (key, env_state, turn_batch), data = jax.lax.scan(
        scan_step, (key, env_state, turn_batch), None, length=cfg.ppo.rollout_steps
    )
    returns_step = discounted_returns(data["reward"], data["done"], cfg.ppo.gamma)
    returns = jnp.broadcast_to(returns_step[..., None], data["value"].shape)
    advantages = returns - data["value"]
    transitions = JaxTransitionBatch(
        self_features=data["self_features"],
        candidate_features=data["candidate_features"],
        global_features=data["global_features"],
        candidate_mask=data["candidate_mask"],
        decision_mask=data["decision_mask"],
        target_index=data["target_index"],
        ship_bucket=data["ship_bucket"],
        log_prob=data["log_prob"],
        returns=returns,
        advantages=advantages,
    )
    metrics = {
        "env_steps": jnp.array(
            cfg.ppo.rollout_steps * turn_batch.self_features.shape[0], dtype=jnp.float32
        ),
        "samples": transitions.decision_mask.astype(jnp.float32).sum(),
        "episode_done": data["done"].astype(jnp.float32).sum(),
    }
    return key, env_state, turn_batch, transitions, metrics


def discounted_returns(rewards: jax.Array, done: jax.Array, gamma: float) -> jax.Array:
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
    policy: JaxPlanetPolicy,
    batch: JaxTransitionBatch,
    cfg: TrainConfig,
) -> tuple[JaxTrainState, dict[str, jax.Array]]:
    mask = batch.decision_mask.reshape(-1).astype(jnp.float32)
    self_features = batch.self_features.reshape(-1, self_feature_dim())
    candidate_features = batch.candidate_features.reshape(
        -1, cfg.env.candidate_count, candidate_feature_dim()
    )
    global_features = batch.global_features.reshape(-1, global_feature_dim())
    candidate_mask = batch.candidate_mask.reshape(-1, cfg.env.candidate_count)
    target = batch.target_index.reshape(-1)
    bucket = batch.ship_bucket.reshape(-1)
    old_log_prob = batch.log_prob.reshape(-1)
    returns = batch.returns.reshape(-1)
    advantages = batch.advantages.reshape(-1)
    advantages = (advantages - masked_mean(advantages, mask)) / jnp.sqrt(
        masked_mean((advantages - masked_mean(advantages, mask)) ** 2, mask) + 1e-8
    )

    def loss_fn(params):
        output = policy.apply(
            params, self_features, candidate_features, global_features, candidate_mask
        )
        new_log_prob, entropy = action_log_prob_and_entropy(output, target, bucket)
        ratio = jnp.exp(new_log_prob - old_log_prob)
        policy_loss = -masked_mean(
            jnp.minimum(
                advantages * ratio,
                advantages
                * jnp.clip(ratio, 1.0 - cfg.ppo.clip_coef, 1.0 + cfg.ppo.clip_coef),
            ),
            mask,
        )
        value_loss = 0.5 * masked_mean((returns - output.value) ** 2, mask)
        entropy_loss = masked_mean(entropy, mask)
        loss = (
            policy_loss + cfg.ppo.vf_coef * value_loss - cfg.ppo.ent_coef * entropy_loss
        )
        return loss, {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy_loss,
            "loss": loss,
        }

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        train_state.params
    )
    updates, opt_state = train_state.optimizer.update(
        grads, train_state.opt_state, train_state.params
    )
    params = optax.apply_updates(train_state.params, updates)
    metrics = dict(metrics)
    metrics["total_loss"] = loss
    return JaxTrainState(
        params=params, opt_state=opt_state, optimizer=train_state.optimizer
    ), metrics


def masked_mean(x: jax.Array, mask: jax.Array) -> jax.Array:
    return (x * mask).sum() / jnp.maximum(mask.sum(), 1.0)
