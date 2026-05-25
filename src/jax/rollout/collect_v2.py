from __future__ import annotations

import jax
import jax.numpy as jnp

from src.config import TrainConfig
from src.jax.encode_dispatch import encode_turn_dispatch
from src.jax.env import (
    JaxEnvState,
    assign_learner_players,
    batched_reset,
    batched_step,
)
from src.jax.features_v2 import JaxTurnBatchV2
from src.jax.policy_v2 import edge_action_count
from src.jax.ppo_update import discounted_returns
from src.jax.rollout.types import JaxTrainState, JaxTransitionBatchV2
from src.opponents.jax_actions.builders_v2 import (
    _sample_shielded_sequence_v2_with_params,
    build_action_from_edge_batch,
    build_random_action_from_edge_batch,
)
from src.game.trajectory_shield import apply_trajectory_shield_to_turn_batch_v2


def _rollout_metrics_v2(data, cfg: TrainConfig, env_count: int) -> dict[str, jax.Array]:
    done_float = data["done"].astype(jnp.float32)
    episode_done = done_float.sum()
    return {
        "env_steps": jnp.array(
            cfg.training.rollout_steps * env_count,
            dtype=jnp.float32,
        ),
        "reward_mean": data["reward"].mean(),
        "episode_done": episode_done,
        "episodes_2p": jnp.where(cfg.task.player_count == 2, episode_done, 0.0),
        "episodes_4p": jnp.where(cfg.task.player_count == 4, episode_done, 0.0),
        "win_rate_2p": jnp.where(
            cfg.task.player_count == 2,
            (data["terminal_is_first"] * done_float).sum()
            / jnp.maximum(episode_done, 1.0),
            0.0,
        ),
        "first_place_rate_4p": jnp.array(0.0, dtype=jnp.float32),
        "average_placement_4p": jnp.array(0.0, dtype=jnp.float32),
        "loss_sample_count_2p": jnp.array(0.0, dtype=jnp.float32),
        "loss_sample_count_4p": jnp.array(0.0, dtype=jnp.float32),
    }


def collect_rollout_jax_v2(
    key: jax.Array,
    env_state: JaxEnvState,
    turn_batch: JaxTurnBatchV2,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    opponent_params_by_player: tuple[dict, ...] | None = None,
    stage_view=None,
    historical_params_pool: dict | None = None,
    update: int = 0,
    env_index_offset: int | jax.Array = 0,
):
    del opponent_params_by_player, stage_view, historical_params_pool, update
    if cfg.task.player_count != 2:
        raise ValueError(
            "v2 rollout currently supports 2p random-opponent training only; "
            f"got player_count={cfg.task.player_count}."
        )
    if cfg.opponents.mode.opponent != "random":
        raise ValueError(
            "v2 rollout currently supports opponent='random' only; "
            f"got {cfg.opponents.mode.opponent!r}."
        )

    env_count = turn_batch.planet_features.shape[0]
    env_indices = jnp.arange(env_count, dtype=jnp.int32) + jnp.asarray(
        env_index_offset, dtype=jnp.int32
    )
    edge_count = edge_action_count(cfg.task)

    def scan_step(carry, _):
        key, state, batch, opp_batch_cache = carry
        key, learner_key, opp_key, reset_key = jax.random.split(key, 4)
        sample = _sample_shielded_sequence_v2_with_params(
            learner_key,
            state.game,
            batch,
            train_state.params,
            policy,
            cfg,
            deterministic=False,
        )
        learner_action = build_action_from_edge_batch(
            state.game, batch, sample.target_index, sample.ship_bucket, cfg
        )
        opp_game = state.game._replace(
            player=(1 - state.learner_player).astype(jnp.int32)
        )
        opp_shielded = jax.vmap(
            lambda game_row, batch_row: apply_trajectory_shield_to_turn_batch_v2(
                game_row, batch_row, cfg.task
            )
        )(opp_game, opp_batch_cache)
        opponent_action = build_random_action_from_edge_batch(
            opp_key,
            opp_game,
            opp_shielded.batch,
            cfg,
            opp_shielded.ship_bucket_mask.reshape(
                env_count, edge_count, cfg.task.ship_bucket_count
            ),
        )
        next_state, result = batched_step(
            state, learner_action, opponent_action, cfg.task, cfg.reward
        )

        def maybe_reset(new, old):
            cond = result.done.reshape(result.done.shape + (1,) * (old.ndim - 1))
            return jnp.where(cond, new, old)

        def reset_branch(_):
            reset_keys = jax.random.split(reset_key, env_count)
            reset_states, reset_batches = batched_reset(reset_keys, cfg.task)
            reset_episode_counts = state.episode_count + result.done.astype(jnp.int32)
            reset_states, reset_batches = assign_learner_players(
                reset_states,
                env_indices,
                reset_episode_counts,
                cfg.task,
                cfg.opponents.mode.alternate_player_sides,
            )
            merged_state = jax.tree.map(maybe_reset, reset_states, next_state)
            merged_batch = jax.tree.map(maybe_reset, reset_batches, result.batch)
            return merged_state, merged_batch

        next_state, next_batch = jax.lax.cond(
            jnp.any(result.done),
            reset_branch,
            lambda _: (next_state, result.batch),
            operand=None,
        )
        next_opp_game = next_state.game._replace(
            player=(1 - next_state.learner_player).astype(jnp.int32)
        )
        next_opp_batch_cache = jax.vmap(
            lambda game: encode_turn_dispatch(game, cfg.task)
        )(next_opp_game)
        transition = {
            "planet_features": batch.planet_features,
            "planet_mask": batch.planet_mask,
            "edge_features": batch.edge_features,
            "edge_mask": batch.edge_mask,
            "edge_src_ids": batch.edge_src_ids,
            "edge_tgt_ids": batch.edge_tgt_ids,
            "global_features": batch.global_features,
            "theta_ref": batch.theta_ref,
            "player_count": jnp.full((env_count,), cfg.task.player_count, dtype=jnp.int32),
            "ship_bucket_mask": sample.ship_bucket_mask,
            "target_index": sample.target_index,
            "ship_bucket": sample.ship_bucket,
            "log_prob": sample.log_prob,
            "value": sample.value,
            "reward": result.reward,
            "done": result.done,
            "terminal_is_first": result.terminal_is_first,
        }
        return (key, next_state, next_batch, next_opp_batch_cache), transition

    initial_opp_game = env_state.game._replace(
        player=(1 - env_state.learner_player).astype(jnp.int32)
    )
    initial_opp_batch_cache = jax.vmap(
        lambda game: encode_turn_dispatch(game, cfg.task)
    )(initial_opp_game)
    (_, env_state, turn_batch, _), data = jax.lax.scan(
        scan_step,
        (key, env_state, turn_batch, initial_opp_batch_cache),
        None,
        length=cfg.training.rollout_steps,
    )
    returns_step = discounted_returns(data["reward"], data["done"], cfg.training.gamma)
    returns = jnp.broadcast_to(
        returns_step[..., None], data["target_index"].shape
    )
    advantages = returns - data["value"][..., None]
    transitions = JaxTransitionBatchV2(
        planet_features=data["planet_features"],
        planet_mask=data["planet_mask"],
        edge_features=data["edge_features"],
        edge_mask=data["edge_mask"],
        edge_src_ids=data["edge_src_ids"],
        edge_tgt_ids=data["edge_tgt_ids"],
        global_features=data["global_features"],
        theta_ref=data["theta_ref"],
        player_count=data["player_count"],
        ship_bucket_mask=data["ship_bucket_mask"],
        target_index=data["target_index"],
        ship_bucket=data["ship_bucket"],
        log_prob=data["log_prob"],
        returns=returns,
        advantages=advantages,
    )
    metrics = _rollout_metrics_v2(data, cfg, env_count)
    return key, env_state, turn_batch, transitions, metrics
