from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from .config import TrainConfig, default_train_config_path, load_train_config
from .env import OrbitWarsEnv
from .features import (
    NO_OP_CANDIDATE_INDEX,
    TurnBatch,
    candidate_feature_dim,
    global_feature_dim,
    self_feature_dim,
    ship_count_for_bucket,
)
from .game_types import GameState, PlanetState
from .opponents import SelfPlayOpponent, SelfPlayOpponentPool, build_opponent
from .normalization import ObservationNormalizer
from .policy import build_policy
from .ppo import TransitionBatch, ppo_update, sample_actions

if TYPE_CHECKING:
    from .jax_env import JaxAction


@dataclass(slots=True)
class JaxBatchedEnv:
    """Torch-training wrapper around batched JAX environment state.

    This adapter is used only when ``env_backend='jax'`` with the Torch PPO
    stack. It caches JIT-compiled helpers alongside the current vectorized JAX
    state to avoid recompiling every rollout.
    """

    states: object
    turn_batches: object
    step_fn: object | None = None
    reset_fn: object | None = None
    opponent_encode_fn: object | None = None


def make_jax_batched_env(cfg: TrainConfig, seeds: np.ndarray) -> JaxBatchedEnv:
    """Create a batched JAX environment wrapper for Torch PPO rollouts."""

    import jax
    import jax.numpy as jnp

    from .jax_env import (
        batched_reset as jax_batched_reset,
        batched_step as jax_batched_step,
        reset as jax_reset,
    )
    from .jax_features import encode_turn as jax_encode_turn

    keys = jax.vmap(jax.random.PRNGKey)(seeds.astype(np.uint32))
    states, turn_batches = jax_batched_reset(keys, cfg.env)
    step_fn = jax.jit(lambda s, a0, a1: jax_batched_step(s, a0, a1, cfg.env))
    reset_fn = jax.jit(lambda key: jax_reset(key, cfg.env))
    opponent_encode_fn = jax.jit(
        jax.vmap(
            lambda game: jax_encode_turn(
                game._replace(player=jnp.array(1, dtype=jnp.int32)), cfg.env
            )
        )
    )
    return JaxBatchedEnv(
        states=states,
        turn_batches=turn_batches,
        step_fn=step_fn,
        reset_fn=reset_fn,
        opponent_encode_fn=opponent_encode_fn,
    )


@dataclass(slots=True)
class StepGroup:
    """Rollout bookkeeping for all source decisions emitted in one env step."""

    indices: list[int]
    reward: float
    done: bool
    value: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(default_train_config_path()))
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    """Resolve a configured Torch device string into a ``torch.device``."""

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch random number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_rollout(
    envs: list[OrbitWarsEnv] | JaxBatchedEnv,
    batches: list[TurnBatch],
    policy: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    next_seed: int,
    normalizer: ObservationNormalizer | None = None,
    running_episode_rewards: list[float] | None = None,
) -> tuple[TransitionBatch, list[TurnBatch] | JaxBatchedEnv, int, dict[str, float]]:
    """Collect one PPO rollout from either configured environment backend."""

    if isinstance(envs, JaxBatchedEnv):
        return collect_jax_rollout(
            envs, policy, cfg, device, next_seed, normalizer, running_episode_rewards
        )

    empty_candidate = (cfg.env.candidate_count, candidate_feature_dim())
    self_rows: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    global_rows: list[np.ndarray] = []
    candidate_masks: list[np.ndarray] = []
    target_indices: list[int] = []
    ship_buckets: list[int] = []
    log_probs: list[float] = []
    values: list[float] = []
    step_ids: list[int] = []
    groups_per_env: list[list[StepGroup]] = [[] for _ in envs]
    episode_rewards: list[float] = []
    episode_wins = 0
    rollout_step_id = 0
    decisions_total = 0
    emitted_moves_total = 0
    candidate_valid_total = 0.0
    candidate_source_rows = 0
    candidate_owner_totals = {"enemy": 0.0, "neutral": 0.0, "friendly": 0.0}
    if running_episode_rewards is None:
        running_episode_rewards = [0.0 for _ in envs]

    for _ in range(cfg.ppo.rollout_steps):
        offsets = np.cumsum(
            [0] + [batch.self_features.shape[0] for batch in batches[:-1]]
        )
        merged = merge_batches(batches)
        rollout_candidate_stats = candidate_diagnostics(merged)
        candidate_valid_total += rollout_candidate_stats["candidate_valid_total"]
        candidate_source_rows += int(rollout_candidate_stats["candidate_source_rows"])
        candidate_owner_totals["enemy"] += rollout_candidate_stats[
            "candidate_enemy_total"
        ]
        candidate_owner_totals["neutral"] += rollout_candidate_stats[
            "candidate_neutral_total"
        ]
        candidate_owner_totals["friendly"] += rollout_candidate_stats[
            "candidate_friendly_total"
        ]
        if normalizer is not None:
            normalizer.update(merged)
            policy_batch = normalizer.normalize_batch(merged)
        else:
            policy_batch = merged
        row_values = np.zeros((merged.self_features.shape[0],), dtype=np.float32)
        if merged.self_features.shape[0] > 0:
            with torch.inference_mode():
                outputs = policy(
                    torch.from_numpy(policy_batch.self_features).to(device),
                    torch.from_numpy(policy_batch.candidate_features).to(device),
                    torch.from_numpy(policy_batch.global_features).to(device),
                    torch.from_numpy(policy_batch.candidate_mask).to(device).bool(),
                )
                sampled = sample_actions(outputs, deterministic=False)
                row_values = outputs.value.detach().cpu().numpy()
                sampled_target_index = sampled.target_index.detach().cpu().numpy()
                sampled_ship_bucket = sampled.ship_bucket.detach().cpu().numpy()
                sampled_log_prob = sampled.log_prob.detach().cpu().numpy()
        else:
            sampled_target_index = np.zeros((0,), dtype=np.int64)
            sampled_ship_bucket = np.zeros((0,), dtype=np.int64)
            sampled_log_prob = np.zeros((0,), dtype=np.float32)

        next_batches: list[TurnBatch] = []
        for env_idx, env in enumerate(envs):
            batch = batches[env_idx]
            start = int(offsets[env_idx])
            moves = []
            group_indices: list[int] = []
            for local_idx, context in enumerate(batch.contexts):
                global_idx = start + local_idx
                self_rows.append(policy_batch.self_features[global_idx])
                candidate_rows.append(policy_batch.candidate_features[global_idx])
                global_rows.append(policy_batch.global_features[global_idx])
                candidate_masks.append(batch.candidate_mask[local_idx])
                values.append(float(row_values[global_idx]))
                step_ids.append(rollout_step_id)
                tgt_idx = (
                    int(sampled_target_index[global_idx])
                    if batch.self_features.shape[0] > 0
                    else 0
                )
                bucket_idx = (
                    int(sampled_ship_bucket[global_idx])
                    if batch.self_features.shape[0] > 0
                    else 0
                )
                is_valid_send = (
                    tgt_idx > 0
                    and tgt_idx < len(context.candidate_ids)
                    and context.candidate_mask[tgt_idx]
                    and bucket_idx > 0
                )
                target_indices.append(tgt_idx)
                ship_buckets.append(bucket_idx)
                log_probs.append(
                    float(sampled_log_prob[global_idx])
                    if batch.self_features.shape[0] > 0
                    else 0.0
                )
                group_indices.append(len(values) - 1)
                if not is_valid_send:
                    continue
                ships = ship_count_for_bucket(
                    context.source_ships, bucket_idx, cfg.env.ship_bucket_count
                )
                if ships <= 0:
                    continue
                src_planet = find_planet(batch.state.planets, context.source_id)
                if src_planet is None or src_planet.ships < ships:
                    continue
                moves.append(
                    [context.source_id, float(context.target_angles[tgt_idx]), ships]
                )
            decisions_total += len(group_indices)
            emitted_moves_total += len(moves)
            step_value = (
                float(np.mean([values[idx] for idx in group_indices]))
                if group_indices
                else 0.0
            )
            result = env.step(moves)
            running_episode_rewards[env_idx] += float(result.reward)
            groups_per_env[env_idx].append(
                StepGroup(
                    indices=group_indices,
                    reward=float(result.reward),
                    done=result.done,
                    value=step_value,
                )
            )
            rollout_step_id += 1
            if result.done:
                episode_rewards.append(running_episode_rewards[env_idx])
                episode_wins += int(
                    float(result.info.get("terminal_reward", 0.0)) > 0.0
                )
                running_episode_rewards[env_idx] = 0.0
                next_seed += 1
                next_batch = env.reset(seed=next_seed)
            else:
                next_batch = result.batch
            next_batches.append(next_batch)
        batches = next_batches

    returns: list[float] = [0.0] * len(values)
    advantages: list[float] = [0.0] * len(values)
    next_state_values = bootstrap_values(policy, batches, device, normalizer)
    for env_idx, groups in enumerate(groups_per_env):
        future_return = next_state_values[env_idx]
        for group in reversed(groups):
            future_return = group.reward + cfg.ppo.gamma * future_return * (
                1.0 - float(group.done)
            )
            for idx in group.indices:
                returns[idx] = future_return
                advantages[idx] = future_return - group.value
    batch = TransitionBatch(
        self_features=torch.from_numpy(
            np.asarray(self_rows, dtype=np.float32).reshape(-1, self_feature_dim())
        ),
        candidate_features=torch.from_numpy(
            np.asarray(candidate_rows, dtype=np.float32).reshape(
                -1, empty_candidate[0], empty_candidate[1]
            )
        ),
        global_features=torch.from_numpy(
            np.asarray(global_rows, dtype=np.float32).reshape(-1, global_feature_dim())
        ),
        candidate_mask=torch.from_numpy(
            np.asarray(candidate_masks, dtype=bool).reshape(-1, cfg.env.candidate_count)
        ),
        target_index=torch.tensor(target_indices, dtype=torch.long),
        ship_bucket=torch.tensor(ship_buckets, dtype=torch.long),
        log_prob=torch.tensor(log_probs, dtype=torch.float32),
        returns=torch.tensor(returns, dtype=torch.float32),
        advantages=torch.tensor(advantages, dtype=torch.float32),
        step_id=torch.tensor(step_ids, dtype=torch.long),
    )
    candidate_owner_total = sum(candidate_owner_totals.values())
    stats = {
        "episode_reward_mean": float(np.mean(episode_rewards))
        if episode_rewards
        else 0.0,
        "episode_reward_median": float(np.median(episode_rewards))
        if episode_rewards
        else 0.0,
        "episodes_finished": float(len(episode_rewards)),
        "win_rate": episode_wins / len(episode_rewards) if episode_rewards else 0.0,
        "samples": float(len(values)),
        "env_steps": float(cfg.ppo.rollout_steps * len(envs)),
        "decisions_per_step": decisions_total / max(rollout_step_id, 1),
        "moves_emitted_per_step": emitted_moves_total / max(rollout_step_id, 1),
        "move_emit_rate": emitted_moves_total / decisions_total
        if decisions_total
        else 0.0,
        "candidate_valid_avg": candidate_valid_total / candidate_source_rows
        if candidate_source_rows
        else 0.0,
        "candidate_enemy_share": candidate_owner_totals["enemy"] / candidate_owner_total
        if candidate_owner_total
        else 0.0,
        "candidate_neutral_share": candidate_owner_totals["neutral"]
        / candidate_owner_total
        if candidate_owner_total
        else 0.0,
        "candidate_friendly_share": candidate_owner_totals["friendly"]
        / candidate_owner_total
        if candidate_owner_total
        else 0.0,
    }
    return batch, batches, next_seed, stats


def collect_jax_rollout(
    envs: JaxBatchedEnv,
    policy: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    next_seed: int,
    normalizer: ObservationNormalizer | None = None,
    running_episode_rewards: list[float] | None = None,
) -> tuple[TransitionBatch, JaxBatchedEnv, int, dict[str, float]]:
    """Collect Torch PPO transitions from the JAX environment backend.

    The policy remains a Torch module, so encoded JAX turn batches are converted
    to NumPy ``TurnBatch`` rows before action sampling. Environment stepping,
    opponent encoding, and resets remain vectorized in JAX.
    """

    import jax
    import jax.numpy as jnp

    from .jax_env import (
        JaxAction,
        batched_step as jax_batched_step,
        reset as jax_reset,
    )
    from .jax_features import encode_turn as jax_encode_turn

    num_envs = int(envs.turn_batches.self_features.shape[0])
    if running_episode_rewards is None:
        running_episode_rewards = [0.0 for _ in range(num_envs)]

    self_rows: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    global_rows: list[np.ndarray] = []
    candidate_masks: list[np.ndarray] = []
    target_indices: list[int] = []
    ship_buckets: list[int] = []
    log_probs: list[float] = []
    values: list[float] = []
    step_ids: list[int] = []
    groups_per_env: list[list[StepGroup]] = [[] for _ in range(num_envs)]
    episode_rewards: list[float] = []
    episode_wins = 0
    rollout_step_id = 0
    decisions_total = 0
    emitted_moves_total = 0
    candidate_valid_total = 0.0
    candidate_source_rows = 0
    candidate_owner_totals = {"enemy": 0.0, "neutral": 0.0, "friendly": 0.0}

    states = envs.states
    jax_batches = envs.turn_batches
    latest_batches = jax_turn_batches_to_numpy(jax_batches, cfg)
    step_fn = envs.step_fn or jax.jit(
        lambda s, a0, a1: jax_batched_step(s, a0, a1, cfg.env)
    )
    reset_fn = envs.reset_fn or jax.jit(lambda key: jax_reset(key, cfg.env))
    opponent_encode_fn = envs.opponent_encode_fn or jax.jit(
        jax.vmap(
            lambda game: jax_encode_turn(
                game._replace(player=jnp.array(1, dtype=jnp.int32)), cfg.env
            )
        )
    )

    for _ in range(cfg.ppo.rollout_steps):
        rollout_candidate_stats = candidate_diagnostics(merge_batches(latest_batches))
        candidate_valid_total += rollout_candidate_stats["candidate_valid_total"]
        candidate_source_rows += int(rollout_candidate_stats["candidate_source_rows"])
        candidate_owner_totals["enemy"] += rollout_candidate_stats[
            "candidate_enemy_total"
        ]
        candidate_owner_totals["neutral"] += rollout_candidate_stats[
            "candidate_neutral_total"
        ]
        candidate_owner_totals["friendly"] += rollout_candidate_stats[
            "candidate_friendly_total"
        ]

        learner_sample = sample_policy_for_batches(
            latest_batches,
            policy,
            cfg,
            device,
            normalizer,
            deterministic=False,
            update_normalizer=True,
        )
        row_values = learner_sample["values"]
        sampled_target_index = learner_sample["target_index"]
        sampled_ship_bucket = learner_sample["ship_bucket"]
        sampled_log_prob = learner_sample["log_prob"]
        policy_batch = learner_sample["policy_batch"]
        offsets = learner_sample["offsets"]

        source_id, angle, ships_arr, valid = empty_action_arrays(num_envs, cfg)
        for env_idx, batch in enumerate(latest_batches):
            start = int(offsets[env_idx])
            group_indices: list[int] = []
            move_slot = 0
            for local_idx, context in enumerate(batch.contexts):
                global_idx = start + local_idx
                self_rows.append(policy_batch.self_features[global_idx])
                candidate_rows.append(policy_batch.candidate_features[global_idx])
                global_rows.append(policy_batch.global_features[global_idx])
                candidate_masks.append(batch.candidate_mask[local_idx])
                values.append(float(row_values[global_idx]))
                step_ids.append(rollout_step_id)
                tgt_idx = int(sampled_target_index[global_idx])
                bucket_idx = int(sampled_ship_bucket[global_idx])
                target_indices.append(tgt_idx)
                ship_buckets.append(bucket_idx)
                log_probs.append(float(sampled_log_prob[global_idx]))
                group_indices.append(len(values) - 1)
                move_slot = fill_action_slot_from_context(
                    context,
                    tgt_idx,
                    bucket_idx,
                    cfg,
                    env_idx,
                    move_slot,
                    source_id,
                    angle,
                    ships_arr,
                    valid,
                )
            decisions_total += len(group_indices)
            emitted_moves_total += move_slot
            step_value = (
                float(np.mean([values[idx] for idx in group_indices]))
                if group_indices
                else 0.0
            )
            groups_per_env[env_idx].append(
                StepGroup(
                    indices=group_indices, reward=0.0, done=False, value=step_value
                )
            )

        learner_action = JaxAction(
            jnp.asarray(source_id),
            jnp.asarray(angle),
            jnp.asarray(ships_arr),
            jnp.asarray(valid),
        )
        opponent_batches = jax_turn_batches_to_numpy(
            opponent_encode_fn(states.game), cfg
        )
        opponent_action = build_jax_opponent_action(
            opponent_batches,
            policy,
            cfg,
            device,
            normalizer,
            num_envs,
        )
        states, results = step_fn(states, learner_action, opponent_action)
        rewards = np.asarray(results.reward)
        dones = np.asarray(results.done)
        terminals = np.asarray(results.terminal_reward)
        for env_idx in range(num_envs):
            running_episode_rewards[env_idx] += float(rewards[env_idx])
            groups_per_env[env_idx][-1].reward = float(rewards[env_idx])
            groups_per_env[env_idx][-1].done = bool(dones[env_idx])
            rollout_step_id += 1
            if dones[env_idx]:
                episode_rewards.append(running_episode_rewards[env_idx])
                episode_wins += int(float(terminals[env_idx]) > 0.0)
                running_episode_rewards[env_idx] = 0.0
                next_seed += 1
                new_state, new_batch = reset_fn(jax.random.PRNGKey(next_seed))
                states = jax.tree.map(
                    lambda old, new: old.at[env_idx].set(new), states, new_state
                )
                results = results._replace(
                    batch=jax.tree.map(
                        lambda old, new: old.at[env_idx].set(new),
                        results.batch,
                        new_batch,
                    )
                )
        jax_batches = results.batch
        latest_batches = jax_turn_batches_to_numpy(jax_batches, cfg)

    returns: list[float] = [0.0] * len(values)
    advantages: list[float] = [0.0] * len(values)
    for _env_idx, groups in enumerate(groups_per_env):
        future_return = 0.0
        for group in reversed(groups):
            future_return = group.reward + cfg.ppo.gamma * future_return * (
                1.0 - float(group.done)
            )
            for idx in group.indices:
                returns[idx] = future_return
                advantages[idx] = future_return - group.value

    batch = TransitionBatch(
        self_features=torch.from_numpy(
            np.asarray(self_rows, dtype=np.float32).reshape(-1, self_feature_dim())
        ),
        candidate_features=torch.from_numpy(
            np.asarray(candidate_rows, dtype=np.float32).reshape(
                -1, cfg.env.candidate_count, candidate_feature_dim()
            )
        ),
        global_features=torch.from_numpy(
            np.asarray(global_rows, dtype=np.float32).reshape(-1, global_feature_dim())
        ),
        candidate_mask=torch.from_numpy(
            np.asarray(candidate_masks, dtype=bool).reshape(-1, cfg.env.candidate_count)
        ),
        target_index=torch.tensor(target_indices, dtype=torch.long),
        ship_bucket=torch.tensor(ship_buckets, dtype=torch.long),
        log_prob=torch.tensor(log_probs, dtype=torch.float32),
        returns=torch.tensor(returns, dtype=torch.float32),
        advantages=torch.tensor(advantages, dtype=torch.float32),
        step_id=torch.tensor(step_ids, dtype=torch.long),
    )
    candidate_owner_total = sum(candidate_owner_totals.values())
    stats = {
        "episode_reward_mean": float(np.mean(episode_rewards))
        if episode_rewards
        else 0.0,
        "episode_reward_median": float(np.median(episode_rewards))
        if episode_rewards
        else 0.0,
        "episodes_finished": float(len(episode_rewards)),
        "win_rate": episode_wins / len(episode_rewards) if episode_rewards else 0.0,
        "samples": float(len(values)),
        "env_steps": float(cfg.ppo.rollout_steps * num_envs),
        "decisions_per_step": decisions_total / max(rollout_step_id, 1),
        "moves_emitted_per_step": emitted_moves_total / max(rollout_step_id, 1),
        "move_emit_rate": emitted_moves_total / decisions_total
        if decisions_total
        else 0.0,
        "candidate_valid_avg": candidate_valid_total / candidate_source_rows
        if candidate_source_rows
        else 0.0,
        "candidate_enemy_share": candidate_owner_totals["enemy"] / candidate_owner_total
        if candidate_owner_total
        else 0.0,
        "candidate_neutral_share": candidate_owner_totals["neutral"]
        / candidate_owner_total
        if candidate_owner_total
        else 0.0,
        "candidate_friendly_share": candidate_owner_totals["friendly"]
        / candidate_owner_total
        if candidate_owner_total
        else 0.0,
    }
    return (
        batch,
        JaxBatchedEnv(
            states=states,
            turn_batches=jax_batches,
            step_fn=step_fn,
            reset_fn=reset_fn,
            opponent_encode_fn=opponent_encode_fn,
        ),
        next_seed,
        stats,
    )


def sample_policy_for_batches(
    batches: list[TurnBatch],
    policy: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    normalizer: ObservationNormalizer | None,
    *,
    deterministic: bool,
    update_normalizer: bool,
) -> dict[str, object]:
    """Run a Torch policy over merged turn batches and return NumPy actions."""

    offsets = np.cumsum([0] + [batch.self_features.shape[0] for batch in batches[:-1]])
    merged = merge_batches(batches)
    if normalizer is not None and update_normalizer:
        normalizer.update(merged)
    policy_batch = (
        normalizer.normalize_batch(merged) if normalizer is not None else merged
    )
    row_count = merged.self_features.shape[0]
    if row_count == 0:
        return {
            "target_index": np.zeros((0,), dtype=np.int64),
            "ship_bucket": np.zeros((0,), dtype=np.int64),
            "log_prob": np.zeros((0,), dtype=np.float32),
            "values": np.zeros((0,), dtype=np.float32),
            "policy_batch": policy_batch,
            "offsets": offsets,
        }
    with torch.inference_mode():
        outputs = policy(
            torch.from_numpy(policy_batch.self_features).to(device),
            torch.from_numpy(policy_batch.candidate_features).to(device),
            torch.from_numpy(policy_batch.global_features).to(device),
            torch.from_numpy(policy_batch.candidate_mask).to(device).bool(),
        )
        sampled = sample_actions(outputs, deterministic=deterministic)
    return {
        "target_index": sampled.target_index.detach().cpu().numpy(),
        "ship_bucket": sampled.ship_bucket.detach().cpu().numpy(),
        "log_prob": sampled.log_prob.detach().cpu().numpy(),
        "values": outputs.value.detach().cpu().numpy(),
        "policy_batch": policy_batch,
        "offsets": offsets,
    }


def empty_action_arrays(
    num_envs: int, cfg: TrainConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create mutable NumPy action buffers matching ``JaxAction`` layout."""

    from .jax_env import empty_action as jax_empty_action

    action = jax_empty_action(cfg.env)
    return (
        np.broadcast_to(
            np.asarray(action.source_id), (num_envs, cfg.env.max_fleets)
        ).copy(),
        np.broadcast_to(
            np.asarray(action.angle), (num_envs, cfg.env.max_fleets)
        ).copy(),
        np.broadcast_to(
            np.asarray(action.ships), (num_envs, cfg.env.max_fleets)
        ).copy(),
        np.broadcast_to(
            np.asarray(action.valid), (num_envs, cfg.env.max_fleets)
        ).copy(),
    )


def fill_action_slot_from_context(
    context: object,
    target_index: int,
    bucket_index: int,
    cfg: TrainConfig,
    env_idx: int,
    move_slot: int,
    source_id: np.ndarray,
    angle: np.ndarray,
    ships_arr: np.ndarray,
    valid: np.ndarray,
) -> int:
    """Append one valid decision context to a mutable JAX action buffer.

    Returns the next free move slot. Invalid no-op, masked, or over-capacity
    decisions leave the buffers unchanged.
    """

    if (
        target_index <= 0
        or target_index >= len(context.candidate_ids)
        or not context.candidate_mask[target_index]
        or bucket_index <= 0
        or move_slot >= cfg.env.max_fleets
    ):
        return move_slot
    ships = ship_count_for_bucket(
        context.source_ships, bucket_index, cfg.env.ship_bucket_count
    )
    if ships <= 0:
        return move_slot
    source_id[env_idx, move_slot] = context.source_id
    angle[env_idx, move_slot] = context.target_angles[target_index]
    ships_arr[env_idx, move_slot] = ships
    valid[env_idx, move_slot] = True
    return move_slot + 1


def build_jax_opponent_action(
    batches: list[TurnBatch],
    policy: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    normalizer: ObservationNormalizer | None,
    num_envs: int,
) -> "JaxAction":
    """Build opponent actions for the mixed JAX-env/Torch-policy path."""

    import jax.numpy as jnp

    from .jax_env import JaxAction

    source_id, angle, ships_arr, valid = empty_action_arrays(num_envs, cfg)
    if cfg.opponent == "self":
        sample = sample_policy_for_batches(
            batches,
            policy,
            cfg,
            device,
            normalizer,
            deterministic=cfg.self_play_deterministic,
            update_normalizer=False,
        )
        target_index = sample["target_index"]
        ship_bucket = sample["ship_bucket"]
        offsets = sample["offsets"]
        for env_idx, batch in enumerate(batches):
            start = int(offsets[env_idx])
            move_slot = 0
            for local_idx, context in enumerate(batch.contexts):
                global_idx = start + local_idx
                move_slot = fill_action_slot_from_context(
                    context,
                    int(target_index[global_idx]),
                    int(ship_bucket[global_idx]),
                    cfg,
                    env_idx,
                    move_slot,
                    source_id,
                    angle,
                    ships_arr,
                    valid,
                )
    elif cfg.opponent == "random":
        rng = np.random.default_rng()
        for env_idx, batch in enumerate(batches):
            move_slot = 0
            for context in batch.contexts:
                real_candidates = np.flatnonzero(context.candidate_mask.copy())
                real_candidates = real_candidates[
                    real_candidates != NO_OP_CANDIDATE_INDEX
                ]
                if context.source_ships < 20 or real_candidates.size == 0:
                    continue
                if move_slot >= cfg.env.max_fleets:
                    break
                target_idx = int(rng.choice(real_candidates))
                source_id[env_idx, move_slot] = context.source_id
                angle[env_idx, move_slot] = context.target_angles[target_idx]
                ships_arr[env_idx, move_slot] = max(1, context.source_ships // 2)
                valid[env_idx, move_slot] = True
                move_slot += 1
    return JaxAction(
        jnp.asarray(source_id),
        jnp.asarray(angle),
        jnp.asarray(ships_arr),
        jnp.asarray(valid),
    )


def jax_turn_batches_to_numpy(jax_batch: object, cfg: TrainConfig) -> list[TurnBatch]:
    """Convert batched JAX feature outputs into Python ``TurnBatch`` objects."""

    import jax

    from .features import DecisionContext

    arrays = jax.tree.map(np.asarray, jax_batch)
    batches: list[TurnBatch] = []
    for env_idx in range(arrays.self_features.shape[0]):
        mask = arrays.decision_mask[env_idx].astype(bool)
        row_idx = np.flatnonzero(mask)
        contexts = [
            DecisionContext(
                env_index=env_idx,
                source_id=int(arrays.source_ids[env_idx, i]),
                candidate_ids=[
                    int(x) for x in arrays.candidate_ids[env_idx, i].tolist()
                ],
                candidate_mask=arrays.candidate_mask[env_idx, i].astype(bool),
                ship_counts=[int(arrays.source_ships[env_idx, i])]
                * cfg.env.candidate_count,
                source_ships=int(arrays.source_ships[env_idx, i]),
                target_angles=[
                    float(x) for x in arrays.target_angles[env_idx, i].tolist()
                ],
            )
            for i in row_idx
        ]
        batches.append(
            TurnBatch(
                self_features=arrays.self_features[env_idx, row_idx].astype(np.float32),
                candidate_features=arrays.candidate_features[env_idx, row_idx].astype(
                    np.float32
                ),
                global_features=arrays.global_features[env_idx, row_idx].astype(
                    np.float32
                ),
                candidate_mask=arrays.candidate_mask[env_idx, row_idx].astype(bool),
                contexts=contexts,
                state=GameState(step=0, player=0, planets=[], fleets=[]),
            )
        )
    return batches


def candidate_diagnostics(batch: TurnBatch) -> dict[str, float]:
    """Summarize real, valid candidates in a batch of source-planet rows.

    Index 0 is the no-op action, so diagnostics exclude it and only count
    candidate slots that point to target planets and are currently valid.
    """

    if batch.candidate_mask.shape[0] == 0:
        return {
            "candidate_valid_total": 0.0,
            "candidate_source_rows": 0.0,
            "candidate_enemy_total": 0.0,
            "candidate_neutral_total": 0.0,
            "candidate_friendly_total": 0.0,
        }

    real_candidate_mask = batch.candidate_mask.copy()
    real_candidate_mask[:, NO_OP_CANDIDATE_INDEX] = False
    valid_targets = real_candidate_mask & (batch.candidate_features[:, :, 0] > 0.0)
    return {
        "candidate_valid_total": float(valid_targets.sum()),
        "candidate_source_rows": float(valid_targets.shape[0]),
        "candidate_enemy_total": float(
            (valid_targets & (batch.candidate_features[:, :, 3] > 0.5)).sum()
        ),
        "candidate_neutral_total": float(
            (valid_targets & (batch.candidate_features[:, :, 1] > 0.5)).sum()
        ),
        "candidate_friendly_total": float(
            (valid_targets & (batch.candidate_features[:, :, 2] > 0.5)).sum()
        ),
    }


def bootstrap_values(
    policy: torch.nn.Module,
    batches: list[TurnBatch],
    device: torch.device,
    normalizer: ObservationNormalizer | None = None,
) -> list[float]:
    """Estimate per-environment bootstrap values from the current batches."""

    merged = merge_batches(batches)
    if merged.self_features.shape[0] == 0:
        return [0.0 for _ in batches]
    offsets = np.cumsum([0] + [batch.self_features.shape[0] for batch in batches[:-1]])
    policy_batch = (
        normalizer.normalize_batch(merged) if normalizer is not None else merged
    )
    with torch.inference_mode():
        outputs = policy(
            torch.from_numpy(policy_batch.self_features).to(device),
            torch.from_numpy(policy_batch.candidate_features).to(device),
            torch.from_numpy(policy_batch.global_features).to(device),
            torch.from_numpy(policy_batch.candidate_mask).to(device).bool(),
        )
    values = outputs.value.detach().cpu().numpy()
    per_env = []
    for env_idx, batch in enumerate(batches):
        start = int(offsets[env_idx])
        count = batch.self_features.shape[0]
        per_env.append(
            0.0 if count == 0 else float(values[start : start + count].mean())
        )
    return per_env


def merge_batches(batches: list[TurnBatch]) -> TurnBatch:
    """Concatenate per-environment ``TurnBatch`` objects into one batch."""

    if not batches:
        raise ValueError("batches must not be empty")
    has_rows = any(batch.self_features.shape[0] > 0 for batch in batches)
    self_rows = (
        np.concatenate([batch.self_features for batch in batches], axis=0)
        if has_rows
        else np.zeros((0, self_feature_dim()), dtype=np.float32)
    )
    candidate_rows = (
        np.concatenate([batch.candidate_features for batch in batches], axis=0)
        if has_rows
        else np.zeros(
            (0, batches[0].candidate_features.shape[1], candidate_feature_dim()),
            dtype=np.float32,
        )
    )
    global_rows = (
        np.concatenate([batch.global_features for batch in batches], axis=0)
        if has_rows
        else np.zeros((0, global_feature_dim()), dtype=np.float32)
    )
    candidate_masks = (
        np.concatenate([batch.candidate_mask for batch in batches], axis=0)
        if has_rows
        else np.zeros((0, batches[0].candidate_mask.shape[1]), dtype=bool)
    )
    return TurnBatch(
        self_features=self_rows,
        candidate_features=candidate_rows,
        global_features=global_rows,
        candidate_mask=candidate_masks,
        contexts=[context for batch in batches for context in batch.contexts],
        state=batches[0].state,
    )


def save_checkpoint(
    save_dir: Path,
    run_name: str,
    update: int,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    normalizer: ObservationNormalizer | None = None,
    self_play_metadata: dict[str, object] | None = None,
) -> None:
    """Write latest and numbered Torch PPO checkpoints for a training run."""

    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "update": update,
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "normalizer": normalizer.state_dict() if normalizer is not None else None,
            "self_play": self_play_metadata,
        },
        run_dir / "ckpt_last.pt",
    )
    torch.save(
        {
            "update": update,
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "normalizer": normalizer.state_dict() if normalizer is not None else None,
            "self_play": self_play_metadata,
        },
        run_dir / f"ckpt_{update:06d}.pt",
    )


def find_planet(planets: list[PlanetState], planet_id: int) -> PlanetState | None:
    """Return the planet with ``planet_id`` or ``None`` if absent."""

    for planet in planets:
        if planet.id == planet_id:
            return planet
    return None


def append_jsonl(path: Path, record: dict[str, float | int]) -> None:
    """Append one JSON metrics record, creating the log directory if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    """Run training from the command-line configuration."""

    args = parse_args()
    cfg = load_train_config(args.config)
    if (
        cfg.env_backend.strip().lower() == "jax"
        and cfg.rl_backend.strip().lower() == "jax"
    ):
        from .jax_train import run_jax_training

        run_jax_training(cfg)
        return
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    opponent = build_opponent(cfg.opponent, cfg=cfg, device=device)
    next_seed = cfg.seed
    env_backend = cfg.env_backend.strip().lower()
    if env_backend == "jax":
        seeds = np.arange(next_seed, next_seed + cfg.ppo.num_envs, dtype=np.int64)
        envs: list[OrbitWarsEnv] | JaxBatchedEnv = make_jax_batched_env(cfg, seeds)
        batches: list[TurnBatch] | JaxBatchedEnv = envs
        next_seed += cfg.ppo.num_envs
    elif env_backend in {"kaggle", "python"}:
        envs = [
            OrbitWarsEnv(cfg, opponent, env_index=idx)
            for idx in range(cfg.ppo.num_envs)
        ]
        batches = []
        for env in envs:
            batches.append(env.reset(seed=next_seed))
            next_seed += 1
    else:
        raise ValueError(
            f"Unknown env_backend: {cfg.env_backend!r}; expected 'jax' or 'kaggle'."
        )
    policy = build_policy(
        architecture=cfg.model.architecture,
        self_dim=self_feature_dim(),
        candidate_dim=candidate_feature_dim(),
        global_dim=global_feature_dim(),
        candidate_count=cfg.env.candidate_count,
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        attention_heads=cfg.model.attention_heads,
    ).to(device)
    normalizer = (
        ObservationNormalizer(clip=cfg.model.obs_norm_clip)
        if cfg.model.normalize_observations
        else None
    )
    if isinstance(opponent, SelfPlayOpponent):
        opponent.sync_from(policy, normalizer)
    elif isinstance(opponent, SelfPlayOpponentPool):
        opponent.sync_from(policy, normalizer, update=0)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.ppo.lr)
    save_dir = Path(cfg.save_dir)
    log_path = Path("artifacts/rl_template/logs") / f"{cfg.run_name}.jsonl"
    total_env_steps = 0
    completed_episodes = 0
    env_count = cfg.ppo.num_envs if isinstance(envs, JaxBatchedEnv) else len(envs)
    running_episode_rewards = [0.0 for _ in range(env_count)]
    train_start_time = time.perf_counter()
    for update in range(1, cfg.ppo.total_updates + 1):
        update_start_time = time.perf_counter()
        rollout_start_time = time.perf_counter()
        batch, batches, next_seed, stats = collect_rollout(
            envs,
            batches,
            policy,
            cfg,
            device,
            next_seed,
            normalizer,
            running_episode_rewards,
        )
        rollout_seconds = time.perf_counter() - rollout_start_time
        ppo_start_time = time.perf_counter()
        metrics = ppo_update(
            policy,
            optimizer,
            batch,
            clip_coef=cfg.ppo.clip_coef,
            ent_coef=cfg.ppo.ent_coef,
            vf_coef=cfg.ppo.vf_coef,
            max_grad_norm=cfg.ppo.max_grad_norm,
            epochs=cfg.ppo.epochs,
            minibatch_size=cfg.ppo.minibatch_size,
            device=device,
        )
        ppo_seconds = time.perf_counter() - ppo_start_time
        update_seconds = time.perf_counter() - update_start_time
        elapsed_seconds = time.perf_counter() - train_start_time
        env_steps_per_sec = stats["env_steps"] / max(update_seconds, 1e-9)
        rollout_env_steps_per_sec = stats["env_steps"] / max(rollout_seconds, 1e-9)
        samples_per_sec = stats["samples"] / max(update_seconds, 1e-9)
        ppo_samples_per_sec = stats["samples"] / max(ppo_seconds, 1e-9)
        total_env_steps += int(stats["env_steps"])
        completed_episodes += int(stats["episodes_finished"])
        log_record: dict[str, float | int] = {
            "update": update,
            "total_env_steps": total_env_steps,
            "completed_episodes": completed_episodes,
            "episode_reward_mean": stats["episode_reward_mean"],
            "episode_reward_median": stats["episode_reward_median"],
            "episodes_finished": int(stats["episodes_finished"]),
            "win_rate": stats["win_rate"],
            "samples": int(stats["samples"]),
            "update_seconds": update_seconds,
            "elapsed_seconds": elapsed_seconds,
            "rollout_seconds": rollout_seconds,
            "ppo_seconds": ppo_seconds,
            "env_steps_per_sec": env_steps_per_sec,
            "rollout_env_steps_per_sec": rollout_env_steps_per_sec,
            "samples_per_sec": samples_per_sec,
            "ppo_samples_per_sec": ppo_samples_per_sec,
            "decisions_per_step": stats["decisions_per_step"],
            "moves_emitted_per_step": stats["moves_emitted_per_step"],
            "move_emit_rate": stats["move_emit_rate"],
            "candidate_valid_avg": stats["candidate_valid_avg"],
            "candidate_enemy_share": stats["candidate_enemy_share"],
            "candidate_neutral_share": stats["candidate_neutral_share"],
            "candidate_friendly_share": stats["candidate_friendly_share"],
            **metrics,
        }
        append_jsonl(log_path, log_record)
        if (
            isinstance(opponent, SelfPlayOpponent)
            and cfg.self_play_update_interval > 0
            and update % cfg.self_play_update_interval == 0
        ):
            opponent.sync_from(policy, normalizer)
        elif isinstance(opponent, SelfPlayOpponentPool):
            if (
                cfg.self_play_snapshot_interval > 0
                and update % cfg.self_play_snapshot_interval == 0
            ):
                opponent.add_snapshot(policy, normalizer, update=update)
            if (
                cfg.self_play_update_interval > 0
                and update % cfg.self_play_update_interval == 0
            ):
                opponent.sync_from(policy, normalizer, update=update)
        if update % cfg.log_every == 0:
            print(
                f"update={update} steps={total_env_steps} episodes={completed_episodes} "
                f"reward_mean={stats['episode_reward_mean']:.4f} "
                f"reward_median={stats['episode_reward_median']:.4f} "
                f"win_rate={stats['win_rate']:.3f} "
                f"loss={metrics['total_loss']:.4f} kl={metrics['approx_kl']:.5f} "
                f"sps={samples_per_sec:.1f} rollout_s={rollout_seconds:.3f} "
                f"ppo_s={ppo_seconds:.3f} "
                f"clip={metrics['clip_fraction']:.3f} "
                f"ev={metrics['explained_variance']:.3f} "
                f"decisions={stats['decisions_per_step']:.2f} "
                f"moves={stats['moves_emitted_per_step']:.2f} "
                f"candidates={stats['candidate_valid_avg']:.2f} "
                f"enemy={stats['candidate_enemy_share']:.2f} "
                f"neutral={stats['candidate_neutral_share']:.2f} "
                f"friendly={stats['candidate_friendly_share']:.2f}"
            )
        if update % cfg.checkpoint_every == 0 or update == cfg.ppo.total_updates:
            self_play_metadata = (
                opponent.metadata()
                if isinstance(opponent, SelfPlayOpponentPool)
                else None
            )
            save_checkpoint(
                save_dir,
                cfg.run_name,
                update,
                policy,
                optimizer,
                cfg,
                normalizer,
                self_play_metadata,
            )


if __name__ == "__main__":
    main()
