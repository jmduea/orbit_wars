from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp

import jax
from src.config import TrainConfig
from src.config.rollout_allocation import resolve_rollout_group_specs
from src.jax.env import JaxEnvState, assign_learner_players, batched_reset
from src.jax.features import TurnBatch
from src.jax.rollout.collect import collect_rollout_jax
from src.jax.rollout.types import JaxTransitionBatch
from src.jax.train.metrics import (
    finalize_cross_chunk_rate_metrics,
    merge_metric_dicts,
    prune_merged_rollout_metrics,
)


@dataclass(slots=True)
class JaxRolloutGroup:
    """State for one statically compiled JAX rollout format."""

    name: str
    cfg: TrainConfig
    env_state: JaxEnvState
    turn_batch: TurnBatch
    collect_fn: Callable


def _copy_config_for_rollout_group(
    cfg: TrainConfig, *, player_count: int, num_envs: int
) -> TrainConfig:
    """Return a rollout-specific config with static player/env counts."""

    group_cfg = deepcopy(cfg)
    group_cfg.task.player_count = int(player_count)
    group_cfg.training.num_envs = int(num_envs)
    return group_cfg


def configured_rollout_groups(cfg: TrainConfig) -> list[dict[str, int | str]]:
    """Resolve rollout group declarations for mixed-format training."""

    return [
        {
            "name": spec.name,
            "player_count": spec.player_count,
            "num_envs": spec.num_envs,
        }
        for spec in resolve_rollout_group_specs(cfg)
    ]


def _resolve_rollout_microbatch_envs(cfg: TrainConfig) -> int:
    env_count = int(cfg.training.num_envs)
    microbatch_envs = cfg.training.rollout_microbatch_envs
    if microbatch_envs is None:
        return env_count
    microbatch_envs = int(microbatch_envs)
    if microbatch_envs > env_count:
        raise ValueError(
            "training.rollout_microbatch_envs must be <= each rollout group's num_envs."
        )
    if env_count % microbatch_envs != 0:
        raise ValueError(
            "training.rollout_microbatch_envs must evenly divide each rollout group's num_envs."
        )
    return microbatch_envs


def _slice_env_axis(
    tree: object, *, start: int | jax.Array, size: int, env_count: int
) -> object:
    """Slice the leading env axis; ``start`` may be traced under ``lax.map``."""

    start_index = jnp.asarray(start, dtype=jnp.int32)

    def slice_leaf(value):
        if (
            isinstance(value, jax.Array)
            and value.ndim > 0
            and value.shape[0] == env_count
        ):
            return jax.lax.dynamic_slice_in_dim(value, start_index, size, axis=0)
        return value

    return jax.tree.map(slice_leaf, tree)


def _collect_rollout_microbatched(
    rollout_key: jax.Array,
    state: JaxEnvState,
    batch: TurnBatch,
    train_state: object,
    policy: object,
    cfg: TrainConfig,
    *,
    microbatch_envs: int,
    stage_view=None,
    historical_params_pool=None,
    update=jnp.asarray(0, dtype=jnp.int32),
    norm_state=None,
) -> tuple[jax.Array, JaxEnvState, TurnBatch, JaxTransitionBatch, dict[str, jax.Array]]:
    env_count = int(cfg.training.num_envs)
    micro = int(microbatch_envs)
    chunk_count = env_count // micro

    def collect_chunk(chunk_index: jax.Array):
        chunk_key = jax.random.fold_in(rollout_key, chunk_index * 9973 + 17)
        start = chunk_index * micro
        chunk_state = _slice_env_axis(
            state, start=start, size=micro, env_count=env_count
        )
        chunk_batch = _slice_env_axis(
            batch, start=start, size=micro, env_count=env_count
        )
        (
            _next_chunk_key,
            next_state,
            next_batch,
            chunk_transitions,
            chunk_metrics,
        ) = collect_rollout_jax(
            chunk_key,
            chunk_state,
            chunk_batch,
            train_state,
            policy,
            cfg,
            stage_view=stage_view,
            historical_params_pool=historical_params_pool,
            update=update,
            env_index_offset=start,
            norm_state=norm_state,
        )
        return next_state, next_batch, chunk_transitions, chunk_metrics

    chunk_indices = jnp.arange(chunk_count, dtype=jnp.int32)
    chunk_states, chunk_batches, chunk_transitions, chunk_metrics = jax.lax.map(
        collect_chunk, chunk_indices
    )

    def reshape_env_leading_chunk_axis(leaf):
        if isinstance(leaf, jax.Array) and leaf.ndim > 0:
            return leaf.reshape((env_count,) + leaf.shape[2:])
        return leaf

    def merge_transition_chunk_axis(leaf):
        if (
            isinstance(leaf, jax.Array)
            and leaf.ndim >= 3
            and leaf.shape[0] == chunk_count
            and leaf.shape[2] == micro
        ):
            rollout_steps_dim = int(leaf.shape[1])
            rest_shape = leaf.shape[3:]
            return leaf.transpose(1, 0, 2, *range(3, leaf.ndim)).reshape(
                (rollout_steps_dim, env_count, *rest_shape)
            )
        return reshape_env_leading_chunk_axis(leaf)

    merged_states = jax.tree.map(reshape_env_leading_chunk_axis, chunk_states)
    merged_batches = jax.tree.map(reshape_env_leading_chunk_axis, chunk_batches)
    merged_transitions = jax.tree.map(merge_transition_chunk_axis, chunk_transitions)

    def merge_metrics_scan(acc, chunk_index):
        chunk_metric = jax.tree.map(
            lambda x: jax.lax.dynamic_index_in_dim(
                x, chunk_index, axis=0, keepdims=False
            ),
            chunk_metrics,
        )
        return merge_metric_dicts([acc, chunk_metric]), None

    merged_metrics = jax.tree.map(
        lambda x: jax.lax.dynamic_index_in_dim(x, 0, axis=0, keepdims=False),
        chunk_metrics,
    )
    if chunk_count > 1:
        merged_metrics, _ = jax.lax.scan(
            merge_metrics_scan,
            merged_metrics,
            jnp.arange(1, chunk_count, dtype=jnp.int32),
        )
    merged_metrics = prune_merged_rollout_metrics(
        finalize_cross_chunk_rate_metrics(merged_metrics),
        cfg,
    )
    return (
        rollout_key,
        merged_states,
        merged_batches,
        merged_transitions,
        merged_metrics,
    )


def _init_rollout_group(
    key: jax.Array,
    cfg: TrainConfig,
    policy: object,
    *,
    name: str,
    player_count: int,
    num_envs: int,
) -> JaxRolloutGroup:
    """Initialize env state and a dedicated compiled collector for one format."""

    group_cfg = _copy_config_for_rollout_group(
        cfg, player_count=player_count, num_envs=num_envs
    )
    microbatch_envs = _resolve_rollout_microbatch_envs(group_cfg)
    reset_keys = jax.random.split(key, group_cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, group_cfg.task)
    env_indices = jnp.arange(group_cfg.training.num_envs, dtype=jnp.int32)
    episode_counts = jnp.zeros((group_cfg.training.num_envs,), dtype=jnp.int32)
    env_state, turn_batch = assign_learner_players(
        env_state,
        env_indices,
        episode_counts,
        group_cfg.task,
        group_cfg.opponents.mode.alternate_player_sides,
    )

    def collect_fn(
        rollout_key,
        state,
        batch,
        ts,
        stage_view=None,
        historical_params_pool=None,
        update_idx=jnp.asarray(0, dtype=jnp.int32),
        norm_state=None,
    ):
        if microbatch_envs >= group_cfg.training.num_envs:
            return collect_rollout_jax(
                rollout_key,
                state,
                batch,
                ts,
                policy,
                group_cfg,
                stage_view=stage_view,
                historical_params_pool=historical_params_pool,
                update=update_idx,
                norm_state=norm_state,
            )
        return _collect_rollout_microbatched(
            rollout_key,
            state,
            batch,
            ts,
            policy,
            group_cfg,
            microbatch_envs=microbatch_envs,
            stage_view=stage_view,
            historical_params_pool=historical_params_pool,
            update=update_idx,
            norm_state=norm_state,
        )

    collect_fn = jax.jit(collect_fn)
    return JaxRolloutGroup(
        name=name,
        cfg=group_cfg,
        env_state=env_state,
        turn_batch=turn_batch,
        collect_fn=collect_fn,
    )


def init_rollout_groups(
    key: jax.Array, cfg: TrainConfig, policy: object
) -> tuple[jax.Array, list[JaxRolloutGroup]]:
    """Create separate JAX rollout groups for all configured static formats."""

    specs = configured_rollout_groups(cfg)
    key, *group_keys = jax.random.split(key, len(specs) + 1)
    groups = [
        _init_rollout_group(
            group_key,
            cfg,
            policy,
            name=str(spec["name"]),
            player_count=int(spec["player_count"]),
            num_envs=int(spec["num_envs"]),
        )
        for group_key, spec in zip(group_keys, specs, strict=True)
    ]
    return key, groups


def replace_rollout_group_state(
    group: JaxRolloutGroup, env_state: JaxEnvState, turn_batch: TurnBatch
) -> JaxRolloutGroup:
    return JaxRolloutGroup(
        name=group.name,
        cfg=group.cfg,
        env_state=env_state,
        turn_batch=turn_batch,
        collect_fn=group.collect_fn,
    )


def active_group_indices(
    groups: list[JaxRolloutGroup],
    format_weights: dict[int, float],
    *,
    update: int = 1,
    rotate_format_rollouts: bool = False,
) -> list[int]:
    """Return rollout group indices to collect on this update.

    When ``rotate_format_rollouts`` is enabled and multiple formats have positive
    curriculum weight, only one group is selected per update using a fixed-period
    weighted schedule so long-run sample mix tracks ``format_weights``.
    """

    weighted_indices: list[tuple[int, float]] = []
    for idx, group in enumerate(groups):
        player_count = int(group.cfg.task.player_count)
        weight = float(format_weights.get(player_count, 0.0))
        if weight > 0.0:
            weighted_indices.append((idx, weight))
    if not weighted_indices:
        return list(range(len(groups)))
    if not rotate_format_rollouts or len(weighted_indices) == 1:
        return [idx for idx, _ in weighted_indices]

    total_weight = sum(weight for _, weight in weighted_indices)
    slot = float((int(update) - 1) % 100) / 100.0
    cumulative = 0.0
    for idx, weight in weighted_indices:
        cumulative += weight / total_weight
        if slot < cumulative:
            return [idx]
    return [weighted_indices[-1][0]]


def empty_per_format_rollout_stats() -> dict[int, dict[str, float]]:
    return {
        2: {"seconds": 0.0, "env_steps": 0.0, "samples": 0.0},
        4: {"seconds": 0.0, "env_steps": 0.0, "samples": 0.0},
    }
