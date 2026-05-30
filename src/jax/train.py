from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.artifacts.checkpoint_compat import (
    checkpoint_feature_metadata,
    feature_metadata,
    load_checkpoint_payload,
    validate_checkpoint_config_compatibility,
    validate_checkpoint_encoder_compatibility,
    validate_checkpoint_feature_compatibility,
    validate_checkpoint_pointer_decoder_compatibility,
)
from src.artifacts.checkpoint_retention import prune_checkpoints
from src.artifacts.promotion import promote_if_better
from src.artifacts.pipeline import (
    ArtifactPipelineError,
    AsyncArtifactPipeline,
    CheckpointJob,
    CheckpointResult,
    commit_checkpoint_payload,
    load_active_optional_jobs,
    protected_paths_from_jobs,
    write_optional_job,
)
from src.artifacts.replay import maybe_write_jax_checkpoint_replay
from src.artifacts.run_paths import resolve_run_paths, write_run_manifests
from src.config import TrainConfig
from src.telemetry import build_telemetry
from src.training.curriculum import CurriculumController
from src.training.seed_scheduler import SeedScheduleConfig, SeedScheduler

from .device import configure_jax_runtime_for_host, ensure_jax_accelerator_backend

configure_jax_runtime_for_host()
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

import jax.numpy as jnp  # noqa: E402

import jax  # noqa: E402

# jax.config.update("jax_debug_nans", True)
from src.jax.rollout.metrics import (  # noqa: E402
    BASE_ROLLOUT_SCALAR_KEYS as _BASE_ROLLOUT_SCALAR_KEYS,
)
from src.jax.rollout.metrics import (  # noqa: E402
    trajectory_shield_legal_rate,
)

from .env import JaxEnvState, assign_learner_players, batched_reset  # noqa: E402
from .features import TurnBatch  # noqa: E402
from .policy import build_jax_policy  # noqa: E402
from .ppo_update import concatenate_transition_batches, ppo_update_jax  # noqa: E402
from .rollout.collect import collect_rollout_jax  # noqa: E402
from .rollout.types import JaxTransitionBatch  # noqa: E402
from .normalization import (
    init_observation_norm_state,
    normalize_transition_batch,
    update_norm_state_from_transitions,
)  # noqa: E402
from .train_state import init_train_state, validate_policy_param_shapes  # noqa: E402


@dataclass(slots=True)
class JaxRolloutGroup:
    """State for one statically compiled JAX rollout format."""

    name: str
    cfg: TrainConfig
    env_state: JaxEnvState
    turn_batch: TurnBatch
    collect_fn: Callable


@dataclass(slots=True)
class HistoricalSnapshotPool:
    params: dict
    snapshot_ids: jax.Array
    snapshot_updates: jax.Array
    valid_mask: jax.Array
    next_slot: int = 0
    next_id: int = 1



def _copy_config_for_rollout_group(
    cfg: TrainConfig, *, player_count: int, num_envs: int
) -> TrainConfig:
    """Return a rollout-specific config with static player/env counts."""

    group_cfg = deepcopy(cfg)
    group_cfg.task.player_count = int(player_count)
    group_cfg.training.num_envs = int(num_envs)
    return group_cfg


def _configured_rollout_groups(cfg: TrainConfig) -> list[dict[str, int | str]]:
    """Resolve rollout group declarations for Option A mixed-format training.

    The JAX trainer keeps independent 2-player and 4-player environment states
    and compiles one collector per declared static format. If no groups are
    configured, it uses the single-format collector for the configured task.
    """

    raw_groups = cfg.format.rollout_groups
    groups: list[dict[str, int | str]] = []
    for index, group in enumerate(raw_groups):
        player_count = int(group.get("player_count", cfg.task.player_count))
        if player_count not in {2, 4}:
            raise ValueError(
                f"JAX rollout groups support player_count 2 or 4, got {player_count}."
            )
        num_envs = int(group.get("num_envs", cfg.training.num_envs))
        if num_envs <= 0:
            continue
        groups.append(
            {
                "name": str(group.get("name", f"{player_count}p_{index}")),
                "player_count": player_count,
                "num_envs": num_envs,
            }
        )
    if groups:
        return groups
    return [
        {
            "name": f"{cfg.task.player_count}p",
            "player_count": int(cfg.task.player_count),
            "num_envs": int(cfg.training.num_envs),
        }
    ]


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
        if isinstance(value, jax.Array) and value.ndim > 0 and value.shape[0] == env_count:
            return jax.lax.dynamic_slice_in_dim(value, start_index, size, axis=0)
        return value

    return jax.tree.map(slice_leaf, tree)


def _concat_env_axis(trees: list[object]) -> object:
    if len(trees) == 1:
        return trees[0]
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *trees)


def _finalize_cross_chunk_rate_metrics(
    metrics: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    """Derived rates that exist only after cross-chunk or cross-group aggregation."""

    metrics["win_rate_2p"] = jnp.where(
        metrics["episodes_2p"] > 0.0,
        metrics["wins_2p"] / metrics["episodes_2p"],
        0.0,
    )
    metrics["first_place_rate_4p"] = jnp.where(
        metrics["episodes_4p"] > 0.0,
        metrics["first_places_4p"] / metrics["episodes_4p"],
        0.0,
    )
    metrics["average_placement_4p"] = jnp.where(
        metrics["episodes_4p"] > 0.0,
        metrics["placement_4p_sum"] / metrics["episodes_4p"],
        0.0,
    )
    metrics["survival_time"] = jnp.where(
        metrics["episode_done"] > 0.0,
        metrics["survival_time_sum"] / metrics["episode_done"],
        0.0,
    )
    metrics["score_share"] = jnp.where(
        metrics["episode_done"] > 0.0,
        metrics["score_share_sum"] / metrics["episode_done"],
        0.0,
    )
    return metrics


def _merge_metric_dicts(
    metrics_by_chunk: list[dict[str, jax.Array]],
) -> dict[str, jax.Array]:
    """Sum per-chunk rollout metrics while preserving the per-chunk key set."""

    if len(metrics_by_chunk) == 1:
        return metrics_by_chunk[0]
    metrics = jax.tree.map(
        lambda *xs: jnp.stack(xs).sum(axis=0), *metrics_by_chunk
    )
    reward_weight = jnp.maximum(metrics["env_steps"], 1.0)
    metrics["average_reward"] = (
        jnp.stack(
            [chunk["average_reward"] * chunk["env_steps"] for chunk in metrics_by_chunk]
        ).sum()
        / reward_weight
    )
    metrics["episode_reward_mean"] = jnp.where(
        metrics["episode_done"] > 0.0,
        jnp.stack(
            [
                chunk["episode_reward_mean"] * chunk["episode_done"]
                for chunk in metrics_by_chunk
            ]
        ).sum()
        / metrics["episode_done"],
        0.0,
    )
    metrics["valid_non_noop_targets_per_row"] = jnp.where(
        metrics["valid_non_noop_target_rows"] > 0.0,
        metrics["valid_non_noop_targets_sum"] / metrics["valid_non_noop_target_rows"],
        0.0,
    )
    metrics["only_noop_fraction"] = jnp.where(
        metrics["valid_non_noop_target_rows"] > 0.0,
        metrics["only_noop_rows"] / metrics["valid_non_noop_target_rows"],
        0.0,
    )
    metrics["trajectory_shield_legal_non_noop_rate"] = trajectory_shield_legal_rate(
        legal=metrics["trajectory_shield_legal_non_noop_count"],
        original=metrics["trajectory_shield_original_non_noop_count"],
    )
    metrics["overall_win_rate"] = jnp.where(
        metrics["episode_done"] > 0.0,
        (metrics["wins_2p"] + metrics["first_places_4p"]) / metrics["episode_done"],
        0.0,
    )
    metrics["noop_percent"] = jnp.where(
        metrics["decision_count"] > 0.0,
        (metrics["noop_count"] / metrics["decision_count"]) * 100.0,
        0.0,
    )
    metrics["friendly_target_percent"] = jnp.where(
        metrics["decision_count"] > 0.0,
        (metrics["friendly_target_count"] / metrics["decision_count"]) * 100.0,
        0.0,
    )
    metrics["enemy_target_percent"] = jnp.where(
        metrics["decision_count"] > 0.0,
        (metrics["enemy_target_count"] / metrics["decision_count"]) * 100.0,
        0.0,
    )
    metrics["neutral_target_percent"] = jnp.where(
        metrics["decision_count"] > 0.0,
        (metrics["neutral_target_count"] / metrics["decision_count"]) * 100.0,
        0.0,
    )
    metrics["won_non_noop_actions_per_step"] = jnp.where(
        metrics["win_episode_rows"] > 0.0,
        metrics["non_noop_count"] / metrics["win_episode_rows"],
        0.0,
    )
    metrics["lost_non_noop_actions_per_step"] = jnp.where(
        metrics["loss_episode_rows"] > 0.0,
        metrics["non_noop_count"] / metrics["loss_episode_rows"],
        0.0,
    )
    metrics["won_avg_fleet_launch_size"] = jnp.where(
        metrics["win_episode_rows"] > 0.0,
        metrics["launched_ship_total"] / jnp.maximum(metrics["launched_ship_count"], 1.0),
        0.0,
    )
    metrics["lost_avg_fleet_launch_size"] = jnp.where(
        metrics["loss_episode_rows"] > 0.0,
        metrics["launched_ship_total"] / jnp.maximum(metrics["launched_ship_count"], 1.0),
        0.0,
    )
    metrics["won_avg_planets_owned"] = jnp.where(
        metrics["win_episode_rows"] > 0.0,
        metrics["won_planets_owned_total"] / metrics["win_episode_rows"],
        0.0,
    )
    metrics["lost_avg_planets_owned"] = jnp.where(
        metrics["loss_episode_rows"] > 0.0,
        metrics["lost_planets_owned_total"] / metrics["loss_episode_rows"],
        0.0,
    )
    metrics["won_avg_planets_lost"] = jnp.where(
        metrics["win_episode_rows"] > 0.0,
        metrics["won_planets_lost_total"] / metrics["win_episode_rows"],
        0.0,
    )
    metrics["lost_avg_planets_lost"] = jnp.where(
        metrics["loss_episode_rows"] > 0.0,
        metrics["lost_planets_lost_total"] / metrics["loss_episode_rows"],
        0.0,
    )
    metrics["won_avg_planets_taken"] = jnp.where(
        metrics["win_episode_rows"] > 0.0,
        metrics["won_planets_taken_total"] / metrics["win_episode_rows"],
        0.0,
    )
    metrics["lost_avg_planets_taken"] = jnp.where(
        metrics["loss_episode_rows"] > 0.0,
        metrics["lost_planets_taken_total"] / metrics["loss_episode_rows"],
        0.0,
    )
    metrics["won_avg_garrisoned_ships_per_planet"] = jnp.where(
        metrics["win_episode_rows"] > 0.0,
        metrics["won_garrisoned_ships_per_planet_total"] / metrics["win_episode_rows"],
        0.0,
    )
    metrics["lost_avg_garrisoned_ships_per_planet"] = jnp.where(
        metrics["loss_episode_rows"] > 0.0,
        metrics["lost_garrisoned_ships_per_planet_total"] / metrics["loss_episode_rows"],
        0.0,
    )
    metrics["won_avg_planet_diff"] = jnp.where(
        metrics["win_episode_rows"] > 0.0,
        metrics["won_planet_diff_total"] / metrics["win_episode_rows"],
        0.0,
    )
    metrics["lost_avg_planet_diff"] = jnp.where(
        metrics["loss_episode_rows"] > 0.0,
        metrics["lost_planet_diff_total"] / metrics["loss_episode_rows"],
        0.0,
    )
    metrics["won_avg_production_diff"] = jnp.where(
        metrics["win_episode_rows"] > 0.0,
        metrics["won_production_diff_total"] / metrics["win_episode_rows"],
        0.0,
    )
    metrics["lost_avg_production_diff"] = jnp.where(
        metrics["loss_episode_rows"] > 0.0,
        metrics["lost_production_diff_total"] / metrics["loss_episode_rows"],
        0.0,
    )
    metrics["won_avg_launch_fleet_speed"] = jnp.where(
        metrics["win_episode_rows"] > 0.0,
        metrics["launched_ship_speed_total"]
        / jnp.maximum(metrics["launched_ship_count"], 1.0),
        0.0,
    )
    metrics["lost_avg_launch_fleet_speed"] = jnp.where(
        metrics["loss_episode_rows"] > 0.0,
        metrics["launched_ship_speed_total"]
        / jnp.maximum(metrics["launched_ship_count"], 1.0),
        0.0,
    )
    return metrics


def _sum_metric_dicts(metrics_by_chunk: list[dict[str, jax.Array]]) -> dict[str, jax.Array]:
    if len(metrics_by_chunk) == 1:
        return metrics_by_chunk[0]
    return _finalize_cross_chunk_rate_metrics(_merge_metric_dicts(metrics_by_chunk))


def _empty_per_format_rollout_stats() -> dict[int, dict[str, float]]:
    return {
        2: {"seconds": 0.0, "env_steps": 0.0, "samples": 0.0},
        4: {"seconds": 0.0, "env_steps": 0.0, "samples": 0.0},
    }


def _build_per_format_timing_metrics(
    format_stats: dict[int, dict[str, float]],
    *,
    update_seconds: float,
    rollout_seconds: float,
    ppo_seconds: float,
) -> dict[str, float]:
    metrics = {
        "update_time_rollout_fraction": rollout_seconds / max(update_seconds, 1e-9),
        "update_time_ppo_fraction": ppo_seconds / max(update_seconds, 1e-9),
    }
    for player_count, suffix in ((2, "2p"), (4, "4p")):
        stats = format_stats.get(player_count, {})
        seconds = float(stats.get("seconds", 0.0))
        env_steps = float(stats.get("env_steps", 0.0))
        samples = float(stats.get("samples", 0.0))
        metrics[f"rollout_seconds_{suffix}"] = seconds
        metrics[f"env_steps_per_sec_{suffix}"] = env_steps / max(update_seconds, 1e-9)
        metrics[f"rollout_env_steps_per_sec_{suffix}"] = env_steps / max(seconds, 1e-9)
        metrics[f"samples_per_sec_{suffix}"] = samples / max(update_seconds, 1e-9)
        metrics[f"rollout_samples_per_sec_{suffix}"] = samples / max(seconds, 1e-9)
    return metrics


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
        return _merge_metric_dicts([acc, chunk_metric]), None

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
    merged_metrics = _finalize_cross_chunk_rate_metrics(merged_metrics)
    return (
        rollout_key,
        merged_states,
        merged_batches,
        merged_transitions,
        merged_metrics,
    )


def init_rollout_groups(
    key: jax.Array, cfg: TrainConfig, policy: object
) -> tuple[jax.Array, list[JaxRolloutGroup]]:
    """Create separate JAX rollout groups for all configured static formats."""

    specs = _configured_rollout_groups(cfg)
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


def _replace_rollout_group_state(
    group: JaxRolloutGroup, env_state: JaxEnvState, turn_batch: TurnBatch
) -> JaxRolloutGroup:
    return JaxRolloutGroup(
        name=group.name,
        cfg=group.cfg,
        env_state=env_state,
        turn_batch=turn_batch,
        collect_fn=group.collect_fn,
    )


def _checkpoint_payload_builder(
    train_state: object,
    cfg: TrainConfig,
    *,
    key: jax.Array,
    update: int,
    total_env_steps: int,
    completed_episodes: int,
    curriculum: CurriculumController | None = None,
    historical_pool: HistoricalSnapshotPool | None = None,
) -> Callable[[], dict[str, object]]:
    params = train_state.params
    opt_state = train_state.opt_state
    rng_key = key
    cfg_snapshot = deepcopy(cfg)
    metadata = feature_metadata(
        cfg_snapshot.task,
        model_cfg=cfg_snapshot.model,
    )
    curriculum_state_snapshot = (
        deepcopy(curriculum.state_dict()) if curriculum is not None else None
    )
    historical_pool_snapshot = None
    if historical_pool is not None:
        historical_pool_snapshot = {
            "params": jax.device_get(historical_pool.params),
            "snapshot_ids": jax.device_get(historical_pool.snapshot_ids),
            "snapshot_updates": jax.device_get(historical_pool.snapshot_updates),
            "valid_mask": jax.device_get(historical_pool.valid_mask),
            "next_slot": historical_pool.next_slot,
            "next_id": historical_pool.next_id,
        }

    def build_payload() -> dict[str, object]:
        payload: dict[str, object] = {
            "update": update,
            "params": jax.device_get(params),
            "opt_state": jax.device_get(opt_state),
            "rng_key": jax.device_get(rng_key),
            "config": cfg_snapshot,
            "feature_metadata": metadata,
            "total_env_steps": total_env_steps,
            "completed_episodes": completed_episodes,
        }
        if curriculum_state_snapshot is not None:
            payload["curriculum_state"] = deepcopy(curriculum_state_snapshot)
        if historical_pool_snapshot is not None:
            payload["historical_snapshot_pool"] = historical_pool_snapshot
        return payload

    return build_payload


def _checkpoint_replay_due(cfg: TrainConfig, update: int) -> bool:
    if not cfg.artifacts.replay.enabled:
        return False
    every_n = max(int(cfg.artifacts.replay.every_n_checkpoints), 1)
    checkpoint_index = max(update // max(int(cfg.artifacts.checkpoint_every), 1), 1)
    return checkpoint_index % every_n == 0 or update == cfg.training.total_updates


def _queue_optional_jobs_if_due(
    cfg: TrainConfig,
    *,
    update: int,
    checkpoint_path: Path,
    log_path: Path,
    queue_dir: Path,
    result_root: Path | None = None,
    queue_replay: bool,
    queue_docker_validation: bool,
) -> list[Path]:
    job_paths: list[Path] = []
    if queue_replay and _checkpoint_replay_due(cfg, update):
        job_paths.append(
            write_optional_job(
                queue_dir,
                kind="replay",
                update=update,
                checkpoint_path=checkpoint_path,
                payload={
                    "backend": cfg.artifacts.artifact_pipeline.replay_backend,
                    "log_path": str(log_path),
                    "replay_output_dir": cfg.artifacts.replay.output_dir,
                    "docker_image": cfg.artifacts.artifact_pipeline.docker_image,
                    "player_count": cfg.artifacts.artifact_pipeline.docker_player_count,
                    "timeout_seconds": cfg.artifacts.artifact_pipeline.docker_timeout_seconds,
                    "episode_steps": cfg.artifacts.replay.max_steps,
                    "seed": cfg.seed + update,
                },
                result_root=result_root,
            )
        )
    if queue_docker_validation:
        job_paths.append(
            write_optional_job(
                queue_dir,
                kind="docker_validation",
                update=update,
                checkpoint_path=checkpoint_path,
                payload={
                    "docker_image": cfg.artifacts.artifact_pipeline.docker_image,
                    "player_count": cfg.artifacts.artifact_pipeline.docker_player_count,
                    "timeout_seconds": cfg.artifacts.artifact_pipeline.docker_timeout_seconds,
                    "episode_steps": cfg.artifacts.replay.max_steps,
                    "seed": cfg.seed + update,
                },
                result_root=result_root,
            )
        )
    return job_paths


def _start_artifact_worker_if_needed(
    cfg: TrainConfig,
    *,
    queue_dir: Path,
    result_root: Path | None = None,
    worker_state: dict[str, subprocess.Popen[object]],
) -> None:
    if not cfg.artifacts.artifact_pipeline.worker_autostart:
        return
    worker = worker_state.get("process")
    if worker is not None and worker.poll() is None:
        return
    queue_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = queue_dir / "worker.stdout.log"
    stderr_path = queue_dir / "worker.stderr.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "scripts" / "run_artifact_worker.py"),
        str(queue_dir),
        "--poll-seconds",
        str(cfg.artifacts.artifact_pipeline.worker_poll_seconds),
        "--idle-exit-seconds",
        str(cfg.artifacts.artifact_pipeline.worker_idle_exit_seconds),
    ]
    if result_root is not None:
        command.extend(["--result-root", str(result_root)])
    from src.artifacts.worker_env import artifact_worker_subprocess_env

    stdout = stdout_path.open("a", encoding="utf-8")
    stderr = stderr_path.open("a", encoding="utf-8")
    worker_state["process"] = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[1],
        stdout=stdout,
        stderr=stderr,
        env=artifact_worker_subprocess_env(),
        start_new_session=True,
    )


def _active_group_indices(
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


def _init_historical_snapshot_pool(
    params: dict, pool_size: int
) -> HistoricalSnapshotPool:
    capacity = max(int(pool_size), 1)
    stacked_params = jax.tree.map(
        lambda value: jnp.broadcast_to(
            jnp.asarray(value)[None, ...], (capacity,) + jnp.asarray(value).shape
        ),
        params,
    )
    return HistoricalSnapshotPool(
        params=stacked_params,
        snapshot_ids=jnp.zeros((capacity,), dtype=jnp.int32),
        snapshot_updates=jnp.zeros((capacity,), dtype=jnp.int32),
        valid_mask=jnp.zeros((capacity,), dtype=bool),
    )


def _add_historical_snapshot(
    pool: HistoricalSnapshotPool, params: dict, *, update: int
) -> tuple[HistoricalSnapshotPool, dict[str, object]]:
    slot = int(pool.next_slot)
    snapshot_id = int(pool.next_id)
    new_params = jax.tree.map(
        lambda bank, value: bank.at[slot].set(value), pool.params, params
    )
    was_valid = bool(jax.device_get(pool.valid_mask[slot]))
    next_pool = HistoricalSnapshotPool(
        params=new_params,
        snapshot_ids=pool.snapshot_ids.at[slot].set(snapshot_id),
        snapshot_updates=pool.snapshot_updates.at[slot].set(int(update)),
        valid_mask=pool.valid_mask.at[slot].set(True),
        next_slot=(slot + 1) % int(pool.valid_mask.shape[0]),
        next_id=snapshot_id + 1,
    )
    event = {
        "event": "historical_snapshot_added",
        "update": int(update),
        "snapshot_id": snapshot_id,
        "snapshot_slot": slot,
        "historical_snapshot_evicted": was_valid,
    }
    return next_pool, event


def _restore_historical_snapshot_pool(
    payload: object, fallback: HistoricalSnapshotPool
) -> HistoricalSnapshotPool:
    if not isinstance(payload, dict):
        return fallback
    try:
        return HistoricalSnapshotPool(
            params=jax.device_put(payload["params"]),
            snapshot_ids=jax.device_put(payload["snapshot_ids"]),
            snapshot_updates=jax.device_put(payload["snapshot_updates"]),
            valid_mask=jax.device_put(payload["valid_mask"]),
            next_slot=int(payload.get("next_slot", fallback.next_slot)),
            next_id=int(payload.get("next_id", fallback.next_id)),
        )
    except KeyError:
        return fallback


def _restore_curriculum_artifacts(
    checkpoint_path: str,
    curriculum: CurriculumController,
    historical_pool: HistoricalSnapshotPool,
) -> HistoricalSnapshotPool:
    checkpoint = load_checkpoint_payload(checkpoint_path)
    validate_checkpoint_config_compatibility(
        checkpoint, checkpoint_path=checkpoint_path
    )
    if not isinstance(checkpoint, dict):
        return historical_pool
    state = checkpoint.get("curriculum_state")
    if isinstance(state, dict):
        curriculum.load_state_dict(state)
    return _restore_historical_snapshot_pool(
        checkpoint.get("historical_snapshot_pool"), historical_pool
    )


def _snapshot_due(cfg: TrainConfig, update: int) -> bool:
    if not cfg.curriculum.enabled:
        return False
    interval = int(cfg.opponents.snapshot.interval_updates)
    return interval > 0 and update % interval == 0


def run_jax_training(cfg: TrainConfig, resume_checkpoint: str | None = None) -> Path:
    """Run an end-to-end JAX training loop for the JAX environment backend.

    This path keeps environment state, feature encoding, action sampling, rollout
    storage, return/advantage computation, and PPO updates in JAX. Mixed 2p/4p
    training uses Option A: each format owns its env state and jitted collector,
    then compatible transition batches are concatenated before PPO updates.

    Returns:
        Path to the JSONL metrics log for this run.
    """

    ensure_jax_accelerator_backend()

    key = jax.random.PRNGKey(cfg.seed)
    _, rollout_init_key, policy_key = jax.random.split(key, 3)
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(policy_key, policy, cfg)
    key, rollout_groups = init_rollout_groups(rollout_init_key, cfg, policy)
    total_env_steps = 0
    completed_episodes = 0
    start_update = 1
    if resume_checkpoint is not None:
        train_state, key, start_update, total_env_steps, completed_episodes = (
            load_jax_checkpoint(resume_checkpoint, train_state, cfg)
        )
        validate_policy_param_shapes(train_state.params, cfg.task)
        print(
            f"Resuming JAX training from {resume_checkpoint} at update {start_update}"
        )
    norm_state = (
        init_observation_norm_state(rollout_groups[0].turn_batch)
        if cfg.model.normalize_observations
        else None
    )
    update_fn = jax.jit(
        lambda ts, transitions: ppo_update_jax(ts, policy, transitions, cfg)
    )
    cfg, run_context = resolve_run_paths(cfg)
    run_dir = run_context.checkpoints_dir
    log_path = run_context.log_path
    write_run_manifests(
        cfg,
        run_context,
        {
            "backend": "jax",
            "job_type": "train",
            "wandb_dir": str(run_context.wandb_dir),
            "wandb_artifact_dir": str(run_context.wandb_artifact_dir),
            "wandb_data_dir": str(run_context.wandb_data_dir),
        },
    )
    telemetry = build_telemetry(
        cfg,
        {
            "backend": "jax",
            "seed": cfg.seed,
            "job_type": "train",
            "campaign": run_context.campaign_slug,
            "run_id": run_context.run_id,
            "model_compatibility_family": run_context.model_compatibility_family,
            "retention_class": run_context.retention_class,
            "wandb_dir": str(run_context.wandb_dir),
            "wandb_artifact_dir": str(run_context.wandb_artifact_dir),
            "wandb_data_dir": str(run_context.wandb_data_dir),
        },
    )
    seed_scheduler = SeedScheduler(
        base_seed=cfg.seed,
        cfg=SeedScheduleConfig(
            reseed_every_updates=cfg.training.reseed_every_updates,
            reseed_on_plateau=cfg.training.reseed_on_plateau,
            plateau_metric=cfg.training.plateau_metric,
            plateau_window=cfg.training.plateau_window,
            plateau_delta=cfg.training.plateau_delta,
            heldout_eval_seed_set=cfg.heldout_eval_seed_set,
        ),
    )
    curriculum = CurriculumController(cfg.curriculum, cfg.opponents.snapshot)
    historical_pool = _init_historical_snapshot_pool(
        train_state.params, cfg.opponents.snapshot.pool_size
    )
    if resume_checkpoint is not None:
        historical_pool = _restore_curriculum_artifacts(
            resume_checkpoint, curriculum, historical_pool
        )
    phase_events: list[dict[str, object]] = []
    train_start_time = time.perf_counter()
    artifact_cfg = cfg.artifacts.artifact_pipeline
    artifact_queue_dir = run_context.queue_dir
    checkpoint_pipeline = (
        AsyncArtifactPipeline(
            checkpoint_queue_size=artifact_cfg.checkpoint_queue_size,
            coalesce_intermediate_checkpoints=artifact_cfg.coalesce_intermediate_checkpoints,
            ledger_path=(run_context.logs_dir / "artifact_pipeline.jsonl")
            if artifact_cfg.ledger_enabled
            else None,
        )
        if artifact_cfg.enabled
        else None
    )
    checkpoint_failures: list[CheckpointResult] = []
    artifact_worker_state: dict[str, subprocess.Popen[object]] = {}
    run_promotion_best: float | None = None

    def protected_artifact_paths() -> set[Path]:
        paths = {run_dir / "jax_ckpt_last.pkl"}
        if checkpoint_pipeline is not None:
            paths.update(checkpoint_pipeline.protected_paths())
        paths.update(protected_paths_from_jobs(load_active_optional_jobs(artifact_queue_dir)))
        return paths

    def handle_checkpoint_results(results: list[CheckpointResult]) -> None:
        nonlocal run_promotion_best
        for result in results:
            event_record = {
                "event": "checkpoint_result",
                "update": result.update,
                "checkpoint_status": result.status,
                "checkpoint_final": result.final,
                "checkpoint_reason": result.reason,
                "checkpoint_error": result.error,
            }
            append_jsonl(log_path, event_record)
            telemetry.log(event_record, step=result.update)
            if result.status == "failed":
                checkpoint_failures.append(result)
                continue
            if not result.committed or result.numbered_path is None:
                continue

            protected_paths = protected_artifact_paths()
            protected_paths.add(result.numbered_path)
            if result.latest_path is not None:
                protected_paths.add(result.latest_path)
            if result.final:
                protected_paths.add(result.numbered_path)
            if artifact_cfg.replay_async or artifact_cfg.docker_validation_async:
                job_paths = _queue_optional_jobs_if_due(
                    cfg,
                    update=result.update,
                    checkpoint_path=result.numbered_path,
                    log_path=log_path,
                    queue_dir=artifact_queue_dir,
                    result_root=run_context.evaluations_dir,
                    queue_replay=artifact_cfg.replay_async,
                    queue_docker_validation=artifact_cfg.docker_validation_async,
                )
                if job_paths:
                    _start_artifact_worker_if_needed(
                        cfg,
                        queue_dir=artifact_queue_dir,
                        result_root=run_context.evaluations_dir,
                        worker_state=artifact_worker_state,
                    )
                protected_paths.update(
                    protected_paths_from_jobs(load_active_optional_jobs(artifact_queue_dir))
                )

            retention = cfg.artifacts.checkpoint_retention
            pruning = prune_checkpoints(
                run_dir,
                log_path=log_path,
                keep_last_n=retention.keep_last_n,
                keep_every_n_updates=retention.keep_every_n_updates,
                keep_best_k_by_metric=retention.keep_best_k_by_metric,
                best_metric_name=retention.best_metric_name,
                best_metric_mode=retention.best_metric_mode,
                min_update_for_pruning=retention.min_update_for_pruning,
                dry_run_pruning=retention.dry_run_pruning,
                protected_paths=protected_paths,
            )
            action_label = "would prune" if pruning.dry_run else "pruned"
            print(
                f"checkpoint retention: {action_label} {len(pruning.deleted)} files, "
                f"reclaimed_bytes={pruning.reclaimed_bytes}"
            )
            promotion_attempt, run_promotion_best = promote_if_better(
                cfg,
                run_context,
                checkpoint_path=result.numbered_path,
                update=result.update,
                log_path=log_path,
                run_best_value=run_promotion_best,
            )
            if promotion_attempt.promoted:
                append_jsonl(
                    log_path,
                    {
                        "event": "checkpoint_promoted",
                        "update": result.update,
                        "metric_name": promotion_attempt.metric_name,
                        "metric_value": promotion_attempt.metric_value,
                        "promoted_manifest_path": str(
                            promotion_attempt.promoted_manifest_path
                        ),
                    },
                )
                if cfg.telemetry.wandb.log_artifacts:
                    telemetry.log_promoted_checkpoint(
                        result.numbered_path,
                        update=result.update,
                        metric_name=promotion_attempt.metric_name,
                        metric_value=float(promotion_attempt.metric_value or 0.0),
                    )
            if not artifact_cfg.replay_async:
                replay_meta_path = maybe_write_jax_checkpoint_replay(
                    cfg,
                    update=result.update,
                    checkpoint_path=result.numbered_path,
                    log_path=log_path,
                )
                if replay_meta_path is not None:
                    telemetry.log_artifact(
                        replay_meta_path,
                        name=f"replay-meta-u{result.update}",
                        artifact_type="replay_metadata",
                    )

    completed_training = False
    close_error: Exception | None = None
    try:
        for update in range(start_update, cfg.training.total_updates + 1):
            if checkpoint_pipeline is not None:
                handle_checkpoint_results(checkpoint_pipeline.drain_results())
            update_start = time.perf_counter()
            reseed_events: list[dict[str, object]] = []
            rollout_start = time.perf_counter()
            transitions_by_group: list[JaxTransitionBatch] = []
            rollout_metrics_by_group: list[dict[str, jax.Array]] = []
            format_rollout_stats = _empty_per_format_rollout_stats()
            next_groups: list[JaxRolloutGroup] = []
            should_reseed, reseed_reason = seed_scheduler.should_reseed(update)
            if should_reseed:
                reseed_event = seed_scheduler.reseed(update, reseed_reason)
                key = jax.random.PRNGKey(reseed_event.new_seed)
                reseed_events.append(
                    {
                        "update": reseed_event.update,
                        "old_seed": reseed_event.old_seed,
                        "new_seed": reseed_event.new_seed,
                        "reason": reseed_event.reason,
                        "policy": reseed_event.policy,
                    }
                )
            stage_view = curriculum.stage_view(
                update,
                snapshot_ids=historical_pool.snapshot_ids,
                snapshot_valid_mask=historical_pool.valid_mask,
                snapshot_updates=historical_pool.snapshot_updates,
            )
            active_indices = _active_group_indices(
                rollout_groups,
                curriculum.current_format_weights(),
                update=update,
                rotate_format_rollouts=cfg.training.rotate_format_rollouts,
            )
            key, *rollout_keys = jax.random.split(key, len(active_indices) + 1)
            for group_idx, rollout_key in zip(active_indices, rollout_keys, strict=True):
                group = rollout_groups[group_idx]
                group_rollout_start = time.perf_counter()
                (
                    _next_rollout_key,
                    env_state,
                    turn_batch,
                    transitions,
                    rollout_metrics,
                ) = group.collect_fn(
                    rollout_key,
                    group.env_state,
                    group.turn_batch,
                    train_state,
                    stage_view,
                    historical_pool.params,
                    jnp.asarray(update, dtype=jnp.int32),
                    norm_state,
                )
                group_env_steps, group_samples = jax.device_get(
                    jnp.asarray(
                        [rollout_metrics["env_steps"], rollout_metrics["samples"]],
                        dtype=jnp.float32,
                    )
                ).tolist()
                group_seconds = time.perf_counter() - group_rollout_start
                stats = format_rollout_stats[int(group.cfg.task.player_count)]
                stats["seconds"] += group_seconds
                stats["env_steps"] += float(group_env_steps)
                stats["samples"] += float(group_samples)
                next_groups.append(
                    _replace_rollout_group_state(group, env_state, turn_batch)
                )
                transitions_by_group.append(transitions)
                rollout_metrics_by_group.append(rollout_metrics)
            merged_groups = list(rollout_groups)
            for group_idx, updated_group in zip(active_indices, next_groups, strict=True):
                merged_groups[group_idx] = updated_group
            rollout_groups = merged_groups
            transitions = concatenate_transition_batches(transitions_by_group)
            if norm_state is not None and cfg.model.normalize_observations:
                ppo_transitions = normalize_transition_batch(
                    transitions, norm_state, cfg.model
                )
            else:
                ppo_transitions = transitions
            rollout_metrics = _sum_metric_dicts(rollout_metrics_by_group)
            rollout_scalar_keys = (
                *_BASE_ROLLOUT_SCALAR_KEYS,
                cfg.training.plateau_metric,
            )
            rollout_scalar_values = jnp.asarray(
                [rollout_metrics.get(key, 0.0) for key in rollout_scalar_keys],
                dtype=jnp.float32,
            )
            # Intentional sync boundary: transfer only compact rollout scalars once so
            # rollout timing reflects completed device work without materializing trees.
            rollout_scalars_host = jax.device_get(rollout_scalar_values)
            rollout_scalars = dict(
                zip(rollout_scalar_keys, rollout_scalars_host.tolist(), strict=True)
            )
            rollout_samples = float(rollout_scalars["samples"])
            rollout_seconds = time.perf_counter() - rollout_start
    
            ppo_start = time.perf_counter()
            metrics_accum: dict[str, jax.Array] | None = None
            for _ in range(cfg.training.epochs):
                train_state, update_metrics = update_fn(train_state, ppo_transitions)
                metrics_accum = (
                    update_metrics
                    if metrics_accum is None
                    else jax.tree.map(jnp.add, metrics_accum, update_metrics)
                )
            assert metrics_accum is not None
            if norm_state is not None and cfg.model.normalize_observations:
                norm_state = update_norm_state_from_transitions(norm_state, transitions)
            metrics = jax.tree.map(
                lambda x: x / float(max(cfg.training.epochs, 1)), metrics_accum
            )
            metric_names = tuple(metrics.keys())
            metric_values = jnp.asarray([metrics[name] for name in metric_names])
            # Intentional sync boundary: perform a single compact host transfer for
            # PPO scalars and keep logging values identical.
            metric_values_host = jax.device_get(metric_values)
            metrics_host = dict(zip(metric_names, metric_values_host.tolist(), strict=True))
            ppo_seconds = time.perf_counter() - ppo_start
            update_seconds = time.perf_counter() - update_start
            per_format_timing_metrics = _build_per_format_timing_metrics(
                format_rollout_stats,
                update_seconds=update_seconds,
                rollout_seconds=rollout_seconds,
                ppo_seconds=ppo_seconds,
            )
            env_steps = int(rollout_scalars["env_steps"])
            episodes = int(rollout_scalars["episode_done"])
            episodes_2p = float(rollout_scalars["episodes_2p"])
            episodes_4p = float(rollout_scalars["episodes_4p"])
            episode_count = float(rollout_scalars["episode_done"])
            win_rate_2p = (
                float(rollout_scalars["wins_2p"]) / episodes_2p
                if episodes_2p
                else 0.0
            )
            first_place_rate_4p = (
                float(rollout_scalars["first_places_4p"]) / episodes_4p
                if episodes_4p
                else 0.0
            )
            average_placement_4p = (
                float(rollout_scalars["placement_4p_sum"]) / episodes_4p
                if episodes_4p
                else 0.0
            )
            survival_time = (
                float(rollout_scalars["survival_time_sum"]) / episode_count
                if episode_count
                else 0.0
            )
            score_share = (
                float(rollout_scalars["score_share_sum"]) / episode_count
                if episode_count
                else 0.0
            )
            average_reward = float(rollout_scalars["average_reward"])
            average_episode_reward = float(rollout_scalars["episode_reward_mean"])
            overall_win_rate = (
                (float(rollout_scalars["wins_2p"]) + float(rollout_scalars["first_places_4p"]))
                / episode_count
                if episode_count
                else 0.0
            )
            # decision_count = float(rollout_scalars["decision_count"])
            # noop_percent = (
            #     (float(rollout_scalars["noop_count"]) / decision_count) * 100.0
            #     if decision_count
            #     else 0.0
            # )
            # friendly_target_percent = (
            #     (float(rollout_scalars["friendly_target_count"]) / decision_count) * 100.0
            #     if decision_count
            #     else 0.0
            # )
            # enemy_target_percent = (
            #     (float(rollout_scalars["enemy_target_count"]) / decision_count) * 100.0
            #     if decision_count
            #     else 0.0
            # )
            # neutral_target_percent = (
            #     (float(rollout_scalars["neutral_target_count"]) / decision_count) * 100.0
            #     if decision_count
            #     else 0.0
            # )
            total_env_steps += env_steps
            completed_episodes += episodes
            seed_scheduler.update_metric(float(rollout_scalars[cfg.training.plateau_metric]))
            curriculum_telemetry = curriculum.stage_telemetry(stage_view, update)
            update_events = list(phase_events)
            transition = curriculum.update(
                update,
                {
                    "overall_win_rate": overall_win_rate,
                    "win_rate_2p": win_rate_2p,
                    "first_place_rate_4p": first_place_rate_4p,
                    "average_reward": average_reward,
                    "average_episode_reward": average_episode_reward,
                    "episode_reward_mean": average_episode_reward,
                    "survival_time": survival_time,
                    "score_share": score_share,
                    "approx_kl": float(metrics_host.get("approx_kl", 0.0)),
                },
            )
            if transition is not None:
                update_events.append(transition)
            if _snapshot_due(cfg, update):
                historical_pool, snapshot_event = _add_historical_snapshot(
                    historical_pool, train_state.params, update=update
                )
                update_events.append(snapshot_event)
            phase_events = []
            historical_ids = jax.device_get(historical_pool.snapshot_ids).tolist()
            historical_ages = jax.device_get(
                jnp.where(
                    historical_pool.valid_mask,
                    jnp.asarray(update, dtype=jnp.int32)
                    - historical_pool.snapshot_updates,
                    0,
                )
            ).tolist()
            record: dict[str, object] = {
                "update": update,
                "total_env_steps": total_env_steps,
                "completed_episodes": completed_episodes,
                "samples": int(rollout_samples),
                "win_rate_2p": win_rate_2p,
                "first_place_rate_4p": first_place_rate_4p,
                "average_placement_4p": average_placement_4p,
                "overall_win_rate": overall_win_rate,
                "average_reward": average_reward,
                "average_episode_reward": average_episode_reward,
                # "noop_percent": noop_percent,
                # "friendly_target_percent": friendly_target_percent,
                # "enemy_target_percent": enemy_target_percent,
                # "neutral_target_percent": neutral_target_percent,
                "trajectory_shield_blocked_count": float(
                    rollout_scalars["trajectory_shield_blocked_count"]
                ),
                "trajectory_shield_blocked_sun_count": float(
                    rollout_scalars["trajectory_shield_blocked_sun_count"]
                ),
                "trajectory_shield_blocked_bounds_count": float(
                    rollout_scalars["trajectory_shield_blocked_bounds_count"]
                ),
                "trajectory_shield_blocked_unintended_hit_count": float(
                    rollout_scalars["trajectory_shield_blocked_unintended_hit_count"]
                ),
                "trajectory_shield_blocked_horizon_count": float(
                    rollout_scalars["trajectory_shield_blocked_horizon_count"]
                ),
                "trajectory_shield_fallback_noop_count": float(
                    rollout_scalars["trajectory_shield_fallback_noop_count"]
                ),
                "trajectory_shield_legal_non_noop_rate": float(
                    rollout_scalars["trajectory_shield_legal_non_noop_rate"]
                ),
                "stop_rate": float(rollout_scalars["stop_rate"]),
                "mean_active_launches_per_turn": float(
                    rollout_scalars["mean_active_launches_per_turn"]
                ),
                "stop_utilization_ratio": float(
                    rollout_scalars["mean_active_launches_per_turn"]
                )
                / max(float(cfg.model.max_moves_k), 1.0),
                "survival_time": survival_time,
                "score_share": score_share,
                "update_seconds": update_seconds,
                "elapsed_seconds": time.perf_counter() - train_start_time,
                "rollout_seconds": rollout_seconds,
                "ppo_seconds": ppo_seconds,
                "env_steps_per_sec": env_steps / max(update_seconds, 1e-9),
                "rollout_env_steps_per_sec": env_steps / max(rollout_seconds, 1e-9),
                "samples_per_sec": rollout_samples / max(update_seconds, 1e-9),
                "ppo_samples_per_sec": rollout_samples / max(ppo_seconds, 1e-9),
                **per_format_timing_metrics,
                "seed_scheduler_policy": seed_scheduler.next_seed_policy(update),
                "seed_scheduler_plateau_metric": cfg.training.plateau_metric,
                "reseed_events": reseed_events,
                **curriculum_telemetry,
                "opponent_slots_total": float(rollout_scalars["opponent_slots_total"]),
                "opponent_slots_latest": float(
                    rollout_scalars["opponent_slots_latest"]
                ),
                "opponent_slots_historical": float(
                    rollout_scalars["opponent_slots_historical"]
                ),
                "opponent_slots_random": float(
                    rollout_scalars["opponent_slots_random"]
                ),
                "opponent_slots_noop": float(rollout_scalars["opponent_slots_noop"]),
                "opponent_slots_nearest_sniper": float(
                    rollout_scalars["opponent_slots_nearest_sniper"]
                ),
                "opponent_slots_turtle": float(
                    rollout_scalars["opponent_slots_turtle"]
                ),
                "opponent_slots_opportunistic": float(
                    rollout_scalars["opponent_slots_opportunistic"]
                ),
                "opponent_historical_fallback_latest_slots": float(
                    rollout_scalars["opponent_historical_fallback_latest_slots"]
                ),
                "historical_pool_size": int(
                    jax.device_get(historical_pool.valid_mask).sum()
                ),
                "historical_pool_capacity": int(historical_pool.valid_mask.shape[0]),
                "historical_snapshot_ids": historical_ids,
                "historical_snapshot_ages_updates": historical_ages,
                **{name: float(value) for name, value in metrics_host.items()},
                # "won_non_noop_actions_per_step": float(
                #     rollout_scalars["won_non_noop_actions_per_step"]
                # ),
                # "lost_non_noop_actions_per_step": float(
                #     rollout_scalars["lost_non_noop_actions_per_step"]
                # ),
                # "won_avg_fleet_launch_size": float(
                #     rollout_scalars["won_avg_fleet_launch_size"]
                # ),
                # "lost_avg_fleet_launch_size": float(
                #     rollout_scalars["lost_avg_fleet_launch_size"]
                # ),
                # "won_avg_planets_owned": float(
                #     rollout_scalars["won_avg_planets_owned"]
                # ),
                # "lost_avg_planets_owned": float(
                #     rollout_scalars["lost_avg_planets_owned"]
                # ),
                # "won_avg_planets_lost": float(rollout_scalars["won_avg_planets_lost"]),
                # "lost_avg_planets_lost": float(
                #     rollout_scalars["lost_avg_planets_lost"]
                # ),
                # "won_avg_planets_taken": float(
                #     rollout_scalars["won_avg_planets_taken"]
                # ),
                # "lost_avg_planets_taken": float(
                #     rollout_scalars["lost_avg_planets_taken"]
                # ),
                # "won_avg_garrisoned_ships_per_planet": float(
                #     rollout_scalars["won_avg_garrisoned_ships_per_planet"]
                # ),
                # "lost_avg_garrisoned_ships_per_planet": float(
                #     rollout_scalars["lost_avg_garrisoned_ships_per_planet"]
                # ),
                # "won_avg_planet_diff": float(rollout_scalars["won_avg_planet_diff"]),
                # "lost_avg_planet_diff": float(rollout_scalars["lost_avg_planet_diff"]),
                # "won_avg_production_diff": float(
                #     rollout_scalars["won_avg_production_diff"]
                # ),
                # "lost_avg_production_diff": float(
                #     rollout_scalars["lost_avg_production_diff"]
                # ),
                # "won_avg_launch_fleet_speed": float(
                #     rollout_scalars["won_avg_launch_fleet_speed"]
                # ),
                # "lost_avg_launch_fleet_speed": float(
                #     rollout_scalars["lost_avg_launch_fleet_speed"]
                # ),
                "opponent_composition": {
                    "latest": float(rollout_scalars["opponent_slots_latest"]),
                    "historical": float(rollout_scalars["opponent_slots_historical"]),
                    "random": float(rollout_scalars["opponent_slots_random"]),
                    "noop": float(rollout_scalars["opponent_slots_noop"]),
                    "nearest_sniper": float(
                        rollout_scalars["opponent_slots_nearest_sniper"]
                    ),
                    "turtle": float(rollout_scalars["opponent_slots_turtle"]),
                    "opportunistic": float(
                        rollout_scalars["opponent_slots_opportunistic"]
                    ),
                },
                "curriculum_phase_id": curriculum_telemetry["curriculum_stage_id"],
                "curriculum_phase_events": list(update_events),
            }
            append_jsonl(log_path, record)
            telemetry.log(record, step=update)
            if update % cfg.training.log_every == 0:
                entropy_line = f"entropy={float(record['entropy']):.4f}"
                if "entropy_stop" in record and "entropy_move" in record:
                    entropy_line = (
                        f"entropy_stop={float(record['entropy_stop']):.4f} "
                        f"entropy_move={float(record['entropy_move']):.4f} "
                        f"entropy={float(record['entropy']):.4f}"
                    )
                print(
                    f"update={update} steps={total_env_steps} episodes={completed_episodes} "
                    f"loss={record['total_loss']:.4f} sps={record['samples_per_sec']:.1f} "
                    f"rollout_s={rollout_seconds:.3f} ppo_s={ppo_seconds:.3f} "
                    f"{entropy_line}"
                )
            if update % cfg.artifacts.checkpoint_every == 0 or update == cfg.training.total_updates:
                is_final = update == cfg.training.total_updates
                if checkpoint_pipeline is None:
                    checkpoint_path = save_jax_checkpoint(
                        run_dir,
                        update,
                        train_state,
                        cfg,
                        key=key,
                        total_env_steps=total_env_steps,
                        completed_episodes=completed_episodes,
                        curriculum=curriculum,
                        historical_pool=historical_pool,
                    )
                    handle_checkpoint_results(
                        [
                            CheckpointResult(
                                job_id=f"sync-{update}",
                                update=update,
                                status="committed",
                                numbered_path=checkpoint_path,
                                latest_path=run_dir / "jax_ckpt_last.pkl",
                                final=is_final,
                            )
                        ]
                    )
                else:
                    job = CheckpointJob(
                        update=update,
                        run_dir=run_dir,
                        build_payload=_checkpoint_payload_builder(
                            train_state,
                            cfg,
                            key=key,
                            update=update,
                            total_env_steps=total_env_steps,
                            completed_episodes=completed_episodes,
                            curriculum=curriculum,
                            historical_pool=historical_pool,
                        ),
                        final=is_final,
                    )
                    handle_checkpoint_results(checkpoint_pipeline.submit_checkpoint(job))
    
        completed_training = True
    finally:
        if checkpoint_pipeline is not None:
            timeout_seconds = (
                artifact_cfg.final_flush_timeout_seconds
                if completed_training
                else artifact_cfg.exception_flush_timeout_seconds
            )
            try:
                handle_checkpoint_results(
                    checkpoint_pipeline.close(timeout_seconds=timeout_seconds)
                )
            except Exception as exc:
                if close_error is None:
                    close_error = exc
        telemetry.finish()
    if close_error is not None:
        raise ArtifactPipelineError(
            f"artifact pipeline shutdown failed: {close_error}"
        ) from close_error
    if checkpoint_failures and artifact_cfg.fail_training_on_checkpoint_error:
        first_failure = checkpoint_failures[0]
        raise ArtifactPipelineError(
            f"checkpoint worker failed at update {first_failure.update}: "
            f"{first_failure.error or first_failure.reason or first_failure.status}"
        )
    return log_path


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    """Append a JSON metrics record to ``path``, creating parents as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


def load_jax_checkpoint(
    checkpoint_path: str, train_state: object, cfg: TrainConfig
) -> tuple[object, jax.Array, int, int, int]:
    """Load JAX training state and counters from a checkpoint payload."""

    checkpoint = load_checkpoint_payload(checkpoint_path)
    if not isinstance(checkpoint, dict) or "params" not in checkpoint:
        raise ValueError(
            f"JAX checkpoint must contain a parameter payload: {checkpoint_path}"
        )
    validate_checkpoint_config_compatibility(
        checkpoint, checkpoint_path=checkpoint_path
    )
    validate_checkpoint_feature_compatibility(
        checkpoint, cfg.task, checkpoint_path=checkpoint_path
    )
    stored_metadata = checkpoint_feature_metadata(checkpoint)
    validate_checkpoint_encoder_compatibility(
        stored_metadata,
        cfg,
        checkpoint_path=checkpoint_path,
    )
    validate_checkpoint_pointer_decoder_compatibility(
        stored_metadata,
        cfg,
        checkpoint_path=checkpoint_path,
    )
    params = jax.device_put(checkpoint["params"])
    opt_state = checkpoint.get("opt_state")
    if opt_state is None:
        opt_state = train_state.optimizer.init(params)
    else:
        opt_state = jax.device_put(opt_state)
    checkpoint_update = int(checkpoint.get("update", 0))
    key_payload = checkpoint.get("rng_key")
    key = (
        jax.device_put(key_payload)
        if key_payload is not None
        else jax.random.PRNGKey(cfg.seed + checkpoint_update)
    )
    total_env_steps = int(
        checkpoint.get(
            "total_env_steps",
            checkpoint_update * cfg.training.rollout_steps * cfg.training.num_envs,
        )
    )
    completed_episodes = int(checkpoint.get("completed_episodes", 0))
    return (
        train_state.replace(params=params, opt_state=opt_state),
        key,
        checkpoint_update + 1,
        total_env_steps,
        completed_episodes,
    )


def save_jax_checkpoint(
    run_dir: Path,
    update: int,
    train_state: object,
    cfg: TrainConfig,
    *,
    key: jax.Array,
    total_env_steps: int,
    completed_episodes: int,
    curriculum: CurriculumController | None = None,
    historical_pool: HistoricalSnapshotPool | None = None,
) -> Path:
    """Persist the latest and update-numbered JAX checkpoint payloads."""
    payload = _checkpoint_payload_builder(
        train_state,
        cfg,
        key=key,
        update=update,
        total_env_steps=total_env_steps,
        completed_episodes=completed_episodes,
        curriculum=curriculum,
        historical_pool=historical_pool,
    )()
    return commit_checkpoint_payload(run_dir, update, payload)
