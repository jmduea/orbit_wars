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

from .artifact_pipeline import (
    ArtifactPipelineError,
    AsyncArtifactPipeline,
    CheckpointJob,
    CheckpointResult,
    commit_checkpoint_payload,
    load_active_optional_jobs,
    protected_paths_from_jobs,
    write_optional_job,
)
from .checkpoint_compat import (
    feature_metadata,
    validate_checkpoint_feature_compatibility,
)
from .checkpoint_retention import prune_checkpoints
from .config import TrainConfig
from .curriculum import CurriculumController
from .jax_device import (
    configure_jax_platform_for_host,
    ensure_cuda_jax_if_nvidia_present,
)
from .replay import maybe_write_jax_checkpoint_replay
from .run_paths import resolve_run_paths
from .seed_scheduler import SeedScheduleConfig, SeedScheduler
from .telemetry import build_telemetry

configure_jax_platform_for_host()
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from .jax_env import JaxEnvState, assign_learner_players, batched_reset  # noqa: E402
from .jax_features import JaxTurnBatch  # noqa: E402
from .jax_policy import build_jax_policy  # noqa: E402
from .jax_ppo import (  # noqa: E402
    JaxTransitionBatch,
    collect_rollout_jax,
    concatenate_transition_batches,
    init_train_state,
    ppo_update_jax,
    validate_policy_param_shapes,
)
from .metric_registry import (  # noqa: E402
    filter_event_record,
    filter_update_record,
    required_ppo_metric_names,
    required_rollout_scalar_names,
)


@dataclass(slots=True)
class JaxRolloutGroup:
    """State for one statically compiled JAX rollout format."""

    name: str
    cfg: TrainConfig
    env_state: JaxEnvState
    turn_batch: JaxTurnBatch
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
    group_cfg.env.player_count = int(player_count)
    group_cfg.ppo.num_envs = int(num_envs)
    return group_cfg


def _configured_rollout_groups(cfg: TrainConfig) -> list[dict[str, int | str]]:
    """Resolve rollout group declarations for Option A mixed-format training.

    The JAX trainer keeps independent 2-player and 4-player environment states
    and compiles one collector per declared static format. If no groups are
    configured, it falls back to the legacy single-format collector.
    """

    raw_groups = cfg.training_format.rollout_groups
    groups: list[dict[str, int | str]] = []
    for index, group in enumerate(raw_groups):
        player_count = int(group.get("player_count", cfg.env.player_count))
        if player_count not in {2, 4}:
            raise ValueError(
                f"JAX rollout groups support player_count 2 or 4, got {player_count}."
            )
        num_envs = int(group.get("num_envs", cfg.ppo.num_envs))
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
            "name": f"{cfg.env.player_count}p",
            "player_count": int(cfg.env.player_count),
            "num_envs": int(cfg.ppo.num_envs),
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
    reset_keys = jax.random.split(key, group_cfg.ppo.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, group_cfg.env)
    env_indices = jnp.arange(group_cfg.ppo.num_envs, dtype=jnp.int32)
    episode_counts = jnp.zeros((group_cfg.ppo.num_envs,), dtype=jnp.int32)
    env_state, turn_batch = assign_learner_players(
        env_state,
        env_indices,
        episode_counts,
        group_cfg.env,
        group_cfg.alternate_player_sides,
    )

    def collect_fn(
        rollout_key,
        state,
        batch,
        ts,
        stage_view=None,
        historical_params_pool=None,
        update_idx=jnp.asarray(0, dtype=jnp.int32),
    ):
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
    group: JaxRolloutGroup, env_state: JaxEnvState, turn_batch: JaxTurnBatch
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
    metadata = feature_metadata(cfg_snapshot.env)
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
    if not cfg.replay.enabled:
        return False
    every_n = max(int(cfg.replay.every_n_checkpoints), 1)
    checkpoint_index = max(update // max(int(cfg.checkpoint_every), 1), 1)
    return checkpoint_index % every_n == 0 or update == cfg.ppo.total_updates


def _queue_optional_jobs_if_due(
    cfg: TrainConfig,
    *,
    update: int,
    checkpoint_path: Path,
    log_path: Path,
    queue_dir: Path,
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
                    "backend": cfg.artifact_pipeline.replay_backend,
                    "log_path": str(log_path),
                    "replay_output_dir": cfg.replay.output_dir,
                    "docker_image": cfg.artifact_pipeline.docker_image,
                    "player_count": cfg.artifact_pipeline.docker_player_count,
                    "timeout_seconds": cfg.artifact_pipeline.docker_timeout_seconds,
                    "episode_steps": cfg.replay.max_steps,
                    "seed": cfg.seed + update,
                },
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
                    "docker_image": cfg.artifact_pipeline.docker_image,
                    "player_count": cfg.artifact_pipeline.docker_player_count,
                    "timeout_seconds": cfg.artifact_pipeline.docker_timeout_seconds,
                    "episode_steps": cfg.replay.max_steps,
                    "seed": cfg.seed + update,
                },
            )
        )
    return job_paths


def _start_artifact_worker_if_needed(
    cfg: TrainConfig,
    *,
    queue_dir: Path,
    worker_state: dict[str, subprocess.Popen[object]],
) -> None:
    if not cfg.artifact_pipeline.worker_autostart:
        return
    worker = worker_state.get("process")
    if worker is not None and worker.poll() is None:
        return
    queue_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = queue_dir / "worker.stdout.log"
    stderr_path = queue_dir / "worker.stderr.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts" / "run_artifact_worker.py"),
        str(queue_dir),
        "--poll-seconds",
        str(cfg.artifact_pipeline.worker_poll_seconds),
        "--idle-exit-seconds",
        str(cfg.artifact_pipeline.worker_idle_exit_seconds),
    ]
    stdout = stdout_path.open("a", encoding="utf-8")
    stderr = stderr_path.open("a", encoding="utf-8")
    worker_state["process"] = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[1],
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )


def _active_group_indices(
    groups: list[JaxRolloutGroup], format_weights: dict[int, float]
) -> list[int]:
    active: list[int] = []
    for idx, group in enumerate(groups):
        player_count = int(group.cfg.env.player_count)
        if float(format_weights.get(player_count, 0.0)) > 0.0:
            active.append(idx)
    return active or list(range(len(groups)))


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
    import pickle

    with Path(checkpoint_path).open("rb") as file:
        checkpoint = pickle.load(file)
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
    interval = int(cfg.curriculum.snapshot.interval_updates)
    return interval > 0 and update % interval == 0


def run_jax_training(cfg: TrainConfig, resume_checkpoint: str | None = None) -> None:
    """Run an end-to-end JAX training loop for the JAX environment backend.

    This path keeps environment state, feature encoding, action sampling, rollout
    storage, return/advantage computation, and PPO updates in JAX. Mixed 2p/4p
    training uses Option A: each format owns its env state and jitted collector,
    then compatible transition batches are concatenated before PPO updates.
    """

    ensure_cuda_jax_if_nvidia_present()

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
        validate_policy_param_shapes(train_state.params, cfg.env)
        print(
            f"Resuming JAX training from {resume_checkpoint} at update {start_update}"
        )
    update_fn = jax.jit(
        lambda ts, transitions: ppo_update_jax(ts, policy, transitions, cfg)
    )
    cfg, run_dir, log_path, _save_dir = resolve_run_paths(cfg)
    log_path = log_path.with_name(f"{cfg.run_name}_jax.jsonl")
    telemetry = build_telemetry(
        cfg,
        {
            "backend": "jax",
            "seed": cfg.seed,
        },
    )
    seed_scheduler = SeedScheduler(
        base_seed=cfg.seed,
        cfg=SeedScheduleConfig(
            reseed_every_updates=cfg.reseed_every_updates,
            reseed_on_plateau=cfg.reseed_on_plateau,
            plateau_metric=cfg.plateau_metric,
            plateau_window=cfg.plateau_window,
            plateau_delta=cfg.plateau_delta,
            heldout_eval_seed_set=cfg.heldout_eval_seed_set,
        ),
    )
    curriculum = CurriculumController(cfg.curriculum)
    historical_pool = _init_historical_snapshot_pool(
        train_state.params, cfg.curriculum.snapshot.pool_size
    )
    if resume_checkpoint is not None:
        historical_pool = _restore_curriculum_artifacts(
            resume_checkpoint, curriculum, historical_pool
        )
    phase_events: list[dict[str, object]] = []
    train_start_time = time.perf_counter()
    artifact_cfg = cfg.artifact_pipeline
    artifact_queue_dir = run_dir / artifact_cfg.queue_dir
    checkpoint_pipeline = (
        AsyncArtifactPipeline(
            checkpoint_queue_size=artifact_cfg.checkpoint_queue_size,
            coalesce_intermediate_checkpoints=artifact_cfg.coalesce_intermediate_checkpoints,
            ledger_path=(run_dir / "artifact_pipeline.jsonl")
            if artifact_cfg.ledger_enabled
            else None,
        )
        if artifact_cfg.enabled
        else None
    )
    checkpoint_failures: list[CheckpointResult] = []
    artifact_worker_state: dict[str, subprocess.Popen[object]] = {}

    def protected_artifact_paths() -> set[Path]:
        paths = {run_dir / "jax_ckpt_last.pkl"}
        if checkpoint_pipeline is not None:
            paths.update(checkpoint_pipeline.protected_paths())
        paths.update(protected_paths_from_jobs(load_active_optional_jobs(artifact_queue_dir)))
        return paths

    def handle_checkpoint_results(results: list[CheckpointResult]) -> None:
        for result in results:
            event_record = {
                "event": "checkpoint_result",
                "update": result.update,
                "checkpoint_status": result.status,
                "checkpoint_final": result.final,
                "checkpoint_reason": result.reason,
                "checkpoint_error": result.error,
            }
            output_event_record = filter_event_record(event_record, cfg)
            append_jsonl(log_path, output_event_record)
            telemetry.log(output_event_record, step=result.update)
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
                    queue_replay=artifact_cfg.replay_async,
                    queue_docker_validation=artifact_cfg.docker_validation_async,
                )
                if job_paths:
                    _start_artifact_worker_if_needed(
                        cfg,
                        queue_dir=artifact_queue_dir,
                        worker_state=artifact_worker_state,
                    )
                protected_paths.update(
                    protected_paths_from_jobs(load_active_optional_jobs(artifact_queue_dir))
                )

            retention = cfg.checkpoint_retention
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
            telemetry.log_checkpoint(result.numbered_path, update=result.update)
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
        for update in range(start_update, cfg.ppo.total_updates + 1):
            if checkpoint_pipeline is not None:
                handle_checkpoint_results(checkpoint_pipeline.drain_results())
            update_start = time.perf_counter()
            reseed_events: list[dict[str, object]] = []
            rollout_start = time.perf_counter()
            transitions_by_group: list[JaxTransitionBatch] = []
            rollout_metrics_by_group: list[dict[str, jax.Array]] = []
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
                rollout_groups, curriculum.current_format_weights()
            )
            key, *rollout_keys = jax.random.split(key, len(active_indices) + 1)
            for group_idx, rollout_key in zip(active_indices, rollout_keys, strict=True):
                group = rollout_groups[group_idx]
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
                )
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
            rollout_metrics = jax.tree.map(lambda *xs: sum(xs), *rollout_metrics_by_group)
            rollout_metrics = dict(rollout_metrics)
            shield_original_count = rollout_metrics.get(
                "trajectory_shield_original_non_noop_count", 0.0
            )
            shield_legal_count = rollout_metrics.get(
                "trajectory_shield_legal_non_noop_count", 0.0
            )
            rollout_metrics["trajectory_shield_legal_non_noop_rate"] = jnp.where(
                shield_original_count > 0.0,
                shield_legal_count / shield_original_count,
                0.0,
            )
            rollout_scalar_keys = required_rollout_scalar_names(cfg)
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
            for _ in range(cfg.ppo.epochs):
                train_state, update_metrics = update_fn(train_state, transitions)
                metrics_accum = (
                    update_metrics
                    if metrics_accum is None
                    else jax.tree.map(jnp.add, metrics_accum, update_metrics)
                )
            assert metrics_accum is not None
            metrics = jax.tree.map(
                lambda x: x / float(max(cfg.ppo.epochs, 1)), metrics_accum
            )
            metric_names = required_ppo_metric_names(cfg, tuple(metrics.keys()))
            if metric_names:
                metric_values = jnp.asarray([metrics[name] for name in metric_names])
                # Intentional sync boundary: perform one compact host transfer for
                # the PPO scalars still needed for output or training control.
                metric_values_host = jax.device_get(metric_values)
                metrics_host = dict(
                    zip(metric_names, metric_values_host.tolist(), strict=True)
                )
            else:
                metrics_host = {}
            ppo_seconds = time.perf_counter() - ppo_start
            update_seconds = time.perf_counter() - update_start
            env_steps = int(rollout_scalars["env_steps"])
            episodes = int(rollout_scalars["episode_done"])
            win_rate_2p = float(rollout_scalars.get("win_rate_2p", 0.0))
            first_place_rate_4p = float(rollout_scalars.get("first_place_rate_4p", 0.0))
            average_placement_4p = float(
                rollout_scalars.get("average_placement_4p", 0.0)
            )
            survival_time = float(rollout_scalars.get("survival_time", 0.0))
            score_share = float(rollout_scalars.get("score_share", 0.0))
            average_reward = float(rollout_scalars["average_reward"])
            episode_reward_mean = float(rollout_scalars["episode_reward_mean"])
            overall_win_rate = float(rollout_scalars["overall_win_rate"])
            noop_percent = float(rollout_scalars.get("noop_percent", 0.0))
            friendly_target_percent = float(
                rollout_scalars.get("friendly_target_percent", 0.0)
            )
            enemy_target_percent = float(
                rollout_scalars.get("enemy_target_percent", 0.0)
            )
            neutral_target_percent = float(
                rollout_scalars.get("neutral_target_percent", 0.0)
            )
            total_env_steps += env_steps
            completed_episodes += episodes
            curriculum_telemetry = curriculum.stage_telemetry(stage_view, update)
            update_events = list(phase_events)
            transition = curriculum.update(
                update,
                {
                    "overall_win_rate": overall_win_rate,
                    "win_rate_2p": win_rate_2p,
                    "first_place_rate_4p": first_place_rate_4p,
                    "average_reward": average_reward,
                    "episode_reward_mean": episode_reward_mean,
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
                "episode_reward_mean": episode_reward_mean,
                "noop_percent": noop_percent,
                "friendly_target_percent": friendly_target_percent,
                "enemy_target_percent": enemy_target_percent,
                "neutral_target_percent": neutral_target_percent,
                "trajectory_shield_blocked_count": float(
                    rollout_scalars.get("trajectory_shield_blocked_count", 0.0)
                ),
                "trajectory_shield_blocked_sun_count": float(
                    rollout_scalars.get("trajectory_shield_blocked_sun_count", 0.0)
                ),
                "trajectory_shield_blocked_bounds_count": float(
                    rollout_scalars.get("trajectory_shield_blocked_bounds_count", 0.0)
                ),
                "trajectory_shield_blocked_unintended_hit_count": float(
                    rollout_scalars.get(
                        "trajectory_shield_blocked_unintended_hit_count", 0.0
                    )
                ),
                "trajectory_shield_blocked_horizon_count": float(
                    rollout_scalars.get("trajectory_shield_blocked_horizon_count", 0.0)
                ),
                "trajectory_shield_fallback_noop_count": float(
                    rollout_scalars.get("trajectory_shield_fallback_noop_count", 0.0)
                ),
                "trajectory_shield_legal_non_noop_rate": float(
                    rollout_scalars.get("trajectory_shield_legal_non_noop_rate", 0.0)
                ),
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
                "seed_scheduler_policy": seed_scheduler.next_seed_policy(update),
                "seed_scheduler_plateau_metric": cfg.plateau_metric,
                "reseed_events": reseed_events,
                **curriculum_telemetry,
                "opponent_slots_total": float(
                    rollout_scalars.get("opponent_slots_total", 0.0)
                ),
                "opponent_slots_latest": float(
                    rollout_scalars.get("opponent_slots_latest", 0.0)
                ),
                "opponent_slots_historical": float(
                    rollout_scalars.get("opponent_slots_historical", 0.0)
                ),
                "opponent_slots_random": float(
                    rollout_scalars.get("opponent_slots_random", 0.0)
                ),
                "opponent_slots_noop": float(
                    rollout_scalars.get("opponent_slots_noop", 0.0)
                ),
                "opponent_slots_nearest_sniper": float(
                    rollout_scalars.get("opponent_slots_nearest_sniper", 0.0)
                ),
                "opponent_slots_turtle": float(
                    rollout_scalars.get("opponent_slots_turtle", 0.0)
                ),
                "opponent_slots_opportunistic": float(
                    rollout_scalars.get("opponent_slots_opportunistic", 0.0)
                ),
                "opponent_historical_fallback_latest_slots": float(
                    rollout_scalars.get(
                        "opponent_historical_fallback_latest_slots", 0.0
                    )
                ),
                "historical_pool_size": int(
                    jax.device_get(historical_pool.valid_mask).sum()
                ),
                "historical_pool_capacity": int(historical_pool.valid_mask.shape[0]),
                "historical_snapshot_ids": historical_ids,
                "historical_snapshot_ages_updates": historical_ages,
                **{name: float(value) for name, value in metrics_host.items()},
                "won_non_noop_actions_per_step": float(
                    rollout_scalars.get("won_non_noop_actions_per_step", 0.0)
                ),
                "lost_non_noop_actions_per_step": float(
                    rollout_scalars.get("lost_non_noop_actions_per_step", 0.0)
                ),
                "won_avg_fleet_launch_size": float(
                    rollout_scalars.get("won_avg_fleet_launch_size", 0.0)
                ),
                "lost_avg_fleet_launch_size": float(
                    rollout_scalars.get("lost_avg_fleet_launch_size", 0.0)
                ),
                "won_avg_planets_owned": float(
                    rollout_scalars.get("won_avg_planets_owned", 0.0)
                ),
                "lost_avg_planets_owned": float(
                    rollout_scalars.get("lost_avg_planets_owned", 0.0)
                ),
                "won_avg_planets_lost": float(
                    rollout_scalars.get("won_avg_planets_lost", 0.0)
                ),
                "lost_avg_planets_lost": float(
                    rollout_scalars.get("lost_avg_planets_lost", 0.0)
                ),
                "won_avg_planets_taken": float(
                    rollout_scalars.get("won_avg_planets_taken", 0.0)
                ),
                "lost_avg_planets_taken": float(
                    rollout_scalars.get("lost_avg_planets_taken", 0.0)
                ),
                "won_avg_garrisoned_ships_per_planet": float(
                    rollout_scalars.get("won_avg_garrisoned_ships_per_planet", 0.0)
                ),
                "lost_avg_garrisoned_ships_per_planet": float(
                    rollout_scalars.get("lost_avg_garrisoned_ships_per_planet", 0.0)
                ),
                "won_avg_planet_diff": float(
                    rollout_scalars.get("won_avg_planet_diff", 0.0)
                ),
                "lost_avg_planet_diff": float(
                    rollout_scalars.get("lost_avg_planet_diff", 0.0)
                ),
                "won_avg_production_diff": float(
                    rollout_scalars.get("won_avg_production_diff", 0.0)
                ),
                "lost_avg_production_diff": float(
                    rollout_scalars.get("lost_avg_production_diff", 0.0)
                ),
                "won_avg_launch_fleet_speed": float(
                    rollout_scalars.get("won_avg_launch_fleet_speed", 0.0)
                ),
                "lost_avg_launch_fleet_speed": float(
                    rollout_scalars.get("lost_avg_launch_fleet_speed", 0.0)
                ),
                "opponent_composition": {
                    "latest": float(rollout_scalars.get("opponent_slots_latest", 0.0)),
                    "historical": float(
                        rollout_scalars.get("opponent_slots_historical", 0.0)
                    ),
                    "random": float(rollout_scalars.get("opponent_slots_random", 0.0)),
                    "noop": float(rollout_scalars.get("opponent_slots_noop", 0.0)),
                    "nearest_sniper": float(
                        rollout_scalars.get("opponent_slots_nearest_sniper", 0.0)
                    ),
                    "turtle": float(rollout_scalars.get("opponent_slots_turtle", 0.0)),
                    "opportunistic": float(
                        rollout_scalars.get("opponent_slots_opportunistic", 0.0)
                    ),
                },
                "curriculum_phase_id": curriculum_telemetry["curriculum_stage_id"],
                "curriculum_phase_events": list(update_events),
            }
            plateau_metric_value = record.get(cfg.plateau_metric)
            if not isinstance(plateau_metric_value, int | float):
                raise KeyError(
                    "Configured plateau_metric was not produced by the telemetry record: "
                    f"{cfg.plateau_metric}"
                )
            seed_scheduler.update_metric(float(plateau_metric_value))
            output_record = filter_update_record(record, cfg)
            append_jsonl(log_path, output_record)
            telemetry.log(output_record, step=update)
            if update % cfg.log_every == 0:
                total_loss_display = (
                    f"{record['total_loss']:.4f}" if "total_loss" in record else "n/a"
                )
                entropy_display = (
                    f"{record['entropy']:.3f}" if "entropy" in record else "n/a"
                )
                print(
                    f"update={update} steps={total_env_steps} episodes={completed_episodes} "
                    f"loss={total_loss_display} sps={record['samples_per_sec']:.1f} "
                    f"rollout_s={rollout_seconds:.3f} ppo_s={ppo_seconds:.3f} "
                    f"entropy={entropy_display}"
                )
            if update % cfg.checkpoint_every == 0 or update == cfg.ppo.total_updates:
                is_final = update == cfg.ppo.total_updates
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


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    """Append a JSON metrics record to ``path``, creating parents as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


def load_jax_checkpoint(
    checkpoint_path: str, train_state: object, cfg: TrainConfig
) -> tuple[object, jax.Array, int, int, int]:
    """Load JAX training state and counters from a checkpoint payload."""

    import pickle

    with Path(checkpoint_path).open("rb") as file:
        checkpoint = pickle.load(file)
    if not isinstance(checkpoint, dict) or "params" not in checkpoint:
        raise ValueError(
            f"JAX checkpoint must contain a parameter payload: {checkpoint_path}"
        )
    validate_checkpoint_feature_compatibility(
        checkpoint, cfg.env, checkpoint_path=checkpoint_path
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
            checkpoint_update * cfg.ppo.rollout_steps * cfg.ppo.num_envs,
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
