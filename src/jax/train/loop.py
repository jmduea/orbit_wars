from __future__ import annotations

import logging
import time
from pathlib import Path

import jax.numpy as jnp

import jax
from src.artifacts.checkpoint_compat import is_planet_flow_pointer_decoder
from src.artifacts.pipeline import (
    ArtifactPipelineError,
    AsyncArtifactPipeline,
    CheckpointResult,
)
from src.artifacts.run_paths import resolve_run_paths, write_run_manifests
from src.config import TrainConfig
from src.config.rollout_allocation import infer_static_format_weights
from src.jax.device import (
    configure_jax_runtime_for_host,
    ensure_jax_accelerator_backend,
)
from src.jax.normalization import (
    init_observation_norm_state,
    normalize_transition_batch,
    update_norm_state_from_transitions,
)
from src.jax.policy import build_jax_policy
from src.jax.ppo_update import concatenate_transition_batches, ppo_update_jax
from src.jax.preflight_calibration import default_calibration_json_path, load_thresholds
from src.jax.rollout.metrics import FINALIZED_ROLLOUT_RATE_KEYS
from src.jax.rollout.types import JaxTransitionBatch
from src.jax.train.checkpoint import (
    CheckpointHandler,
    load_jax_checkpoint,
    restore_curriculum_artifacts,
    save_jax_checkpoint,
)
from src.jax.train.metrics import prune_merged_rollout_metrics, sum_metric_dicts
from src.jax.train.rollout_groups import (
    JaxRolloutGroup,
    active_group_indices,
    empty_per_format_rollout_stats,
    init_rollout_groups,
    replace_rollout_group_state,
    reset_rollout_groups_envs,
)
from src.jax.train.snapshots import (
    add_historical_snapshot,
    init_historical_snapshot_pool,
    snapshot_due,
)
from src.jax.train.state import init_train_state, validate_policy_param_shapes
from src.jax.train.sweep_score import (
    MetricWindowTracker,
    WinRateTrendTracker,
    collect_planet_flow_sweep_metrics,
    collect_ssot_preflight_sweep_metrics,
    is_ssot_preflight_sweep,
    planet_flow_max_post_mask_unreachable_rate,
)
from src.jax.train.telemetry import (
    build_per_format_timing_metrics,
    build_update_record,
    historical_pool_snapshot_telemetry,
    write_filtered_update_records,
)
from src.telemetry import build_telemetry
from src.telemetry.gpu_memory import GpuMemoryTracker
from src.telemetry.metric_registry import (
    enabled_metric_groups,
    metric_groups_cfg_from_config,
    required_ppo_metric_names,
    required_rollout_scalar_names,
)
from src.jax.train.bracket_training import bracket_training_enabled, bracket_training_tick
from src.jax.tournament_qualifiers.runner import (
    ssot_pipeline_enabled,
    ssot_qualifier_tick,
    ssot_qualifier_telemetry,
)
from src.training.curriculum import CurriculumController
from src.training.seed_scheduler import (
    SeedScheduleConfig,
    SeedScheduler,
    resolve_reseed_every_updates,
)

configure_jax_runtime_for_host()
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)


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

    metric_group_cfg = metric_groups_cfg_from_config(cfg)
    active_metric_groups = enabled_metric_groups(metric_group_cfg)
    track_gpu_memory = "timing" in active_metric_groups
    gpu_tracker = GpuMemoryTracker() if track_gpu_memory else None
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
    debug_log_path = run_context.debug_log_path
    write_run_manifests(
        cfg,
        run_context,
        {
            "backend": "jax",
            "job_type": "train",
            "wandb_dir": str(run_context.wandb_dir),
            "wandb_artifact_dir": str(run_context.wandb_artifact_dir),
            "wandb_data_dir": str(run_context.wandb_data_dir),
            **(gpu_tracker.run_metadata() if gpu_tracker is not None else {}),
        },
    )
    wandb_enabled = bool(cfg.telemetry.wandb.enabled)
    print(
        f"orbit_train_start run_dir={run_context.run_dir} log_path={log_path} "
        f"queue_dir={run_context.queue_dir} wandb={'on' if wandb_enabled else 'off'}"
    )
    telemetry = build_telemetry(
        cfg,
        {
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
    effective_reseed_every = resolve_reseed_every_updates(
        configured=cfg.training.reseed_every_updates,
        total_updates=cfg.training.total_updates,
    )
    seed_scheduler = SeedScheduler(
        base_seed=cfg.seed,
        cfg=SeedScheduleConfig(
            reseed_every_updates=effective_reseed_every,
            reseed_on_plateau=cfg.training.reseed_on_plateau,
            plateau_metric=cfg.training.plateau_metric,
            plateau_window=cfg.training.plateau_window,
            plateau_delta=cfg.training.plateau_delta,
            training_seed_set=cfg.training_seed_set,
            eval_seed_set=cfg.eval_seed_set,
        ),
    )
    curriculum = CurriculumController(
        cfg.curriculum,
        cfg.opponents.snapshot,
        static_format_weights=infer_static_format_weights(cfg),
    )
    historical_pool = init_historical_snapshot_pool(
        train_state.params, cfg.opponents.snapshot.pool_size
    )
    if resume_checkpoint is not None:
        historical_pool = restore_curriculum_artifacts(
            resume_checkpoint, curriculum, historical_pool
        )
    phase_events: list[dict[str, object]] = []
    train_start_time = time.perf_counter()
    track_planet_flow_sweep = is_planet_flow_pointer_decoder(cfg.model)
    track_ssot_preflight_sweep = is_ssot_preflight_sweep(cfg)
    track_learning_signal_sweep = track_planet_flow_sweep or track_ssot_preflight_sweep
    win_rate_trend = WinRateTrendTracker() if track_learning_signal_sweep else None
    approx_kl_window = MetricWindowTracker() if track_learning_signal_sweep else None
    entropy_window = MetricWindowTracker() if track_learning_signal_sweep else None
    planet_flow_unreachable_ceiling = (
        planet_flow_max_post_mask_unreachable_rate(
            load_thresholds(
                default_calibration_json_path(Path(__file__).resolve().parents[3])
            )
        )
        if track_planet_flow_sweep
        else None
    )
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
    checkpoint_handler = CheckpointHandler(
        cfg=cfg,
        run_dir=run_dir,
        log_path=log_path,
        run_context=run_context,
        telemetry=telemetry,
        artifact_queue_dir=artifact_queue_dir,
        checkpoint_pipeline=checkpoint_pipeline,
    )

    wandb_status = "on" if cfg.telemetry.wandb.enabled else "off"
    print(
        f"JAX training starting: campaign={run_context.campaign_slug} "
        f"run_id={run_context.run_id} updates={start_update}-"
        f"{cfg.training.total_updates} log_every={cfg.training.log_every} "
        f"wandb={wandb_status} log={log_path}",
        flush=True,
    )
    if not cfg.telemetry.wandb.enabled:
        print(
            "Terminal progress: one line per log_every update(s). "
            "First update may stall during JAX compile.",
            flush=True,
        )

    completed_training = False
    close_error: Exception | None = None
    try:
        for update in range(start_update, cfg.training.total_updates + 1):
            if checkpoint_pipeline is not None:
                checkpoint_handler.handle_results(checkpoint_pipeline.drain_results())
            update_start = time.perf_counter()
            reseed_events: list[dict[str, object]] = []
            rollout_start = time.perf_counter()
            transitions_by_group: list[JaxTransitionBatch] = []
            rollout_metrics_by_group: list[dict[str, jax.Array]] = []
            format_rollout_stats = empty_per_format_rollout_stats()
            next_groups: list[JaxRolloutGroup] = []
            should_reseed, reseed_reason = seed_scheduler.should_reseed(update)
            if should_reseed:
                reseed_event = seed_scheduler.reseed(update, reseed_reason)
                key = jax.random.PRNGKey(reseed_event.new_seed)
                key, rollout_groups = reset_rollout_groups_envs(key, rollout_groups)
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
            active_indices = active_group_indices(
                rollout_groups,
                curriculum.current_format_weights(),
                update=update,
                rotate_format_rollouts=cfg.training.rotate_format_rollouts,
            )
            key, *rollout_keys = jax.random.split(key, len(active_indices) + 1)
            for group_idx, rollout_key in zip(
                active_indices, rollout_keys, strict=True
            ):
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
                    replace_rollout_group_state(group, env_state, turn_batch)
                )
                transitions_by_group.append(transitions)
                rollout_metrics_by_group.append(rollout_metrics)
            merged_groups = list(rollout_groups)
            for group_idx, updated_group in zip(
                active_indices, next_groups, strict=True
            ):
                merged_groups[group_idx] = updated_group
            rollout_groups = merged_groups
            transitions = concatenate_transition_batches(transitions_by_group)
            if norm_state is not None and cfg.model.normalize_observations:
                ppo_transitions = normalize_transition_batch(
                    transitions, norm_state, cfg.model
                )
            else:
                ppo_transitions = transitions
            rollout_metrics = prune_merged_rollout_metrics(
                sum_metric_dicts(rollout_metrics_by_group),
                cfg,
            )
            rollout_scalar_keys = tuple(
                dict.fromkeys(
                    (
                        *required_rollout_scalar_names(cfg),
                        *FINALIZED_ROLLOUT_RATE_KEYS,
                        cfg.training.plateau_metric,
                    )
                )
            )
            rollout_scalar_values = jnp.asarray(
                [rollout_metrics.get(key, 0.0) for key in rollout_scalar_keys],
                dtype=jnp.float32,
            )
            # Intentional sync boundary: pull only registry-selected rollout scalars
            # (plus finalized cross-group rates and plateau metric) once per update.
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
            ppo_metric_names = required_ppo_metric_names(cfg, tuple(metrics.keys()))
            metric_values = jnp.asarray(
                [metrics[name] for name in ppo_metric_names], dtype=jnp.float32
            )
            metric_values_host = jax.device_get(metric_values)
            metrics_host = dict(
                zip(ppo_metric_names, metric_values_host.tolist(), strict=True)
            )
            ppo_seconds = time.perf_counter() - ppo_start
            update_seconds = time.perf_counter() - update_start
            per_format_timing_metrics = build_per_format_timing_metrics(
                format_rollout_stats,
                update_seconds=update_seconds,
                rollout_seconds=rollout_seconds,
                ppo_seconds=ppo_seconds,
                include_per_format="debug"
                in enabled_metric_groups(metric_groups_cfg_from_config(cfg)),
            )
            env_steps = int(rollout_scalars["env_steps"])
            episodes = int(rollout_scalars["episode_done"])
            win_rate_2p = float(rollout_scalars["win_rate_2p"])
            first_place_rate_4p = float(rollout_scalars["first_place_rate_4p"])
            average_placement_4p = float(rollout_scalars["average_placement_4p"])
            survival_time = float(rollout_scalars["survival_time"])
            score_share = float(rollout_scalars["score_share"])
            average_reward = float(rollout_scalars["average_reward"])
            episode_reward_mean = float(rollout_scalars["episode_reward_mean"])
            overall_win_rate = float(rollout_scalars["overall_win_rate"])
            total_env_steps += env_steps
            completed_episodes += episodes
            seed_scheduler.update_metric(
                float(rollout_scalars[cfg.training.plateau_metric])
            )
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
            if snapshot_due(cfg, update):
                historical_pool, snapshot_event = add_historical_snapshot(
                    historical_pool, train_state.params, update=update
                )
                snapshot_event.update(
                    historical_pool_snapshot_telemetry(historical_pool, update=update)
                )
                update_events.append(snapshot_event)
            phase_events = []
            planet_flow_sweep_metrics: dict[str, float] = {}
            if win_rate_trend is not None:
                if track_ssot_preflight_sweep:
                    planet_flow_sweep_metrics = collect_ssot_preflight_sweep_metrics(
                        win_rate_trend=win_rate_trend,
                        approx_kl_window=approx_kl_window,
                        entropy_window=entropy_window,
                        overall_win_rate=overall_win_rate,
                        metrics_host=metrics_host,
                    )
                elif track_planet_flow_sweep and "action_decision" in enabled_metric_groups(
                    metric_groups_cfg_from_config(cfg)
                ):
                    planet_flow_sweep_metrics = collect_planet_flow_sweep_metrics(
                        win_rate_trend=win_rate_trend,
                        approx_kl_window=approx_kl_window,
                        entropy_window=entropy_window,
                        overall_win_rate=overall_win_rate,
                        metrics_host=metrics_host,
                        rollout_scalars=rollout_scalars,
                        max_post_mask_unreachable_rate=(
                            planet_flow_unreachable_ceiling
                            if planet_flow_unreachable_ceiling is not None
                            else 0.05
                        ),
                    )
            saved_checkpoint_path: Path | None = None
            checkpoint_every = int(cfg.artifacts.checkpoint_every)
            checkpoint_due = checkpoint_every > 0 and update % checkpoint_every == 0
            if checkpoint_due or update == cfg.training.total_updates:
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
                    saved_checkpoint_path = checkpoint_path
                    checkpoint_handler.handle_results(
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
                    job = checkpoint_handler.build_checkpoint_job(
                        update=update,
                        train_state=train_state,
                        key=key,
                        total_env_steps=total_env_steps,
                        completed_episodes=completed_episodes,
                        curriculum=curriculum,
                        historical_pool=historical_pool,
                        final=is_final,
                    )
                    checkpoint_results = checkpoint_pipeline.submit_checkpoint(job)
                    checkpoint_handler.handle_results(checkpoint_results)
                    saved_checkpoint_path = None
                    for result in checkpoint_results:
                        if result.committed and result.numbered_path is not None:
                            saved_checkpoint_path = result.numbered_path
                            break
                    if saved_checkpoint_path is None:
                        candidate = run_dir / f"jax_ckpt_{update:06d}.pkl"
                        if candidate.is_file():
                            saved_checkpoint_path = candidate
            bracket_metrics: dict[str, object] = {}
            if bracket_training_enabled(cfg):
                tick = bracket_training_tick(
                    cfg,
                    update=update,
                    total_env_steps=total_env_steps,
                    checkpoint_path=saved_checkpoint_path,
                    queue_dir=run_context.queue_dir,
                    output_root=Path(cfg.output.root),
                    result_root=run_context.evaluations_dir,
                )
                bracket_metrics = {
                    "bracket_training_phase": tick.phase,
                    "weak_config": tick.weak_config,
                }
            elif ssot_pipeline_enabled(cfg):
                ssot_tick = ssot_qualifier_tick(
                    cfg,
                    update=update,
                    total_env_steps=total_env_steps,
                    checkpoint_path=saved_checkpoint_path,
                    output_root=Path(cfg.output.root),
                )
                bracket_metrics = ssot_qualifier_telemetry(ssot_tick)
                if ssot_tick.events:
                    update_events = list(update_events)
                    update_events.extend(ssot_tick.events)
            record = build_update_record(
                update=update,
                total_env_steps=total_env_steps,
                completed_episodes=completed_episodes,
                rollout_samples=rollout_samples,
                rollout_scalars=rollout_scalars,
                metrics_host=metrics_host,
                update_seconds=update_seconds,
                rollout_seconds=rollout_seconds,
                ppo_seconds=ppo_seconds,
                train_start_time=train_start_time,
                per_format_timing_metrics=per_format_timing_metrics,
                curriculum_telemetry=curriculum_telemetry,
                reseed_events=reseed_events,
                update_events=update_events,
                historical_pool=historical_pool,
                gpu_update_metrics=(
                    gpu_tracker.sample_update_metrics()
                    if gpu_tracker is not None
                    else {}
                ),
                seed_scheduler_policy=seed_scheduler.next_seed_policy(update),
                plateau_metric=cfg.training.plateau_metric,
                cfg=cfg,
                planet_flow_sweep_metrics=planet_flow_sweep_metrics or None,
            )
            if bracket_metrics:
                record.update(bracket_metrics)
            write_filtered_update_records(
                log_path=log_path,
                debug_log_path=debug_log_path,
                record=record,
                cfg=cfg,
                telemetry=telemetry,
                update=update,
            )
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

        completed_training = True
    finally:
        if checkpoint_pipeline is not None:
            timeout_seconds = (
                artifact_cfg.final_flush_timeout_seconds
                if completed_training
                else artifact_cfg.exception_flush_timeout_seconds
            )
            try:
                checkpoint_handler.handle_results(
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
    if (
        checkpoint_handler.checkpoint_failures
        and artifact_cfg.fail_training_on_checkpoint_error
    ):
        first_failure = checkpoint_handler.first_failure()
        assert first_failure is not None
        raise ArtifactPipelineError(
            f"checkpoint worker failed at update {first_failure.update}: "
            f"{first_failure.error or first_failure.reason or first_failure.status}"
        )
    last_ckpt = run_dir / "jax_ckpt_last.pkl"
    ckpt_hint = str(last_ckpt) if last_ckpt.exists() else "none"
    print(
        f"orbit_train_complete updates={cfg.training.total_updates} "
        f"log_path={log_path} checkpoint={ckpt_hint}"
    )
    return log_path
