from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import logging
import time
from pathlib import Path
from typing import Callable

from .checkpoint_compat import (
    feature_metadata,
    validate_checkpoint_feature_compatibility,
)
from .config import TrainConfig
from .checkpoint_retention import prune_checkpoints
from .replay import maybe_write_jax_checkpoint_replay
from .seed_scheduler import SeedScheduleConfig, SeedScheduler
from .telemetry import build_telemetry
from .curriculum import CurriculumController
from .run_paths import resolve_run_paths
from .jax_device import (
    configure_jax_platform_for_host,
    ensure_cuda_jax_if_nvidia_present,
)

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


@dataclass(slots=True)
class JaxRolloutGroup:
    """State for one statically compiled JAX rollout format."""

    name: str
    cfg: TrainConfig
    env_state: JaxEnvState
    turn_batch: JaxTurnBatch
    collect_fn: Callable


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
        update_idx=jnp.asarray(0, dtype=jnp.int32),
    ):
        return collect_rollout_jax(
            rollout_key, state, batch, ts, policy, group_cfg, update=update_idx
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


def _active_group_indices(
    groups: list[JaxRolloutGroup], format_weights: dict[int, float]
) -> list[int]:
    active: list[int] = []
    for idx, group in enumerate(groups):
        player_count = int(group.cfg.env.player_count)
        if float(format_weights.get(player_count, 0.0)) > 0.0:
            active.append(idx)
    return active or list(range(len(groups)))


def run_jax_training(cfg: TrainConfig, resume_checkpoint: str | None = None) -> None:
    """Run an end-to-end JAX training loop for the JAX environment backend.

    This path keeps environment state, feature encoding, action sampling, rollout
    storage, return/advantage computation, and PPO updates in JAX. Mixed 2p/4p
    training uses Option A: each format owns its env state and jitted collector,
    then compatible transition batches are concatenated before PPO updates.
    """

    ensure_cuda_jax_if_nvidia_present()

    key = jax.random.PRNGKey(cfg.seed)
    key, rollout_init_key, policy_key = jax.random.split(key, 3)
    policy = build_jax_policy(
        candidate_count=cfg.env.candidate_count,
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        architecture=cfg.model.architecture,
        attention_heads=cfg.model.attention_heads,
        enable_gradient_checkpointing=cfg.ppo.enable_gradient_checkpointing,
    )
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
            "env_backend": cfg.env_backend,
            "rl_backend": cfg.rl_backend,
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
    curriculum = CurriculumController(cfg.training_format.phases)
    curriculum.apply(cfg)
    phase_events: list[dict[str, object]] = []
    train_start_time = time.perf_counter()

    for update in range(start_update, cfg.ppo.total_updates + 1):
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
        curriculum.apply(cfg)
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
        rollout_scalar_keys = (
            "samples",
            "env_steps",
            "episode_done",
            "avg_reward",
            "episode_reward_sum",
            "episodes_2p",
            "episodes_4p",
            "wins_2p",
            "first_places_4p",
            "placement_4p_sum",
            "survival_time_sum",
            "score_share_sum",
            "decision_count",
            "noop_count",
            "friendly_target_count",
            "enemy_target_count",
            "neutral_target_count",
            "overall_win_rate",
            "noop_percent",
            "friendly_target_percent",
            "enemy_target_percent",
            "neutral_target_percent",
            "opponent_current_slots",
            "opponent_random_slots",
            "opponent_snapshot_slots",
            "won_non_noop_actions_per_step",
            "lost_non_noop_actions_per_step",
            "won_avg_fleet_launch_size",
            "lost_avg_fleet_launch_size",
            "won_avg_planets_owned",
            "lost_avg_planets_owned",
            "won_avg_planets_lost",
            "lost_avg_planets_lost",
            "won_avg_planets_taken",
            "lost_avg_planets_taken",
            "won_avg_garrisoned_ships_per_planet",
            "lost_avg_garrisoned_ships_per_planet",
            "won_avg_planet_diff",
            "lost_avg_planet_diff",
            "won_avg_production_diff",
            "lost_avg_production_diff",
            "won_avg_launch_fleet_speed",
            "lost_avg_launch_fleet_speed",
            cfg.plateau_metric,
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
        metric_names = tuple(metrics.keys())
        metric_values = jnp.asarray([metrics[name] for name in metric_names])
        # Intentional sync boundary: perform a single compact host transfer for
        # PPO scalars and keep logging values identical.
        metric_values_host = jax.device_get(metric_values)
        metrics_host = dict(zip(metric_names, metric_values_host.tolist(), strict=True))
        ppo_seconds = time.perf_counter() - ppo_start
        update_seconds = time.perf_counter() - update_start
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
        average_reward = float(rollout_scalars["avg_reward"])
        average_episode_reward = (
            float(rollout_scalars["episode_reward_sum"]) / episode_count if episode_count else 0.0
        )
        overall_win_rate = (
            (float(rollout_scalars["wins_2p"]) + float(rollout_scalars["first_places_4p"]))
            / episode_count
            if episode_count
            else 0.0
        )
        decision_count = float(rollout_scalars["decision_count"])
        noop_percent = (
            (float(rollout_scalars["noop_count"]) / decision_count) * 100.0
            if decision_count
            else 0.0
        )
        friendly_target_percent = (
            (float(rollout_scalars["friendly_target_count"]) / decision_count) * 100.0
            if decision_count
            else 0.0
        )
        enemy_target_percent = (
            (float(rollout_scalars["enemy_target_count"]) / decision_count) * 100.0
            if decision_count
            else 0.0
        )
        neutral_target_percent = (
            (float(rollout_scalars["neutral_target_count"]) / decision_count) * 100.0
            if decision_count
            else 0.0
        )
        total_env_steps += env_steps
        completed_episodes += episodes
        seed_scheduler.update_metric(float(rollout_scalars[cfg.plateau_metric]))
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
            "noop_percent": noop_percent,
            "friendly_target_percent": friendly_target_percent,
            "enemy_target_percent": enemy_target_percent,
            "neutral_target_percent": neutral_target_percent,
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
            **{name: float(value) for name, value in metrics_host.items()},
            "won_non_noop_actions_per_step": float(rollout_scalars["won_non_noop_actions_per_step"]),
            "lost_non_noop_actions_per_step": float(rollout_scalars["lost_non_noop_actions_per_step"]),
            "won_avg_fleet_launch_size": float(rollout_scalars["won_avg_fleet_launch_size"]),
            "lost_avg_fleet_launch_size": float(rollout_scalars["lost_avg_fleet_launch_size"]),
            "won_avg_planets_owned": float(rollout_scalars["won_avg_planets_owned"]),
            "lost_avg_planets_owned": float(rollout_scalars["lost_avg_planets_owned"]),
            "won_avg_planets_lost": float(rollout_scalars["won_avg_planets_lost"]),
            "lost_avg_planets_lost": float(rollout_scalars["lost_avg_planets_lost"]),
            "won_avg_planets_taken": float(rollout_scalars["won_avg_planets_taken"]),
            "lost_avg_planets_taken": float(rollout_scalars["lost_avg_planets_taken"]),
            "won_avg_garrisoned_ships_per_planet": float(rollout_scalars["won_avg_garrisoned_ships_per_planet"]),
            "lost_avg_garrisoned_ships_per_planet": float(rollout_scalars["lost_avg_garrisoned_ships_per_planet"]),
            "won_avg_planet_diff": float(rollout_scalars["won_avg_planet_diff"]),
            "lost_avg_planet_diff": float(rollout_scalars["lost_avg_planet_diff"]),
            "won_avg_production_diff": float(rollout_scalars["won_avg_production_diff"]),
            "lost_avg_production_diff": float(rollout_scalars["lost_avg_production_diff"]),
            "won_avg_launch_fleet_speed": float(rollout_scalars["won_avg_launch_fleet_speed"]),
            "lost_avg_launch_fleet_speed": float(rollout_scalars["lost_avg_launch_fleet_speed"]),
            "opponent_composition": {
                "latest": float(rollout_scalars["opponent_current_slots"]),
                "random": float(rollout_scalars["opponent_random_slots"]),
                "historical": float(rollout_scalars["opponent_snapshot_slots"]),
            },
            "curriculum_phase_id": curriculum.current_phase_id(),
            "curriculum_phase_events": list(phase_events),
        }
        phase_events = []
        transition = curriculum.update(
            update,
            {
                "win_rate_2p": win_rate_2p,
                "first_place_rate_4p": first_place_rate_4p,
                "survival_time": survival_time,
                "score_share": score_share,
                "kl_stability": float(record.get("approx_kl", 0.0)),
            },
        )
        if transition is not None:
            phase_events.append(transition)
        append_jsonl(log_path, record)
        telemetry.log(record, step=update)
        if update % cfg.log_every == 0:
            print(
                f"update={update} steps={total_env_steps} episodes={completed_episodes} "
                f"loss={record['total_loss']:.4f} sps={record['samples_per_sec']:.1f} "
                f"rollout_s={rollout_seconds:.3f} ppo_s={ppo_seconds:.3f} "
                f"entropy={record['entropy']:.3f}"
            )
        if update % cfg.checkpoint_every == 0 or update == cfg.ppo.total_updates:
            checkpoint_path = save_jax_checkpoint(
                run_dir,
                update,
                train_state,
                cfg,
                key=key,
                total_env_steps=total_env_steps,
                completed_episodes=completed_episodes,
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
            )
            action_label = "would prune" if pruning.dry_run else "pruned"
            print(
                f"checkpoint retention: {action_label} {len(pruning.deleted)} files, "
                f"reclaimed_bytes={pruning.reclaimed_bytes}"
            )
            telemetry.log_checkpoint(checkpoint_path, update=update)
            replay_meta_path = maybe_write_jax_checkpoint_replay(
                cfg,
                update=update,
                checkpoint_path=checkpoint_path,
                log_path=log_path,
            )
            if replay_meta_path is not None:
                telemetry.log_artifact(
                    replay_meta_path,
                    name=f"replay-meta-u{update}",
                    artifact_type="replay_metadata",
                )

    telemetry.finish()


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
) -> Path:
    """Persist the latest and update-numbered JAX checkpoint payloads."""

    import pickle

    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "update": update,
        "params": jax.device_get(train_state.params),
        "opt_state": jax.device_get(train_state.opt_state),
        "rng_key": jax.device_get(key),
        "config": cfg,
        "feature_metadata": feature_metadata(cfg.env),
        "total_env_steps": total_env_steps,
        "completed_episodes": completed_episodes,
    }
    with (run_dir / "jax_ckpt_last.pkl").open("wb") as file:
        pickle.dump(payload, file)
    update_path = run_dir / f"jax_ckpt_{update:06d}.pkl"
    with update_path.open("wb") as file:
        pickle.dump(payload, file)
    return update_path
