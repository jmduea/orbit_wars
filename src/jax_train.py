from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Callable

from .config import TrainConfig
from .jax_device import (
    configure_jax_platform_for_host,
    ensure_cuda_jax_if_nvidia_present,
)

configure_jax_platform_for_host()

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

    raw_groups = cfg.training_format.rollout_groups or cfg.ppo.rollout_groups
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
    collect_fn = jax.jit(
        lambda rollout_key, state, batch, ts, update_idx: collect_rollout_jax(
            rollout_key, state, batch, ts, policy, group_cfg, update=update_idx
        )
    )
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
        print(
            f"Resuming JAX training from {resume_checkpoint} at update {start_update}"
        )
    update_fn = jax.jit(
        lambda ts, transitions: ppo_update_jax(ts, policy, transitions, cfg)
    )
    save_dir = Path(cfg.save_dir)
    log_path = Path("artifacts/rl_template/logs") / f"{cfg.run_name}_jax.jsonl"
    train_start_time = time.perf_counter()

    for update in range(start_update, cfg.ppo.total_updates + 1):
        update_start = time.perf_counter()
        rollout_start = time.perf_counter()
        transitions_by_group: list[JaxTransitionBatch] = []
        rollout_metrics_by_group: list[dict[str, jax.Array]] = []
        next_groups: list[JaxRolloutGroup] = []
        key, *rollout_keys = jax.random.split(key, len(rollout_groups) + 1)
        for group, rollout_key in zip(rollout_groups, rollout_keys, strict=True):
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
        rollout_groups = next_groups
        transitions = concatenate_transition_batches(transitions_by_group)
        rollout_metrics = jax.tree.map(lambda *xs: sum(xs), *rollout_metrics_by_group)
        # Block once so timing reflects device work, not just dispatch.
        rollout_samples = float(jax.device_get(rollout_metrics["samples"]))
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
        metrics = jax.device_get(metrics)
        ppo_seconds = time.perf_counter() - ppo_start
        update_seconds = time.perf_counter() - update_start
        rollout_metrics_host = jax.device_get(rollout_metrics)
        env_steps = int(rollout_metrics_host["env_steps"])
        episodes = int(rollout_metrics_host["episode_done"])
        episodes_2p = float(rollout_metrics_host.get("episodes_2p", 0.0))
        episodes_4p = float(rollout_metrics_host.get("episodes_4p", 0.0))
        episode_count = float(rollout_metrics_host.get("episode_done", 0.0))
        win_rate_2p = (
            float(rollout_metrics_host.get("wins_2p", 0.0)) / episodes_2p
            if episodes_2p
            else 0.0
        )
        first_place_rate_4p = (
            float(rollout_metrics_host.get("first_places_4p", 0.0)) / episodes_4p
            if episodes_4p
            else 0.0
        )
        average_placement_4p = (
            float(rollout_metrics_host.get("placement_4p_sum", 0.0)) / episodes_4p
            if episodes_4p
            else 0.0
        )
        survival_time = (
            float(rollout_metrics_host.get("survival_time_sum", 0.0)) / episode_count
            if episode_count
            else 0.0
        )
        score_share = (
            float(rollout_metrics_host.get("score_share_sum", 0.0)) / episode_count
            if episode_count
            else 0.0
        )
        total_env_steps += env_steps
        completed_episodes += episodes
        record: dict[str, object] = {
            "update": update,
            "total_env_steps": total_env_steps,
            "completed_episodes": completed_episodes,
            "samples": int(rollout_samples),
            "win_rate_2p": win_rate_2p,
            "first_place_rate_4p": first_place_rate_4p,
            "average_placement_4p": average_placement_4p,
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
            **{name: float(value) for name, value in metrics.items()},
            "opponent_composition": {
                "latest": float(
                    rollout_metrics_host.get("opponent_current_slots", 0.0)
                ),
                "random": float(rollout_metrics_host.get("opponent_random_slots", 0.0)),
                "historical": float(
                    rollout_metrics_host.get("opponent_snapshot_slots", 0.0)
                ),
            },
        }
        append_jsonl(log_path, record)
        if update % cfg.log_every == 0:
            print(
                f"update={update} steps={total_env_steps} episodes={completed_episodes} "
                f"loss={record['total_loss']:.4f} sps={record['samples_per_sec']:.1f} "
                f"rollout_s={rollout_seconds:.3f} ppo_s={ppo_seconds:.3f} "
                f"entropy={record['entropy']:.3f}"
            )
        if update % cfg.checkpoint_every == 0 or update == cfg.ppo.total_updates:
            save_jax_checkpoint(
                save_dir,
                cfg.run_name,
                update,
                train_state,
                cfg,
                key=key,
                total_env_steps=total_env_steps,
                completed_episodes=completed_episodes,
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
    save_dir: Path,
    run_name: str,
    update: int,
    train_state: object,
    cfg: TrainConfig,
    *,
    key: jax.Array,
    total_env_steps: int,
    completed_episodes: int,
) -> None:
    """Persist the latest and update-numbered JAX checkpoint payloads."""

    import pickle

    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "update": update,
        "params": jax.device_get(train_state.params),
        "opt_state": jax.device_get(train_state.opt_state),
        "rng_key": jax.device_get(key),
        "config": cfg,
        "total_env_steps": total_env_steps,
        "completed_episodes": completed_episodes,
    }
    with (run_dir / "jax_ckpt_last.pkl").open("wb") as file:
        pickle.dump(payload, file)
    with (run_dir / f"jax_ckpt_{update:06d}.pkl").open("wb") as file:
        pickle.dump(payload, file)
