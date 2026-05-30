"""Production-aligned JAX rollout + PPO benchmarking helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping

from src.config import TrainConfig
from src.config.rollout_allocation import infer_static_format_weights


@dataclass(frozen=True, slots=True)
class ProductionBenchmarkResult:
    """Aggregate timing for the production training collect + PPO path."""

    seconds: float
    env_steps: int
    samples: int
    updates: int
    warmup: int
    rollout_steps: int
    rollout_microbatch_envs: int | None
    total_envs: int
    rollout_groups: tuple[Mapping[str, int | str], ...]


def rollout_group_summary(cfg: TrainConfig) -> tuple[Mapping[str, int | str], ...]:
    """Return resolved rollout group declarations for benchmark metadata."""

    from src.jax.train import _configured_rollout_groups

    return tuple(_configured_rollout_groups(cfg))


def run_production_benchmark(
    cfg: TrainConfig,
    *,
    warmup: int,
    updates: int,
) -> ProductionBenchmarkResult:
    """Benchmark one training update using the production rollout-group path."""

    import jax
    import jax.numpy as jnp

    from src.jax.device import ensure_jax_accelerator_backend
    from src.jax.policy import build_jax_policy
    from src.jax.ppo_update import concatenate_transition_batches, ppo_update_jax
    from src.jax.train import (
        _active_group_indices,
        _init_historical_snapshot_pool,
        _replace_rollout_group_state,
        _sum_metric_dicts,
        init_rollout_groups,
    )
    from src.jax.train_state import init_train_state
    from src.training.curriculum import CurriculumController

    ensure_jax_accelerator_backend()

    group_specs = rollout_group_summary(cfg)
    total_envs = sum(int(spec["num_envs"]) for spec in group_specs)

    key = jax.random.PRNGKey(cfg.seed)
    _, rollout_init_key, policy_key = jax.random.split(key, 3)
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(policy_key, policy, cfg)
    key, rollout_groups = init_rollout_groups(rollout_init_key, cfg, policy)
    historical_pool = _init_historical_snapshot_pool(
        train_state.params, cfg.opponents.snapshot.pool_size
    )
    curriculum = CurriculumController(
        cfg.curriculum,
        cfg.opponents.snapshot,
        static_format_weights=infer_static_format_weights(cfg),
    )
    update_fn = jax.jit(
        lambda ts, transitions: ppo_update_jax(ts, policy, transitions, cfg)
    )

    measurements: list[dict[str, float]] = []
    for iteration in range(warmup + updates):
        update = iteration + 1
        start = time.perf_counter()
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
        transitions_by_group = []
        rollout_metrics_by_group = []
        next_groups = []
        for group_idx, rollout_key in zip(active_indices, rollout_keys, strict=True):
            group = rollout_groups[group_idx]
            _, env_state, turn_batch, transitions, rollout_metrics = group.collect_fn(
                rollout_key,
                group.env_state,
                group.turn_batch,
                train_state,
                stage_view,
                historical_pool.params,
                jnp.asarray(update, dtype=jnp.int32),
            )
            next_groups.append(_replace_rollout_group_state(group, env_state, turn_batch))
            transitions_by_group.append(transitions)
            rollout_metrics_by_group.append(rollout_metrics)

        merged_groups = list(rollout_groups)
        for group_idx, updated_group in zip(active_indices, next_groups, strict=True):
            merged_groups[group_idx] = updated_group
        rollout_groups = merged_groups

        transitions = concatenate_transition_batches(transitions_by_group)
        rollout_metrics = _sum_metric_dicts(rollout_metrics_by_group)
        metrics_accum = None
        for _ in range(cfg.training.epochs):
            train_state, update_metrics = update_fn(train_state, transitions)
            metrics_accum = (
                update_metrics
                if metrics_accum is None
                else jax.tree.map(jnp.add, metrics_accum, update_metrics)
            )
        assert metrics_accum is not None
        jax.block_until_ready(metrics_accum["total_loss"])
        seconds = time.perf_counter() - start
        if iteration >= warmup:
            measurements.append(
                {
                    "seconds": seconds,
                    "env_steps": float(jax.device_get(rollout_metrics["env_steps"])),
                    "samples": float(jax.device_get(rollout_metrics["samples"])),
                }
            )

    total_seconds = sum(item["seconds"] for item in measurements)
    env_steps = int(sum(item["env_steps"] for item in measurements))
    samples = int(sum(item["samples"] for item in measurements))
    return ProductionBenchmarkResult(
        seconds=total_seconds,
        env_steps=env_steps,
        samples=samples,
        updates=updates,
        warmup=warmup,
        rollout_steps=int(cfg.training.rollout_steps),
        rollout_microbatch_envs=cfg.training.rollout_microbatch_envs,
        total_envs=total_envs,
        rollout_groups=group_specs,
    )


def production_benchmark_payload(result: ProductionBenchmarkResult) -> dict[str, object]:
    """Render JSON-serializable benchmark output for CLI and Kaggle workers."""

    return {
        "backend": "jax_rl_production",
        "num_envs": result.total_envs,
        "total_envs": result.total_envs,
        "rollout_groups": [dict(group) for group in result.rollout_groups],
        "rollout_steps": result.rollout_steps,
        "rollout_microbatch_envs": result.rollout_microbatch_envs,
        "updates": result.updates,
        "warmup": result.warmup,
        "seconds": result.seconds,
        "env_steps": result.env_steps,
        "samples": result.samples,
        "env_steps_per_sec": result.env_steps / max(result.seconds, 1e-9),
        "samples_per_sec": result.samples / max(result.seconds, 1e-9),
    }
