"""30-update paired benchmark with PPO + rollout scalar metrics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kaggle_runtime_env import (
    add_worker_cuda_library_path,
    isolate_worker_python_env,
    pin_jax_platform_from_kaggle,
)

pin_jax_platform_from_kaggle()
isolate_worker_python_env()
add_worker_cuda_library_path()

import os

if os.environ.get("KAGGLE_ACCELERATOR_ID", "").strip().lower().startswith("nvidia"):
    os.environ.pop("JAX_PLATFORM_NAME", None)
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"

import jax
import jax.numpy as jnp

from src.config import compose_hydra_train_config
from src.jax.benchmark import rollout_group_summary
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

METRIC_KEYS = ("policy_loss", "value_loss", "approx_kl", "entropy", "total_loss")
ROLLOUT_KEYS = ("mean_active_launches_per_turn",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=["model=transformer_factorized", "opponents=self_play_only", "seed=42"],
    )
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_jax_accelerator_backend()
    devices = [str(d) for d in jax.devices()]
    default_backend = jax.default_backend()
    cfg = compose_hydra_train_config(list(args.overrides))
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
    curriculum = CurriculumController(cfg.curriculum, cfg.opponents.snapshot)
    update_fn = jax.jit(lambda ts, tr: ppo_update_jax(ts, policy, tr, cfg))
    timings = []
    update_sums = {k: 0.0 for k in METRIC_KEYS}
    rollout_sums = {k: 0.0 for k in ROLLOUT_KEYS}
    measured = 0
    for iteration in range(args.warmup + args.updates):
        update = iteration + 1
        t0 = time.perf_counter()
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
        jax.block_until_ready(metrics_accum["total_loss"])
        elapsed = time.perf_counter() - t0
        if iteration >= args.warmup:
            measured += 1
            timings.append(elapsed)
            for mk in METRIC_KEYS:
                if mk in metrics_accum:
                    update_sums[mk] += float(jax.device_get(metrics_accum[mk]))
            for rk in ROLLOUT_KEYS:
                if rk in rollout_metrics:
                    rollout_sums[rk] += float(jax.device_get(rollout_metrics[rk]))
    total_seconds = sum(timings)
    payload = {
        "label": args.label,
        "jax_version": jax.__version__,
        "devices": devices,
        "default_backend": default_backend,
        "overrides": list(args.overrides),
        "updates": args.updates,
        "warmup": args.warmup,
        "measured_updates": measured,
        "num_envs": total_envs,
        "rollout_steps": int(cfg.training.rollout_steps),
        "seconds_total": total_seconds,
        "seconds_per_update_mean": total_seconds / max(measured, 1),
        "env_steps_per_sec": (measured * int(cfg.training.rollout_steps) * total_envs)
        / max(total_seconds, 1e-9),
    }
    for mk in METRIC_KEYS:
        payload[mk] = update_sums[mk] / max(measured, 1)
    for rk in ROLLOUT_KEYS:
        payload[rk] = rollout_sums[rk] / max(measured, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
