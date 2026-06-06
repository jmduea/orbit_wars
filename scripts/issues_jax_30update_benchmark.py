"""30-update paired benchmark with PPO + rollout scalar metrics."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
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

from src.jax.device import configure_jax_runtime_for_host, nvidia_gpu_present

configure_jax_runtime_for_host()
if nvidia_gpu_present() or os.environ.get(
    "KAGGLE_ACCELERATOR_ID", ""
).strip().lower().startswith("nvidia"):
    os.environ.pop("JAX_PLATFORM_NAME", None)
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"

import jax.numpy as jnp

import jax
from src.benchmark.production import rollout_group_summary
from src.config import compose_hydra_train_config
from src.jax.device import ensure_jax_accelerator_backend
from src.jax.policy import build_jax_policy
from src.jax.ppo_update import concatenate_transition_batches, ppo_update_jax
from src.jax.train import init_rollout_groups, init_train_state
from src.jax.train.metrics import sum_metric_dicts
from src.jax.train.rollout_groups import (
    active_group_indices,
    replace_rollout_group_state,
)
from src.jax.train.snapshots import init_historical_snapshot_pool
from src.training.curriculum import CurriculumController

METRIC_KEYS = ("policy_loss", "value_loss", "approx_kl", "entropy", "total_loss")
ROLLOUT_KEYS = ("mean_active_launches_per_turn", "overall_win_rate")

# Workstation validation profile: no curriculum, pure self-play (stability gate).
WORKSTATION_VALIDATION_OVERRIDES = [
    "model=transformer_factorized",
    "training=2p4p_32_split",
    "training.rollout_steps=128",
    "training.epochs=2",
    "training.update_chunk_rows=2048",
    "training.enable_gradient_checkpointing=true",
    "training.lean_rollout_metrics=true",
    "opponents=self_play_only",
    "curriculum=off",
    "telemetry.wandb.enabled=false",
    "artifacts.artifact_pipeline.enabled=false",
    "seed=42",
]

DEFAULT_OVERRIDES = [
    "model=transformer_factorized",
    "opponents=self_play_only",
    "curriculum=off",
    "seed=42",
]


def _git_head_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _scalar_metric(metrics: dict, key: str) -> float | None:
    if key not in metrics:
        return None
    return float(jax.device_get(metrics[key]))


def _survival_time_mean(rollout_metrics: dict) -> float | None:
    episodes = _scalar_metric(rollout_metrics, "episode_done")
    survival_sum = _scalar_metric(rollout_metrics, "survival_time_sum")
    if episodes is None or survival_sum is None:
        return None
    return survival_sum / max(episodes, 1e-9)


def _update_snapshot(
    *,
    update: int,
    update_metrics: dict,
    rollout_metrics: dict,
    curriculum: CurriculumController,
) -> dict[str, float | str | None]:
    snapshot: dict[str, float | str | None] = {
        "update": update,
        "curriculum_stage_id": curriculum.current_stage_id(),
    }
    for key in METRIC_KEYS:
        snapshot[key] = _scalar_metric(update_metrics, key)
    for key in ROLLOUT_KEYS:
        snapshot[key] = _scalar_metric(rollout_metrics, key)
    snapshot["survival_time"] = _survival_time_mean(rollout_metrics)
    return snapshot


def _finite_metrics(snapshot: dict[str, float | str | None]) -> bool:
    for key, value in snapshot.items():
        if key in {"update", "curriculum_stage_id"}:
            continue
        if value is None:
            continue
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="Hydra overrides; with --preset, merged after the preset bundle.",
    )
    parser.add_argument(
        "--preset",
        choices=("validation",),
        default=None,
        help="Use a documented override bundle (validation = WORKSTATION_VALIDATION_OVERRIDES).",
    )
    parser.add_argument(
        "--tier",
        default="micro",
        help="Benchmark tier label (e.g. micro, workstation, cloud_stretch).",
    )
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--snapshot-updates",
        nargs="*",
        type=int,
        default=[],
        help="Record per-update metric snapshots at these global update indices.",
    )
    return parser.parse_args()


def _format_profile_name(overrides: list[str]) -> str | None:
    for override in overrides:
        if override.startswith("format="):
            return override.split("=", 1)[1]
        if override.startswith("training="):
            return override.split("=", 1)[1]
    return None


def main() -> None:
    args = parse_args()
    run_started = time.perf_counter()
    ensure_jax_accelerator_backend()
    devices = [str(d) for d in jax.devices()]
    default_backend = jax.default_backend()
    if args.preset == "validation":
        overrides = list(WORKSTATION_VALIDATION_OVERRIDES)
        if args.overrides:
            overrides.extend(args.overrides)
    elif args.overrides is not None:
        overrides = list(args.overrides)
    else:
        overrides = list(DEFAULT_OVERRIDES)
    cfg = compose_hydra_train_config(overrides)
    group_specs = rollout_group_summary(cfg)
    total_envs = sum(int(spec["num_envs"]) for spec in group_specs)
    key = jax.random.PRNGKey(cfg.seed)
    _, rollout_init_key, policy_key = jax.random.split(key, 3)
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(policy_key, policy, cfg)
    key, rollout_groups = init_rollout_groups(rollout_init_key, cfg, policy)
    historical_pool = init_historical_snapshot_pool(
        train_state.params, cfg.opponents.snapshot.pool_size
    )
    curriculum = CurriculumController(cfg.curriculum, cfg.opponents.snapshot)
    update_fn = jax.jit(lambda ts, tr: ppo_update_jax(ts, policy, tr, cfg))
    timings: list[float] = []
    rollout_timings: list[float] = []
    update_timings: list[float] = []
    update_sums = {k: 0.0 for k in METRIC_KEYS}
    rollout_sums = {k: 0.0 for k in ROLLOUT_KEYS}
    measured = 0
    compile_seconds_to_update_3: float | None = None
    snapshot_targets = set(args.snapshot_updates)
    snapshots: list[dict[str, float | str | None]] = []
    for iteration in range(args.warmup + args.updates):
        update = iteration + 1
        t0 = time.perf_counter()
        t_rollout = time.perf_counter()
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
            next_groups.append(
                replace_rollout_group_state(group, env_state, turn_batch)
            )
            transitions_by_group.append(transitions)
            rollout_metrics_by_group.append(rollout_metrics)
        merged_groups = list(rollout_groups)
        for group_idx, updated_group in zip(active_indices, next_groups, strict=True):
            merged_groups[group_idx] = updated_group
        rollout_groups = merged_groups
        transitions = concatenate_transition_batches(transitions_by_group)
        rollout_metrics = sum_metric_dicts(rollout_metrics_by_group)
        rollout_elapsed = time.perf_counter() - t_rollout
        t_update = time.perf_counter()
        metrics_accum = None
        for _ in range(cfg.training.epochs):
            train_state, update_metrics = update_fn(train_state, transitions)
            metrics_accum = (
                update_metrics
                if metrics_accum is None
                else jax.tree.map(jnp.add, metrics_accum, update_metrics)
            )
        jax.block_until_ready(metrics_accum["total_loss"])
        update_elapsed = time.perf_counter() - t_update
        elapsed = time.perf_counter() - t0
        if update == 3:
            compile_seconds_to_update_3 = time.perf_counter() - run_started
        if iteration >= args.warmup:
            measured += 1
            timings.append(elapsed)
            rollout_timings.append(rollout_elapsed)
            update_timings.append(update_elapsed)
            for mk in METRIC_KEYS:
                if mk in metrics_accum:
                    update_sums[mk] += float(jax.device_get(metrics_accum[mk]))
            for rk in ROLLOUT_KEYS:
                if rk in rollout_metrics:
                    rollout_sums[rk] += float(jax.device_get(rollout_metrics[rk]))
        if update in snapshot_targets and metrics_accum is not None:
            epoch_count = max(int(cfg.training.epochs), 1)
            avg_update_metrics = jax.tree.map(
                lambda value: value / epoch_count,
                metrics_accum,
            )
            snapshots.append(
                _update_snapshot(
                    update=update,
                    update_metrics=avg_update_metrics,
                    rollout_metrics=rollout_metrics,
                    curriculum=curriculum,
                )
            )
    total_seconds = sum(timings)
    rollout_seconds_total = sum(rollout_timings)
    update_seconds_total = sum(update_timings)
    payload = {
        "label": args.label,
        "commit_sha": _git_head_sha(),
        "tier": args.tier,
        "jax_version": jax.__version__,
        "devices": devices,
        "default_backend": default_backend,
        "overrides": overrides,
        "updates": args.updates,
        "warmup": args.warmup,
        "measured_updates": measured,
        "format": _format_profile_name(overrides),
        "num_envs": total_envs,
        "rollout_groups": group_specs,
        "rollout_steps": int(cfg.training.rollout_steps),
        "rollout_microbatch_envs": int(cfg.training.rollout_microbatch_envs),
        "compile_seconds_to_update_3": compile_seconds_to_update_3,
        "seconds_total": total_seconds,
        "seconds_per_update_mean": total_seconds / max(measured, 1),
        "rollout_seconds_mean": rollout_seconds_total / max(measured, 1),
        "update_seconds_mean": update_seconds_total / max(measured, 1),
        "rollout_seconds_total": rollout_seconds_total,
        "update_seconds_total": update_seconds_total,
        "env_steps_per_sec": (measured * int(cfg.training.rollout_steps) * total_envs)
        / max(total_seconds, 1e-9),
    }
    for mk in METRIC_KEYS:
        payload[mk] = update_sums[mk] / max(measured, 1)
    for rk in ROLLOUT_KEYS:
        payload[rk] = rollout_sums[rk] / max(measured, 1)
    if snapshots:
        payload["snapshots"] = snapshots
        payload["snapshots_all_finite"] = all(
            _finite_metrics(item) for item in snapshots
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
