"""Benchmark rollout throughput for Kaggle/Python and JAX Orbit Wars backends.

Examples:
    python scripts/benchmark_env.py --backend both --num-envs 4 --rollout-steps 16 --updates 3 --warmup 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import TrainConfig, default_train_config_path, load_train_config  # noqa: E402
from src.env import OrbitWarsEnv  # noqa: E402
from src.features import candidate_feature_dim, global_feature_dim, self_feature_dim  # noqa: E402
from src.normalization import ObservationNormalizer  # noqa: E402
from src.opponents import SelfPlayOpponent, SelfPlayOpponentPool, build_opponent  # noqa: E402
from src.policy import build_policy  # noqa: E402
from src.train import (  # noqa: E402
    JaxBatchedEnv,
    collect_rollout,
    make_jax_batched_env,
    resolve_device,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for environment throughput benchmarks."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(default_train_config_path()))
    parser.add_argument("--backend", choices=["jax", "kaggle", "both"], default="both")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-normalizer", action="store_true")
    return parser.parse_args()


def build_cfg(args: argparse.Namespace, backend: str) -> TrainConfig:
    """Load and override benchmark configuration for one backend."""

    cfg = load_train_config(args.config)
    cfg = deepcopy(cfg)
    cfg.env_backend = backend
    if args.num_envs is not None:
        cfg.ppo.num_envs = args.num_envs
    if args.rollout_steps is not None:
        cfg.ppo.rollout_steps = args.rollout_steps
    if args.device is not None:
        cfg.device = args.device
    return cfg


def make_policy(cfg: TrainConfig, device: torch.device) -> torch.nn.Module:
    """Create the Torch policy used to sample benchmark actions."""

    return build_policy(
        architecture=cfg.model.architecture,
        self_dim=self_feature_dim(),
        candidate_dim=candidate_feature_dim(),
        global_dim=global_feature_dim(),
        candidate_count=cfg.env.candidate_count,
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        attention_heads=cfg.model.attention_heads,
    ).to(device)


def make_rollout_state(
    cfg: TrainConfig, policy: torch.nn.Module, device: torch.device
) -> tuple[Any, Any, int, list[float]]:
    """Initialize environments, first batches, seeds, and reward trackers."""

    opponent = build_opponent(cfg.opponent, cfg=cfg, device=device)
    if isinstance(opponent, SelfPlayOpponent):
        opponent.sync_from(policy, None)
    elif isinstance(opponent, SelfPlayOpponentPool):
        opponent.sync_from(policy, None, update=0)

    next_seed = cfg.seed
    if cfg.env_backend == "jax":
        seeds = np.arange(next_seed, next_seed + cfg.ppo.num_envs, dtype=np.int64)
        envs = make_jax_batched_env(cfg, seeds)
        batches = envs
        next_seed += cfg.ppo.num_envs
        env_count = cfg.ppo.num_envs
    else:
        envs = [
            OrbitWarsEnv(cfg, opponent, env_index=idx)
            for idx in range(cfg.ppo.num_envs)
        ]
        batches = []
        for env in envs:
            batches.append(env.reset(seed=next_seed))
            next_seed += 1
        env_count = len(envs)
    return envs, batches, next_seed, [0.0 for _ in range(env_count)]


def run_backend(args: argparse.Namespace, backend: str) -> dict[str, float | int | str]:
    """Benchmark one environment backend and return aggregate throughput."""

    cfg = build_cfg(args, backend)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    policy = make_policy(cfg, device)
    policy.eval()
    normalizer = (
        None
        if args.no_normalizer
        else ObservationNormalizer(clip=cfg.model.obs_norm_clip)
    )
    envs, batches, next_seed, running_rewards = make_rollout_state(cfg, policy, device)

    measurements: list[dict[str, float]] = []
    total_iterations = args.warmup + args.updates
    for iteration in range(total_iterations):
        start = time.perf_counter()
        batch, batches, next_seed, stats = collect_rollout(
            envs,
            batches,
            policy,
            cfg,
            device,
            next_seed,
            normalizer,
            running_rewards,
        )
        # Keep the mutable JAX container current across benchmark iterations.
        if isinstance(batches, JaxBatchedEnv):
            envs = batches
        seconds = time.perf_counter() - start
        if iteration >= args.warmup:
            measurements.append(
                {
                    "seconds": seconds,
                    "env_steps": float(stats["env_steps"]),
                    "samples": float(batch.self_features.shape[0]),
                }
            )

    total_seconds = sum(item["seconds"] for item in measurements)
    total_env_steps = sum(item["env_steps"] for item in measurements)
    total_samples = sum(item["samples"] for item in measurements)
    return {
        "backend": backend,
        "num_envs": cfg.ppo.num_envs,
        "rollout_steps": cfg.ppo.rollout_steps,
        "updates": args.updates,
        "warmup": args.warmup,
        "seconds": total_seconds,
        "env_steps": int(total_env_steps),
        "samples": int(total_samples),
        "rollout_env_steps_per_sec": total_env_steps / max(total_seconds, 1e-9),
        "samples_per_sec": total_samples / max(total_seconds, 1e-9),
    }


def main() -> None:
    """Run selected backend benchmarks and print JSON results."""

    args = parse_args()
    backends = ["jax", "kaggle"] if args.backend == "both" else [args.backend]
    results = [run_backend(args, backend) for backend in backends]
    for result in results:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
