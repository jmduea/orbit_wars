"""Benchmark the end-to-end JAX Orbit Wars rollout + PPO stack."""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import compose_hydra_train_config  # noqa: E402
from src.jax.device import ensure_cuda_jax_if_nvidia_present  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line options for end-to-end JAX RL benchmarking."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=["model=attention"],
        help="Hydra overrides for the benchmark config, such as model=attention training.total_updates=10.",
    )
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        choices=["mlp", "attention", "transformer"],
    )
    return parser.parse_args()


def main() -> None:
    """Run the JAX rollout/update benchmark and print aggregate metrics."""

    args = parse_args()
    ensure_cuda_jax_if_nvidia_present()

    import jax

    from src.jax.env import batched_reset
    from src.jax.policy import build_jax_policy
    from src.jax.ppo import collect_rollout_jax, init_train_state, ppo_update_jax

    cfg = deepcopy(compose_hydra_train_config(list(args.overrides)))
    if args.num_envs is not None:
        cfg.training.num_envs = args.num_envs
    if args.rollout_steps is not None:
        cfg.training.rollout_steps = args.rollout_steps
    if args.architecture is not None:
        cfg.model.architecture = args.architecture
    key = jax.random.PRNGKey(cfg.seed)
    key, reset_key, policy_key = jax.random.split(key, 3)
    env_state, turn_batch = batched_reset(
        jax.random.split(reset_key, cfg.training.num_envs), cfg.task
    )
    policy = build_jax_policy(
        cfg=cfg,
    )
    train_state = init_train_state(policy_key, policy, cfg)
    collect_fn = jax.jit(
        lambda rollout_key, state, batch, ts: collect_rollout_jax(
            rollout_key, state, batch, ts, policy, cfg
        )
    )
    update_fn = jax.jit(
        lambda ts, transitions: ppo_update_jax(ts, policy, transitions, cfg)
    )
    measurements: list[dict[str, float]] = []
    for iteration in range(args.warmup + args.updates):
        key, rollout_key = jax.random.split(key)
        start = time.perf_counter()
        key, env_state, turn_batch, transitions, rollout_metrics = collect_fn(
            rollout_key, env_state, turn_batch, train_state
        )
        train_state, update_metrics = update_fn(train_state, transitions)
        jax.block_until_ready(update_metrics["total_loss"])
        seconds = time.perf_counter() - start
        if iteration >= args.warmup:
            measurements.append(
                {
                    "seconds": seconds,
                    "env_steps": float(jax.device_get(rollout_metrics["env_steps"])),
                    "samples": float(jax.device_get(rollout_metrics["samples"])),
                }
            )
    total_seconds = sum(item["seconds"] for item in measurements)
    env_steps = sum(item["env_steps"] for item in measurements)
    samples = sum(item["samples"] for item in measurements)
    print(
        json.dumps(
            {
                "backend": "jax_rl",
                "num_envs": cfg.training.num_envs,
                "rollout_steps": cfg.training.rollout_steps,
                "updates": args.updates,
                "warmup": args.warmup,
                "seconds": total_seconds,
                "env_steps": int(env_steps),
                "samples": int(samples),
                "env_steps_per_sec": env_steps / max(total_seconds, 1e-9),
                "samples_per_sec": samples / max(total_seconds, 1e-9),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
