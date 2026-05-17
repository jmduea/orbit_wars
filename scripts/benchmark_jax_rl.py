"""Benchmark the end-to-end JAX Orbit Wars rollout + PPO stack."""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import jax

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import default_train_config_path, load_train_config  # noqa: E402
from src.jax_env import batched_reset  # noqa: E402
from src.jax_policy import build_jax_policy  # noqa: E402
from src.jax_ppo import collect_rollout_jax, init_train_state, ppo_update_jax  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(default_train_config_path()))
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = deepcopy(load_train_config(args.config))
    if args.num_envs is not None:
        cfg.ppo.num_envs = args.num_envs
    if args.rollout_steps is not None:
        cfg.ppo.rollout_steps = args.rollout_steps
    cfg.model.architecture = "mlp"
    key = jax.random.PRNGKey(cfg.seed)
    key, reset_key, policy_key = jax.random.split(key, 3)
    env_state, turn_batch = batched_reset(
        jax.random.split(reset_key, cfg.ppo.num_envs), cfg.env
    )
    policy = build_jax_policy(
        architecture=cfg.model.architecture,
        candidate_count=cfg.env.candidate_count,
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        attention_heads=cfg.model.attention_heads,
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
                "num_envs": cfg.ppo.num_envs,
                "rollout_steps": cfg.ppo.rollout_steps,
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
