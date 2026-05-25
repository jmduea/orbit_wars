"""Benchmark joint-flat shield vs factorized per-step shield loop (M1 Phase 0 spike)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import compose_hydra_train_config  # noqa: E402
from src.jax.device import ensure_cuda_jax_if_nvidia_present  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[
            "model=gnn_pointer",
            "format=mix_2p_4p_8env",
            "training.rollout_steps=64",
            "training.minibatch_size=256",
            "training.rollout_microbatch_envs=8",
        ],
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "m1" / "shield_spike.json",
    )
    parser.add_argument("--gate-ratio", type=float, default=1.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_cuda_jax_if_nvidia_present()

    import jax
    import jax.numpy as jnp

    from src.features.registry import edge_k
    from src.game.constants import MAX_PLANETS
    from src.game.trajectory_shield import apply_trajectory_shield_to_turn_batch_v2
    from src.jax.env import batched_reset
    from src.opponents.jax_actions.builders import owned_planet_ships

    cfg = compose_hydra_train_config(list(args.overrides))
    env_count = cfg.training.num_envs
    max_moves_k = cfg.model.max_moves_k
    k = edge_k(cfg.task)

    key = jax.random.PRNGKey(cfg.seed)
    env_state, turn_batch = batched_reset(
        jax.random.split(key, env_count), cfg.task
    )

    def joint_flat_shield_once(game, batch, remaining):
        return apply_trajectory_shield_to_turn_batch_v2(
            game, batch, cfg.task, remaining_planet_ships=remaining
        )

    def factorized_step_shield(game, batch, remaining):
        def body(carry, _):
            remaining_ships = carry
            shielded = apply_trajectory_shield_to_turn_batch_v2(
                game, batch, cfg.task, remaining_planet_ships=remaining_ships
            )
            noop_idx = MAX_PLANETS * k
            flat_mask = jnp.concatenate(
                [
                    shielded.batch.edge_mask.reshape(MAX_PLANETS * k),
                    jnp.ones((1,), dtype=bool),
                ],
                axis=0,
            )
            logits = jnp.where(flat_mask, 0.0, jnp.finfo(jnp.float32).min)
            flat_idx = jnp.argmax(logits)
            src_row = flat_idx // k
            launched = jnp.where(
                (flat_idx < noop_idx) & (remaining_ships[src_row] > 0.0),
                remaining_ships[src_row] * 0.25,
                0.0,
            )
            remaining_ships = remaining_ships.at[src_row].set(
                jnp.maximum(remaining_ships[src_row] - launched, 0.0)
            )
            return remaining_ships, None

        return jax.lax.scan(body, remaining, None, length=max_moves_k)[0]

    @jax.jit
    def joint_flat_batch(game, batch, remaining):
        return jax.vmap(joint_flat_shield_once)(game, batch, remaining)

    @jax.jit
    def factorized_batch(game, batch, remaining):
        return jax.vmap(factorized_step_shield)(game, batch, remaining)

    remaining = owned_planet_ships(env_state.game)

    for _ in range(args.warmup):
        joint_flat_batch(env_state.game, turn_batch, remaining).diagnostics.blocked_count.block_until_ready()
        factorized_batch(env_state.game, turn_batch, remaining).block_until_ready()

    joint_times: list[float] = []
    factorized_times: list[float] = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        joint_flat_batch(env_state.game, turn_batch, remaining).diagnostics.blocked_count.block_until_ready()
        joint_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        factorized_batch(env_state.game, turn_batch, remaining).block_until_ready()
        factorized_times.append(time.perf_counter() - t0)

    joint_median = statistics.median(joint_times)
    factorized_median = statistics.median(factorized_times)
    ratio = factorized_median / max(joint_median, 1e-9)
    passed = ratio <= args.gate_ratio

    payload = {
        "config": {
            "num_envs": env_count,
            "max_moves_k": max_moves_k,
            "edge_k": k,
            "overrides": list(args.overrides),
        },
        "joint_flat_median_sec": joint_median,
        "factorized_per_step_median_sec": factorized_median,
        "ratio": ratio,
        "gate_ratio": args.gate_ratio,
        "passed": passed,
        "repeats": args.repeats,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(f"Spike failed gate: ratio {ratio:.3f} > {args.gate_ratio}")


if __name__ == "__main__":
    main()
