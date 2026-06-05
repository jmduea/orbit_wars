"""Micro-benchmark: compare JAX env parity modes on reset+step throughput."""

from __future__ import annotations

import statistics
import time
from typing import Any

import jax.numpy as jnp

import jax
from src.config import TrainConfig
from src.config.schema import RewardConfig, TaskConfig
from src.game.shield_config import env_parity_mode
from src.jax.env import batched_reset, empty_action, step


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _task_cfg(**kwargs: Any) -> TaskConfig:
    base: dict[str, Any] = dict(
        candidate_count=6,
        ship_bucket_count=8,
        max_fleets=32,
        player_count=2,
    )
    base.update(kwargs)
    return TaskConfig(**base)


def _train_cfg(*, env_mode: str) -> TrainConfig:
    cfg = TrainConfig()
    cfg.task = _task_cfg(env_parity_mode=env_mode)
    cfg.reward = RewardConfig()
    return cfg


def _run_arm(
    cfg: TrainConfig,
    *,
    batch_size: int,
    steps_per_episode: int,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    noop = jax.vmap(lambda _: empty_action(cfg.task))(jnp.arange(batch_size))

    @jax.jit
    def episode(keys_in):
        state, _batch = batched_reset(keys_in, cfg.task)

        def body(carry, _):
            state_in = carry
            state_out, _result = jax.vmap(
                lambda s, learner, opponent: step(
                    s, learner, opponent, cfg.task, cfg.reward
                )
            )(state_in, noop, noop)
            return state_out, None

        state_out, _ = jax.lax.scan(
            body, state, jnp.arange(steps_per_episode, dtype=jnp.int32)
        )
        return state_out

    warmup_keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    for _ in range(warmup):
        episode(warmup_keys).game.step.block_until_ready()

    timings: list[float] = []
    for repeat_idx in range(repeats):
        run_keys = jax.random.split(
            jax.random.fold_in(jax.random.PRNGKey(1), repeat_idx), batch_size
        )
        start = time.perf_counter()
        out = episode(run_keys)
        out.game.step.block_until_ready()
        timings.append(time.perf_counter() - start)

    total_env_steps = batch_size * steps_per_episode
    median_s = _median(timings)
    return {
        "median_seconds": median_s,
        "env_steps_per_sec": total_env_steps / max(median_s, 1e-9),
        "env_steps": float(total_env_steps),
    }


def env_parity_ab_payload(
    *,
    arms: list[dict[str, object]],
    batch_size: int,
    steps_per_episode: int,
    warmup: int,
    repeats: int,
    commit_sha: str | None,
    jax_version: str,
    devices: list[str],
) -> dict[str, object]:
    baseline = next((a for a in arms if a["env_parity_mode"] == "legacy"), None)
    train_arm = next((a for a in arms if a["env_parity_mode"] == "train"), None)
    notes = [
        "legacy: pre-#188 comet-free step path; same JAX planet reset as train.",
        "train: post-#188 comet expire/advance without spawn (production default).",
        "kaggle: reference generate_planets + comet spawn (pure_callback; diagnostic).",
    ]
    deltas: dict[str, object] = {}
    if baseline and train_arm:
        legacy_sps = float(baseline["env_steps_per_sec"])
        train_sps = float(train_arm["env_steps_per_sec"])
        deltas["train_vs_legacy_pct"] = (
            100.0 * (train_sps - legacy_sps) / max(legacy_sps, 1e-9)
        )
    return {
        "benchmark": "env_parity_ab",
        "batch_size": batch_size,
        "steps_per_episode": steps_per_episode,
        "warmup": warmup,
        "repeats": repeats,
        "commit_sha": commit_sha,
        "jax_version": jax_version,
        "devices": devices,
        "notes": notes,
        "arms": arms,
        "deltas": deltas,
    }


def run_env_parity_ab_benchmark(
    *,
    batch_size: int = 32,
    steps_per_episode: int = 128,
    warmup: int = 2,
    repeats: int = 3,
    modes: tuple[str, ...] = ("legacy", "train", "kaggle"),
    commit_sha: str | None = None,
) -> dict[str, object]:
    import jax as jax_lib

    arm_rows: list[dict[str, object]] = []
    for mode in modes:
        cfg = _train_cfg(env_mode=mode)
        metrics = _run_arm(
            cfg,
            batch_size=batch_size,
            steps_per_episode=steps_per_episode,
            warmup=warmup,
            repeats=repeats,
        )
        arm_rows.append(
            {
                "label": mode,
                "env_parity_mode": env_parity_mode(cfg.task),
                **metrics,
            }
        )

    devices = [str(d) for d in jax_lib.devices()]
    return env_parity_ab_payload(
        arms=arm_rows,
        batch_size=batch_size,
        steps_per_episode=steps_per_episode,
        warmup=warmup,
        repeats=repeats,
        commit_sha=commit_sha,
        jax_version=jax_lib.__version__,
        devices=devices,
    )
