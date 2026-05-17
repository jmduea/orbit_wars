from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from .config import TrainConfig
from .jax_env import batched_reset
from .jax_policy import build_jax_policy
from .jax_ppo import collect_rollout_jax, init_train_state, ppo_update_jax


def run_jax_training(cfg: TrainConfig) -> None:
    """Run an end-to-end JAX training loop for the JAX environment backend.

    This path keeps environment state, feature encoding, action sampling, rollout
    storage, return/advantage computation, and PPO updates in JAX. Both the MLP
    and attention/transformer policy architectures are supported.
    """

    key = jax.random.PRNGKey(cfg.seed)
    key, reset_key, policy_key = jax.random.split(key, 3)
    reset_keys = jax.random.split(reset_key, cfg.ppo.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.env)
    policy = build_jax_policy(
        candidate_count=cfg.env.candidate_count,
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        architecture=cfg.model.architecture,
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
    save_dir = Path(cfg.save_dir)
    log_path = Path("artifacts/rl_template/logs") / f"{cfg.run_name}_jax.jsonl"
    total_env_steps = 0
    completed_episodes = 0
    train_start_time = time.perf_counter()

    for update in range(1, cfg.ppo.total_updates + 1):
        update_start = time.perf_counter()
        rollout_start = time.perf_counter()
        key, rollout_key = jax.random.split(key)
        key, env_state, turn_batch, transitions, rollout_metrics = collect_fn(
            rollout_key, env_state, turn_batch, train_state
        )
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
        env_steps = int(jax.device_get(rollout_metrics["env_steps"]))
        episodes = int(jax.device_get(rollout_metrics["episode_done"]))
        total_env_steps += env_steps
        completed_episodes += episodes
        record: dict[str, float | int] = {
            "update": update,
            "total_env_steps": total_env_steps,
            "completed_episodes": completed_episodes,
            "samples": int(rollout_samples),
            "update_seconds": update_seconds,
            "elapsed_seconds": time.perf_counter() - train_start_time,
            "rollout_seconds": rollout_seconds,
            "ppo_seconds": ppo_seconds,
            "env_steps_per_sec": env_steps / max(update_seconds, 1e-9),
            "rollout_env_steps_per_sec": env_steps / max(rollout_seconds, 1e-9),
            "samples_per_sec": rollout_samples / max(update_seconds, 1e-9),
            "ppo_samples_per_sec": rollout_samples / max(ppo_seconds, 1e-9),
            **{name: float(value) for name, value in metrics.items()},
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
            save_jax_checkpoint(save_dir, cfg.run_name, update, train_state.params, cfg)


def append_jsonl(path: Path, record: dict[str, float | int]) -> None:
    """Append a JSON metrics record to ``path``, creating parents as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


def save_jax_checkpoint(
    save_dir: Path, run_name: str, update: int, params: dict, cfg: TrainConfig
) -> None:
    """Persist the latest and update-numbered JAX checkpoint payloads."""

    import pickle

    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"update": update, "params": jax.device_get(params), "config": cfg}
    with (run_dir / "jax_ckpt_last.pkl").open("wb") as file:
        pickle.dump(payload, file)
    with (run_dir / f"jax_ckpt_{update:06d}.pkl").open("wb") as file:
        pickle.dump(payload, file)
