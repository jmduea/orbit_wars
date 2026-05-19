from __future__ import annotations

import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp

from .config import TrainConfig
from .features import encode_turn, ship_count_for_bucket
from .jax_policy import build_jax_policy
from .opponents import build_opponent


def _seed_for_update(cfg: TrainConfig, update: int) -> int:
    policy = cfg.replay.seed_policy.strip().lower()
    if policy == "fixed":
        return int(cfg.seed)
    if policy == "update":
        return int(cfg.seed + update)
    raise ValueError(f"Unsupported replay.seed_policy: {cfg.replay.seed_policy!r}")


def _reward_to_result(reward: float) -> str:
    if reward > 0.0:
        return "win"
    if reward < 0.0:
        return "loss"
    return "draw"


def _build_jax_policy_actions(cfg: TrainConfig, checkpoint_path: Path):
    with checkpoint_path.open("rb") as file:
        checkpoint = pickle.load(file)
    if not isinstance(checkpoint, dict) or "params" not in checkpoint:
        raise ValueError(
            f"JAX checkpoint must contain a parameter payload: {checkpoint_path}"
        )

    params = checkpoint["params"]
    if isinstance(params, dict) and "params" in params and len(params) == 1:
        params = params["params"]
    policy = build_jax_policy(
        candidate_count=cfg.env.candidate_count,
        ship_bucket_count=cfg.env.ship_bucket_count,
        hidden_size=cfg.model.hidden_size,
        architecture=cfg.model.architecture,
        attention_heads=cfg.model.attention_heads,
        enable_gradient_checkpointing=cfg.ppo.enable_gradient_checkpointing,
    )

    def act(observation: object) -> list[list[float | int]]:
        batch = encode_turn(observation, cfg.env, env_index=0)
        if batch.self_features.shape[0] == 0:
            return []
        outputs = policy.apply(
            {"params": params},
            jnp.asarray(batch.self_features),
            jnp.asarray(batch.candidate_features),
            jnp.asarray(batch.global_features),
            jnp.asarray(batch.candidate_mask).astype(jnp.bool_),
        )
        target_indices = jax.device_get(jnp.argmax(outputs.target_logits, axis=-1))
        selected_ship_logits = jnp.take_along_axis(
            outputs.ship_logits,
            jnp.asarray(target_indices)[:, None, None].repeat(
                outputs.ship_logits.shape[-1], axis=-1
            ),
            axis=1,
        ).squeeze(axis=1)
        ship_buckets = jax.device_get(jnp.argmax(selected_ship_logits, axis=-1))

        moves: list[list[float | int]] = []
        for row_idx, context in enumerate(batch.contexts):
            target_idx = int(target_indices[row_idx])
            bucket_idx = int(ship_buckets[row_idx])
            if target_idx == 0 or bucket_idx == 0:
                continue
            if target_idx >= len(context.candidate_ids):
                continue
            if not context.candidate_mask[target_idx]:
                continue
            ships = ship_count_for_bucket(
                context.source_ships, bucket_idx, cfg.env.ship_bucket_count
            )
            if ships <= 0:
                continue
            moves.append(
                [context.source_id, float(context.target_angles[target_idx]), ships]
            )
        return moves

    return act


def maybe_write_jax_checkpoint_replay(
    cfg: TrainConfig,
    *,
    update: int,
    checkpoint_path: Path,
    log_path: Path,
) -> Path | None:
    if not cfg.replay.enabled:
        return None
    every_n = max(int(cfg.replay.every_n_checkpoints), 1)
    checkpoint_index = max(update // max(int(cfg.checkpoint_every), 1), 1)
    if checkpoint_index % every_n != 0 and update != cfg.ppo.total_updates:
        return None

    from kaggle_environments import make

    run_dir = checkpoint_path.parent
    replay_dir = run_dir / cfg.replay.output_dir
    replay_dir.mkdir(parents=True, exist_ok=True)
    seed = _seed_for_update(cfg, update)
    opponent_name = cfg.replay.opponent

    learner_act = _build_jax_policy_actions(cfg, checkpoint_path)
    opponent = build_opponent(opponent_name)

    env = make("orbit_wars", configuration={"seed": seed, "randomSeed": seed}, debug=False)
    env.reset(num_agents=2)
    states = env.step([[], []])
    step_count = 0
    while step_count < max(int(cfg.replay.max_steps), 1):
        learner_obs = states[0].observation if states[0] is not None else {}
        opp_obs = states[1].observation if states[1] is not None else {}
        joint_actions = [learner_act(learner_obs), opponent.act(opp_obs)]
        states = env.step(joint_actions)
        step_count += 1
        status = states[0].status if states and states[0] is not None else "DONE"
        if status != "ACTIVE":
            break
    reward_value = states[0].reward if states and states[0] is not None else 0.0
    reward = float(reward_value) if reward_value is not None else 0.0
    result = _reward_to_result(reward)

    html_path = replay_dir / f"replay_u{update:06d}.html"
    html_path.write_text(env.render(mode="html"), encoding="utf-8")
    metadata_path = replay_dir / f"replay_u{update:06d}.json"
    metadata_path.write_text(
        json.dumps(
            {
                "checkpoint_update": update,
                "checkpoint_path": str(checkpoint_path),
                "log_path": str(log_path),
                "seed": seed,
                "opponent": opponent_name,
                "result": result,
                "reward": reward,
                "html_path": str(html_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata_path
