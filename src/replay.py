from __future__ import annotations

import json
import pickle
import random
from pathlib import Path

import jax
import jax.numpy as jnp

from .config import TrainConfig
from .features import encode_turn, ship_count_for_bucket
from .jax_policy import build_jax_policy
from .opponents import build_opponent


def _ensure_target_sequence(values: jax.Array) -> jax.Array:
    if values.ndim == 2:
        return values[:, None, :]
    return values


def _ensure_ship_sequence(values: jax.Array) -> jax.Array:
    if values.ndim == 3:
        return values[:, None, :, :]
    return values


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
        cfg=cfg,
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
        target_logits = _ensure_target_sequence(outputs.target_logits)
        ship_logits = _ensure_ship_sequence(outputs.ship_logits)
        target_indices = jax.device_get(jnp.argmax(target_logits, axis=-1))
        selected_ship_logits = jnp.take_along_axis(
            ship_logits,
            jnp.asarray(target_indices)[..., None, None].repeat(
                ship_logits.shape[-1], axis=-1
            ),
            axis=2,
        ).squeeze(axis=2)
        ship_buckets = jax.device_get(jnp.argmax(selected_ship_logits, axis=-1))

        moves: list[list[float | int]] = []
        for row_idx, context in enumerate(batch.contexts):
            remaining_ships = int(context.source_ships)
            for step_idx in range(target_indices.shape[1]):
                if len(moves) >= int(cfg.env.max_fleets):
                    break
                target_idx = int(target_indices[row_idx, step_idx])
                bucket_idx = int(ship_buckets[row_idx, step_idx])
                if target_idx == 0 or bucket_idx == 0:
                    continue
                if target_idx >= len(context.candidate_ids):
                    continue
                if not bool(context.candidate_mask[target_idx]):
                    continue
                ships = ship_count_for_bucket(
                    remaining_ships, bucket_idx, cfg.env.ship_bucket_count
                )
                if ships <= 0:
                    continue
                remaining_ships = max(0, remaining_ships - ships)
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
    learner_act = _build_jax_policy_actions(cfg, checkpoint_path)
    earlier_checkpoint = _pick_earlier_checkpoint(run_dir, update)
    scenarios = [
        {
            "name": "2p_random",
            "num_agents": 2,
            "opponents": [build_opponent("random")],
        },
        {
            "name": "2p_sniper",
            "num_agents": 2,
            "opponents": [build_opponent("sniper")],
        },
    ]
    if earlier_checkpoint is not None:
        scenarios.append(
            {
                "name": "2p_earlier_ckpt",
                "num_agents": 2,
                "opponents": [_build_jax_policy_actions(cfg, earlier_checkpoint)],
                "extra": {"earlier_checkpoint": str(earlier_checkpoint)},
            }
        )
    opponents_4p: list[object] = [
        build_opponent("random"),
        build_opponent("sniper"),
    ]
    if earlier_checkpoint is not None:
        opponents_4p.append(_build_jax_policy_actions(cfg, earlier_checkpoint))
    else:
        opponents_4p.append(build_opponent("random"))
    scenarios.append(
        {
            "name": "4p_mixed",
            "num_agents": 4,
            "opponents": opponents_4p,
            "extra": (
                {"earlier_checkpoint": str(earlier_checkpoint)}
                if earlier_checkpoint is not None
                else {"earlier_checkpoint": None, "earlier_checkpoint_fallback": "random"}
            ),
        }
    )

    metadata_path: Path | None = None
    for index, scenario in enumerate(scenarios):
        env_seed = seed + index
        env = make(
            "orbit_wars", configuration={"seed": env_seed, "randomSeed": env_seed}, debug=False
        )
        env.reset(num_agents=int(scenario["num_agents"]))
        states = env.step([[] for _ in range(int(scenario["num_agents"]))])
        step_count = 0
        while step_count < max(int(cfg.replay.max_steps), 1):
            learner_obs = states[0].observation if states[0] is not None else {}
            joint_actions: list[list[list[float | int]]] = [learner_act(learner_obs)]
            for opp_idx, opponent in enumerate(scenario["opponents"], start=1):
                opp_obs = states[opp_idx].observation if states[opp_idx] is not None else {}
                if callable(opponent):
                    joint_actions.append(opponent(opp_obs))
                else:
                    joint_actions.append(opponent.act(opp_obs))
            states = env.step(joint_actions)
            step_count += 1
            status = states[0].status if states and states[0] is not None else "DONE"
            if status != "ACTIVE":
                break

        reward_value = states[0].reward if states and states[0] is not None else 0.0
        reward = float(reward_value) if reward_value is not None else 0.0
        result = _reward_to_result(reward)

        replay_name = str(scenario["name"])
        html_path = replay_dir / f"replay_u{update:06d}_{replay_name}.html"
        html_path.write_text(env.render(mode="html"), encoding="utf-8")
        metadata_path = replay_dir / f"replay_u{update:06d}_{replay_name}.json"
        payload = {
            "checkpoint_update": update,
            "checkpoint_path": str(checkpoint_path),
            "log_path": str(log_path),
            "seed": env_seed,
            "scenario": replay_name,
            "num_agents": int(scenario["num_agents"]),
            "result": result,
            "reward": reward,
            "html_path": str(html_path),
        }
        payload.update(scenario.get("extra", {}))
        metadata_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata_path


def _pick_earlier_checkpoint(run_dir: Path, update: int) -> Path | None:
    candidates: list[Path] = []
    for path in run_dir.glob("jax_ckpt_*.pkl"):
        stem = path.stem
        if not stem.startswith("jax_ckpt_"):
            continue
        try:
            ckpt_update = int(stem.removeprefix("jax_ckpt_"))
        except ValueError:
            continue
        if ckpt_update < update:
            candidates.append(path)
    if not candidates:
        return None
    return random.choice(candidates)
