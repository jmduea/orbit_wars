from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import TrainConfig
from .normalization import ObservationNormalizer
from .opponents import SelfPlayOpponent, build_opponent


@dataclass(slots=True)
class ReplayResult:
    html_path: Path
    metadata_path: Path
    seed: int
    opponent: str
    result: str


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


def maybe_write_checkpoint_replay(
    cfg: TrainConfig,
    *,
    update: int,
    checkpoint_path: Path,
    policy: torch.nn.Module,
    normalizer: ObservationNormalizer | None,
    device: torch.device,
    log_path: Path,
) -> ReplayResult | None:
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

    learner = SelfPlayOpponent(cfg, device=device, deterministic=True)
    learner.sync_from(policy, normalizer)
    opponent = build_opponent(opponent_name, cfg=cfg, device=device)

    env = make("orbit_wars", configuration={"seed": seed, "randomSeed": seed}, debug=False)
    env.reset(num_agents=2)
    states = env.step([[], []])
    step_count = 0
    while step_count < max(int(cfg.replay.max_steps), 1):
        learner_obs = states[0].observation if states[0] is not None else {}
        opp_obs = states[1].observation if states[1] is not None else {}
        joint_actions = [learner.act(learner_obs), opponent.act(opp_obs)]
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
    return ReplayResult(
        html_path=html_path,
        metadata_path=metadata_path,
        seed=seed,
        opponent=opponent_name,
        result=result,
    )


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
    run_dir = checkpoint_path.parent
    replay_dir = run_dir / cfg.replay.output_dir
    replay_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = replay_dir / f"replay_u{update:06d}.json"
    metadata_path.write_text(
        json.dumps(
            {
                "checkpoint_update": update,
                "checkpoint_path": str(checkpoint_path),
                "log_path": str(log_path),
                "seed": _seed_for_update(cfg, update),
                "opponent": cfg.replay.opponent,
                "result": "unavailable",
                "reason": "JAX checkpoint replay currently requires a Torch policy export.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata_path
