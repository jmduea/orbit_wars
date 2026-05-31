from __future__ import annotations

import json
import random
from pathlib import Path

from src.config import TrainConfig

from .tournament.runner import build_baseline_agent, build_checkpoint_agent, run_match


def _seed_for_update(cfg: TrainConfig, update: int) -> int:
    policy = cfg.artifacts.replay.seed_policy.strip().lower()
    if policy == "fixed":
        return int(cfg.seed)
    if policy == "update":
        return int(cfg.seed + update)
    raise ValueError(f"Unsupported replay.seed_policy: {cfg.artifacts.replay.seed_policy!r}")


def maybe_write_jax_checkpoint_replay(
    cfg: TrainConfig,
    *,
    update: int,
    checkpoint_path: Path,
    log_path: Path,
    output_dir: Path | None = None,
) -> Path | None:
    if not cfg.artifacts.replay.enabled:
        return None
    every_n = max(int(cfg.artifacts.replay.every_n_checkpoints), 1)
    checkpoint_index = max(update // max(int(cfg.artifacts.checkpoint_every), 1), 1)
    if checkpoint_index % every_n != 0 and update != cfg.training.total_updates:
        return None

    run_dir = checkpoint_path.parent
    replay_dir = output_dir or run_dir / cfg.artifacts.replay.output_dir
    replay_dir.mkdir(parents=True, exist_ok=True)
    seed = _seed_for_update(cfg, update)
    learner_act = build_checkpoint_agent(cfg, checkpoint_path, act_seed=seed)
    earlier_checkpoint = _pick_earlier_checkpoint(run_dir, update)
    scenarios = [
        {
            "name": "2p_random",
            "num_agents": 2,
            "opponents": [build_baseline_agent("random")],
        },
        {
            "name": "2p_sniper",
            "num_agents": 2,
            "opponents": [build_baseline_agent("sniper")],
        },
    ]
    if earlier_checkpoint is not None:
        scenarios.append(
            {
                "name": "2p_earlier_ckpt",
                "num_agents": 2,
                "opponents": [build_checkpoint_agent(cfg, earlier_checkpoint, act_seed=seed)],
                "extra": {"earlier_checkpoint": str(earlier_checkpoint)},
            }
        )
    opponents_4p: list[object] = [
        build_baseline_agent("random"),
        build_baseline_agent("sniper"),
    ]
    if earlier_checkpoint is not None:
        opponents_4p.append(
            build_checkpoint_agent(cfg, earlier_checkpoint, act_seed=seed)
        )
    else:
        opponents_4p.append(build_baseline_agent("random"))
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
        agents = [learner_act, *scenario["opponents"]]
        outcome, env, _timing = run_match(
            match_id=f"replay_u{update:06d}_{scenario['name']}",
            format_name=str(scenario["name"]),
            seed=env_seed,
            agent_ids=tuple(f"seat_{idx}" for idx in range(int(scenario["num_agents"]))),
            agents=agents,
            max_steps=int(cfg.artifacts.replay.max_steps),
        )

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
            "result": outcome.results["seat_0"],
            "reward": outcome.rewards["seat_0"],
            "html_path": str(html_path),
        }
        payload.update(scenario.get("extra", {}))
        metadata_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
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
