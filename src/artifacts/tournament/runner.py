"""Kaggle-env match execution for tournament evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax
from src.artifacts.checkpoint_compat import (
    load_checkpoint_payload,
    validate_checkpoint_config_compatibility,
)
from src.config import TrainConfig
from src.jax.features import encode_turn
from src.jax.policy import build_jax_policy
from src.jax.submission_runtime import (
    batch_game,
    batch_turn,
    jax_game_from_observation,
    moves_from_jax_action,
    select_runtime_shielded_policy_actions,
)
from src.opponents.runtime import build_opponent

from .types import AgentActFn, MatchOutcome


def reward_to_result(reward: float) -> str:
    if reward > 0.0:
        return "win"
    if reward < 0.0:
        return "loss"
    return "draw"


def normalize_baseline_name(name: str) -> str:
    """Map curriculum family ids to Python runtime opponent names."""

    normalized = name.strip().lower()
    if normalized in {"nearest_sniper", "sniper"}:
        return "sniper"
    return normalized


def build_checkpoint_agent(
    cfg: TrainConfig,
    checkpoint_path: Path,
    *,
    act_seed: int = 0,
) -> AgentActFn:
    """Build a deterministic checkpoint policy callable for Kaggle env."""

    checkpoint = load_checkpoint_payload(checkpoint_path)
    if not isinstance(checkpoint, dict) or "params" not in checkpoint:
        raise ValueError(
            f"JAX checkpoint must contain a parameter payload: {checkpoint_path}"
        )
    validate_checkpoint_config_compatibility(
        checkpoint, checkpoint_path=checkpoint_path
    )

    params = checkpoint["params"]
    if isinstance(params, dict) and "params" in params and len(params) == 1:
        params = params["params"]
    policy = build_jax_policy(cfg=cfg)

    def act(observation: object) -> list[list[float | int]]:
        game = jax_game_from_observation(
            observation, max_fleet_slots=int(cfg.task.max_fleets)
        )
        batch = encode_turn(game, cfg.task)
        action = select_runtime_shielded_policy_actions(
            jax.random.PRNGKey(int(act_seed)),
            policy,
            {"params": params},
            batch_game(game),
            batch_turn(batch),
            cfg,
            deterministic=True,
        )
        return moves_from_jax_action(action)

    return act


def build_baseline_agent(name: str) -> AgentActFn:
    opponent = build_opponent(normalize_baseline_name(name))

    def act(observation: object) -> list[list[float | int]]:
        return opponent.act(observation)

    return act


def _agent_action(agent: AgentActFn | Any, observation: object) -> list[list[float | int]]:
    if callable(agent):
        return agent(observation)
    return agent.act(observation)


def _placements_from_rewards(agent_ids: Sequence[str], rewards: dict[str, float]) -> dict[str, int]:
    ranked = sorted(
        agent_ids,
        key=lambda agent_id: (-rewards.get(agent_id, 0.0), agent_id),
    )
    return {agent_id: index + 1 for index, agent_id in enumerate(ranked)}


def run_match(
    *,
    match_id: str,
    format_name: str,
    seed: int,
    agent_ids: Sequence[str],
    agents: Sequence[AgentActFn | Any],
    max_steps: int,
) -> tuple[MatchOutcome, Any]:
    """Run one Kaggle-env episode and return outcome plus env handle."""

    from kaggle_environments import make

    if len(agent_ids) != len(agents):
        raise ValueError("agent_ids and agents must have the same length.")
    num_agents = len(agent_ids)
    env = make(
        "orbit_wars",
        configuration={"seed": seed, "randomSeed": seed},
        debug=False,
    )
    env.reset(num_agents=num_agents)
    states = env.step([[] for _ in range(num_agents)])
    step_count = 0
    while step_count < max(max_steps, 1):
        joint_actions: list[list[list[float | int]]] = []
        for index in range(num_agents):
            observation = states[index].observation if states[index] is not None else {}
            joint_actions.append(_agent_action(agents[index], observation))
        states = env.step(joint_actions)
        step_count += 1
        status = states[0].status if states and states[0] is not None else "DONE"
        if status != "ACTIVE":
            break

    rewards: dict[str, float] = {}
    results: dict[str, str] = {}
    for index, agent_id in enumerate(agent_ids):
        reward_value = states[index].reward if states[index] is not None else 0.0
        reward = float(reward_value) if reward_value is not None else 0.0
        rewards[agent_id] = reward
        results[agent_id] = reward_to_result(reward)

    outcome = MatchOutcome(
        match_id=match_id,
        format_name=format_name,
        seed=seed,
        agent_ids=tuple(agent_ids),
        rewards=rewards,
        results=results,
        placements=_placements_from_rewards(agent_ids, rewards),
    )
    return outcome, env
