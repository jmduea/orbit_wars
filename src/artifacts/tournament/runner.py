"""Kaggle-env match execution for tournament evaluation."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.artifacts.checkpoint_compat import (
    checkpoint_feature_metadata,
    load_checkpoint_payload,
    validate_checkpoint_config_compatibility,
)
from src.artifacts.timing import StepTimingBudget
from src.config import TrainConfig
from src.jax.policy import build_jax_policy
from src.jax.submission_runtime import (
    SubmissionReadyAgent,
    apply_feature_metadata_to_model_config,
    build_submission_ready_agent,
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
    if normalized in {"noop", "noop_only"}:
        return "noop"
    return normalized


def build_checkpoint_agent(
    cfg: TrainConfig,
    checkpoint_path: Path,
    *,
    act_seed: int = 0,
) -> SubmissionReadyAgent:
    """Build a JIT-warmed submission-ready agent from a JAX checkpoint."""

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
    cfg = apply_feature_metadata_to_model_config(
        cfg, checkpoint_feature_metadata(checkpoint)
    )
    policy = build_jax_policy(cfg=cfg)
    agent = build_submission_ready_agent(
        cfg,
        policy,
        {"params": params},
        act_seed=act_seed,
        deterministic=True,
        deterministic_eval=True,
    )
    agent.warmup()
    return agent


def build_baseline_agent(name: str) -> AgentActFn:
    opponent = build_opponent(normalize_baseline_name(name))

    def act(observation: object) -> list[list[float | int]]:
        return opponent.act(observation)

    return act


def _episode_reset(agent: AgentActFn | SubmissionReadyAgent | Any) -> None:
    if isinstance(agent, SubmissionReadyAgent):
        agent.reset_episode()
        return
    reset = getattr(agent, "reset_episode", None)
    if callable(reset):
        reset()


def _agent_action(
    agent: AgentActFn | SubmissionReadyAgent | Any, observation: object
) -> list[list[float | int]]:
    if isinstance(agent, SubmissionReadyAgent):
        return agent.act_fn(observation)
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
    agents: Sequence[AgentActFn | SubmissionReadyAgent | Any],
    max_steps: int,
    per_step_seconds: float = 1.0,
    overage_budget_seconds: float = 60.0,
) -> tuple[MatchOutcome, Any, dict[str, object]]:
    """Run one Kaggle-env episode and return outcome, env handle, and timing summary."""

    from kaggle_environments import make

    if len(agent_ids) != len(agents):
        raise ValueError("agent_ids and agents must have the same length.")
    num_agents = len(agent_ids)
    timing = StepTimingBudget(per_step_seconds, overage_budget_seconds)
    match_started = time.perf_counter()

    env = make(
        "orbit_wars",
        configuration={
            "seed": seed,
            "randomSeed": seed,
            "episodeSteps": int(max_steps),
        },
        debug=False,
    )
    for agent in agents:
        _episode_reset(agent)
    env.reset(num_agents=num_agents)
    states = env.step([[] for _ in range(num_agents)])
    step_count = 0
    while step_count < max(max_steps, 1):
        joint_actions: list[list[list[float | int]]] = []
        for index in range(num_agents):
            observation = states[index].observation if states[index] is not None else {}
            action_started = time.perf_counter()
            joint_actions.append(_agent_action(agents[index], observation))
            timing.record(time.perf_counter() - action_started)
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

    timing_summary: dict[str, object] = {
        **timing.summary(),
        "match_seconds": time.perf_counter() - match_started,
        "env_steps": step_count,
    }
    outcome = MatchOutcome(
        match_id=match_id,
        format_name=format_name,
        seed=seed,
        agent_ids=tuple(agent_ids),
        rewards=rewards,
        results=results,
        placements=_placements_from_rewards(agent_ids, rewards),
    )
    return outcome, env, timing_summary
