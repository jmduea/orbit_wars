"""Shared terminal reward helpers for Python and JAX runtimes."""

from __future__ import annotations

import jax.numpy as jnp

import jax
from src.config import RewardConfig, TaskConfig


def terminal_reward_from_scores(
    scores: list[float],
    task_cfg: TaskConfig,
    reward_cfg: RewardConfig | None = None,
    learner_index: int = 0,
) -> dict[str, float]:
    reward_cfg = reward_cfg or RewardConfig()
    player_count = max(int(getattr(task_cfg, "player_count", len(scores))), 1)
    padded_scores = [0.0 for _ in range(player_count)]
    for index, score in enumerate(scores[:player_count]):
        padded_scores[index] = float(score)
    learner_index = min(max(int(learner_index), 0), player_count - 1)
    learner_score = padded_scores[learner_index]
    best_score = max(padded_scores) if padded_scores else 0.0
    rank = 1.0 + sum(score > learner_score for score in padded_scores)
    ties = sum(score == learner_score for score in padded_scores)
    placement = rank + (float(ties) - 1.0) * 0.5
    is_first = 1.0 if learner_score == best_score and learner_score > 0.0 else 0.0
    total_score = sum(padded_scores)
    score_share = learner_score / total_score if total_score > 0.0 else 0.0
    ranked_reward = (
        1.0 - 2.0 * (placement - 1.0) / (player_count - 1.0)
        if player_count > 1
        else 1.0
    )
    mode = reward_cfg.terminal_reward_mode.strip().lower()
    if mode == "binary_win":
        reward = 1.0 if is_first > 0.0 else -1.0
    elif mode == "ranked":
        reward = ranked_reward
    elif mode == "score_share":
        reward = score_share
    elif mode == "survival_plus_rank":
        reward = ranked_reward
    else:
        raise ValueError(
            "reward.terminal_reward_mode must be one of binary_win, ranked, "
            f"score_share, or survival_plus_rank; got {mode!r}."
        )
    return {
        "terminal_reward_unscaled": float(reward),
        "terminal_rank": float(rank),
        "terminal_placement": float(placement),
        "terminal_is_first": float(is_first),
        "terminal_score_share": float(score_share),
        "terminal_survival_time": 0.0,
        "terminal_ranked_reward": float(ranked_reward),
    }


def apply_early_terminal_reward_shaping(
    terminal_reward_value: float, step_index: int, reward_cfg: RewardConfig
) -> float:
    if not reward_cfg.early_terminal_reward_shaping_enabled:
        return float(terminal_reward_value)
    horizon = max(int(reward_cfg.early_terminal_reward_shaping_horizon), 1)
    step_number = max(int(step_index) + 1, 1)
    if step_number >= horizon:
        return float(terminal_reward_value)
    bonus_scale = (horizon - step_number) / float(horizon)
    return float(terminal_reward_value) * (1.0 + bonus_scale)


def apply_early_terminal_reward_shaping_jax(
    reward: jax.Array, step_index: jax.Array, cfg: RewardConfig
) -> jax.Array:
    if not cfg.early_terminal_reward_shaping_enabled:
        return reward
    horizon = jnp.asarray(
        max(int(cfg.early_terminal_reward_shaping_horizon), 1),
        dtype=jnp.float32,
    )
    step_number = jnp.maximum(step_index.astype(jnp.float32) + 1.0, 1.0)
    bonus = jnp.maximum(horizon - step_number, 0.0) / horizon
    return reward * (1.0 + bonus)
