"""Session/module JAX warmup shared by rollout and PPO integration tests."""

from __future__ import annotations

from dataclasses import dataclass

import jax
from src.config import TrainConfig
from src.jax.env import batched_reset
from src.jax.policy import build_jax_policy
from src.jax.rollout.collect import collect_rollout_jax
from src.jax.train import init_train_state

_WARMUP_DONE: set[str] = set()


@dataclass(frozen=True, slots=True)
class RolloutWarmupProfile:
    """Small 2p planet-graph-transformer config used to prime XLA for rollout tests."""

    key: str = "planet_graph_transformer_2p"
    rollout_steps: int = 1
    num_envs: int = 2
    hidden_size: int = 16
    attention_heads: int = 2
    max_moves_k: int = 3


def default_rollout_warmup_cfg(
    profile: RolloutWarmupProfile | None = None,
) -> TrainConfig:
    profile = profile or RolloutWarmupProfile()
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.model.hidden_size = profile.hidden_size
    cfg.model.attention_heads = profile.attention_heads
    cfg.model.max_moves_k = profile.max_moves_k
    cfg.task.max_fleets = 16
    cfg.task.candidate_count = 4
    cfg.task.player_count = 2
    cfg.training.num_envs = profile.num_envs
    cfg.training.rollout_steps = profile.rollout_steps
    cfg.opponents.dispatch = "random"
    cfg.telemetry.metric_groups.opponent_composition = True
    return cfg


def warmup_rollout_compile(profile: RolloutWarmupProfile | None = None) -> None:
    """Run one collect→PPO path once per profile key to amortize JIT across a pytest module."""

    profile = profile or RolloutWarmupProfile()
    if profile.key in _WARMUP_DONE:
        return
    cfg = default_rollout_warmup_cfg(profile)
    reset_keys = jax.random.split(jax.random.PRNGKey(0), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(jax.random.PRNGKey(1), policy, cfg)
    collect_rollout_jax(
        jax.random.PRNGKey(2),
        env_state,
        turn_batch,
        train_state,
        policy,
        cfg,
    )
    _WARMUP_DONE.add(profile.key)


def reset_warmup_cache_for_tests() -> None:
    """Clear module warmup state (used by tests that need a fresh compile)."""

    _WARMUP_DONE.clear()
