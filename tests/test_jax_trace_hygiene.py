"""JAX trace-tier hygiene: import boundaries and jit contract smokes."""

from __future__ import annotations

import ast
from pathlib import Path

import jax.numpy as jnp
import pytest

import jax
from src.config import RewardConfig, TrainConfig
from src.game.constants import MAX_PLANETS
from src.jax.env import (
    empty_action,
    make_batched_reset_fn,
    make_batched_reset_with_pool_fn,
    make_batched_step_fn,
    make_batched_step_multi_player_fn,
)
from src.jax.map_pool.bake import bake_one_entry, stack_entries
from src.jax.map_pool.load import map_pool_constants_from_numpy
from src.jax.policy import build_jax_policy
from src.jax.train import init_rollout_groups, init_train_state

ROOT = Path(__file__).resolve().parents[1]

TIER_A_CLEAN = (
    "src/jax/env.py",
    "src/jax/features.py",
    "src/jax/action_sampling.py",
    "src/jax/factored_sequence_scan.py",
    "src/jax/factored_decode_scan.py",
    "src/jax/planet_flow.py",
    "src/jax/action_codec.py",
)

# Frozen cross-layer imports — must not grow without an explicit plan to remove debt.
TIER_A_FROZEN_IMPORTS: dict[str, frozenset[str]] = {
    "src/jax/rollout/collect.py": frozenset(
        {
            "src.artifacts.checkpoint_compat",
            "src.telemetry.metric_registry",
        }
    ),
    "src/jax/ppo_update.py": frozenset(
        {
            "src.artifacts.checkpoint_compat",
            "src.telemetry.metric_registry",
        }
    ),
}

ALLOWED_ARTIFACTS_PREFIX = "src.artifacts.checkpoint_compat"


def _top_level_import_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _forbidden_cross_layer_imports(modules: set[str]) -> set[str]:
    bad: set[str] = set()
    for module in modules:
        if module.startswith("src.telemetry"):
            bad.add(module)
        if module.startswith("src.artifacts") and not module.startswith(
            ALLOWED_ARTIFACTS_PREFIX
        ):
            bad.add(module)
    return bad


@pytest.mark.parametrize("rel_path", TIER_A_CLEAN)
def test_tier_a_clean_modules_avoid_telemetry_and_artifacts(rel_path: str) -> None:
    path = ROOT / rel_path
    modules = _top_level_import_modules(path)
    bad = _forbidden_cross_layer_imports(modules)
    assert not bad, f"{rel_path} must not import {sorted(bad)}"


@pytest.mark.parametrize("rel_path", sorted(TIER_A_FROZEN_IMPORTS))
def test_tier_a_frozen_import_debt_unchanged(rel_path: str) -> None:
    path = ROOT / rel_path
    modules = _top_level_import_modules(path)
    observed = {
        module
        for module in modules
        if module.startswith("src.telemetry") or module.startswith("src.artifacts")
    }
    expected = TIER_A_FROZEN_IMPORTS[rel_path]
    assert observed == expected, (
        f"{rel_path}: telemetry/artifacts imports changed.\n"
        f"  observed: {sorted(observed)}\n"
        f"  frozen:   {sorted(expected)}\n"
        "Update TIER_A_FROZEN_IMPORTS only when deliberately refactoring hot-path imports."
    )


def test_tier_a_batched_env_reset_step_under_jit() -> None:
    cfg = TrainConfig()
    cfg.task.max_fleets = 16
    cfg.task.candidate_count = 4
    reward_cfg = RewardConfig()
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    action = empty_action(cfg.task)
    batched_action = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (2,) + jnp.asarray(x).shape), action
    )

    reset_fn = make_batched_reset_fn(cfg.task)
    step_fn = make_batched_step_fn(cfg.task, reward_cfg)

    states, batches = reset_fn(keys)
    assert states.game.planets.x.shape == (2, MAX_PLANETS)
    assert batches.planet_features.shape[0] == 2

    next_states, results = step_fn(states, batched_action, batched_action)
    assert next_states.game.step.shape == (2,)
    assert results.reward.shape == (2,)


def test_tier_a_batched_reset_with_pool_under_jit() -> None:
    entries = [bake_one_entry(seed) for seed in (0, 1, 2)]
    tiny_pool = map_pool_constants_from_numpy(stack_entries(entries))
    cfg = TrainConfig()
    cfg.task.max_fleets = 16
    cfg.task.candidate_count = 4
    cfg.task.map_pool_path = "unused"
    keys = jax.random.split(jax.random.PRNGKey(11), 2)
    map_ids = jnp.array([0, 1], dtype=jnp.int32)

    reset_fn = make_batched_reset_with_pool_fn(cfg.task, tiny_pool)
    states, batches = reset_fn(keys, map_ids)
    assert states.game.planets.x.shape == (2, MAX_PLANETS)
    assert batches.planet_features.shape[0] == 2


def test_tier_a_batched_step_multi_player_under_jit() -> None:
    cfg = TrainConfig()
    cfg.task.player_count = 4
    cfg.task.max_fleets = 16
    cfg.task.candidate_count = 4
    reward_cfg = RewardConfig()
    keys = jax.random.split(jax.random.PRNGKey(12), 2)
    action = empty_action(cfg.task)
    per_player_action = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (cfg.task.player_count,) + jnp.asarray(x).shape),
        action,
    )
    batched_action = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (2,) + jnp.asarray(x).shape), per_player_action
    )

    reset_fn = make_batched_reset_fn(cfg.task)
    step_fn = make_batched_step_multi_player_fn(cfg.task, reward_cfg)

    states, batches = reset_fn(keys)
    assert batches.planet_features.shape[0] == 2

    next_states, results = step_fn(states, batched_action)
    assert next_states.game.step.shape == (2,)
    assert results.reward.shape == (2,)


def test_tier_a_jitted_collect_fn_one_step_smoke() -> None:
    cfg = TrainConfig()
    cfg.task.max_fleets = 16
    cfg.task.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.training.num_envs = 2
    cfg.training.rollout_steps = 1
    cfg.opponents.dispatch = "random"
    cfg.training.format_weights = {2: 1.0}

    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(jax.random.PRNGKey(1), policy, cfg)
    _key, groups = init_rollout_groups(jax.random.PRNGKey(2), cfg, policy)
    assert len(groups) == 1
    group = groups[0]

    _key, _env_state, _turn_batch, transitions, rollout_metrics = group.collect_fn(
        jax.random.PRNGKey(3),
        group.env_state,
        group.turn_batch,
        train_state,
    )
    assert transitions.planet_features.shape[:3] == (
        cfg.training.rollout_steps,
        cfg.training.num_envs,
        MAX_PLANETS,
    )
    assert float(rollout_metrics["env_steps"]) == float(
        cfg.training.rollout_steps * cfg.training.num_envs
    )
