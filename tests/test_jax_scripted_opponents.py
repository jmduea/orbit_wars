"""JAX scripted opponent action builders and opponent-slot metric tests (slow tier).

Ownership:
- Scripted family slot metrics (nearest_sniper, turtle, opportunistic) are asserted here.
- Random/latest/historical family slot patterns live in ``tests/test_curriculum.py``.
"""

import pytest

import jax
from src.config import TrainConfig
from src.jax.env import batched_reset
from src.jax.policy import build_jax_policy
from src.jax.rollout.collect import collect_rollout_jax
from src.jax.train import init_train_state
from src.opponents.jax_actions.builders import (
    build_opportunistic_action_from_edge_batch,
    build_sniper_action_from_edge_batch,
    build_turtle_action_from_edge_batch,
)
from src.opponents.jax_actions.sampling import (
    _random_edge_action,
    _scripted_edge_action,
)
from src.opponents.curriculum import CurriculumController


def _v2_cfg(*, player_count: int = 2) -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.task.player_count = player_count
    cfg.task.candidate_count = 4
    cfg.task.max_fleets = 16
    cfg.model.hidden_size = 16
    cfg.model.max_moves_k = 2
    cfg.training.num_envs = 2
    cfg.training.rollout_steps = 1
    cfg.opponents.dispatch = "self"
    cfg.opponents.self_play.enabled = True
    cfg.curriculum.enabled = True
    cfg.telemetry.metric_groups.opponent_composition = True
    return cfg


def _stage_view(cfg: TrainConfig, *, pool_size: int = 2):
    controller = CurriculumController(cfg.curriculum, cfg.opponents.snapshot)
    return controller.stage_view(
        1,
        snapshot_ids=jax.numpy.zeros((pool_size,), dtype=jax.numpy.int32),
        snapshot_valid_mask=jax.numpy.zeros((pool_size,), dtype=bool),
        snapshot_updates=jax.numpy.zeros((pool_size,), dtype=jax.numpy.int32),
    )


@pytest.mark.jax
@pytest.mark.parametrize(
    "builder",
    [
        build_sniper_action_from_edge_batch,
        build_turtle_action_from_edge_batch,
        build_opportunistic_action_from_edge_batch,
    ],
)
def test_edge_scripted_builders_emit_valid_actions(builder):
    cfg = _v2_cfg()
    state, batch = batched_reset(jax.random.split(jax.random.PRNGKey(0), 1), cfg.task)
    action = builder(state.game, batch, cfg)
    assert action.valid.shape == (1, cfg.task.max_fleets)
    assert bool(jax.numpy.any(action.valid) or jax.numpy.all(~action.valid))


@pytest.mark.jax
@pytest.mark.parametrize(
    ("family", "metric_key", "player_count", "expected_total"),
    [
        ("nearest_sniper", "opponent_slots_nearest_sniper", 2, 2.0),
        ("turtle", "opponent_slots_turtle", 2, 2.0),
        ("opportunistic", "opponent_slots_opportunistic", 2, 2.0),
        ("nearest_sniper", "opponent_slots_nearest_sniper", 4, 6.0),
    ],
)
def test_v2_self_play_scripted_family_slots(
    family: str,
    metric_key: str,
    player_count: int,
    expected_total: float,
) -> None:
    cfg = _v2_cfg(player_count=player_count)
    cfg.curriculum.stages = [{"id": family, "opponent_families": {family: 1.0}}]
    reset_keys = jax.random.split(
        jax.random.PRNGKey(10 + player_count), cfg.training.num_envs
    )
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.PRNGKey(11 + player_count), policy, cfg)
    _key, _env_state, _turn_batch, _transitions, metrics = collect_rollout_jax(
        jax.random.PRNGKey(12 + player_count),
        env_state,
        turn_batch,
        train_state,
        policy,
        cfg,
        stage_view=_stage_view(cfg),
    )
    assert float(metrics["opponent_slots_total"]) == expected_total
    assert float(metrics[metric_key]) == expected_total


@pytest.mark.jax
@pytest.mark.parametrize(
    ("family", "sampler", "seed"),
    [
        (
            "nearest_sniper",
            lambda key, game, batch, cfg: _scripted_edge_action(
                game, batch, cfg, build_sniper_action_from_edge_batch
            ),
            44,
        ),
        (
            "random",
            lambda key, game, batch, cfg: _random_edge_action(key, game, batch, cfg),
            10,
        ),
    ],
)
def test_baseline_opponents_launch_under_cheap_shield(
    family: str,
    sampler,
    seed: int,
) -> None:
    """Learner cheap shield must not noop scripted/random baseline emitters."""
    cfg = _v2_cfg()
    cfg.task.trajectory_shield_mode = "cheap"
    cfg.curriculum.stages = [{"id": family, "opponent_families": {family: 1.0}}]
    reset_keys = jax.random.split(jax.random.PRNGKey(seed), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.PRNGKey(seed + 1), policy, cfg)
    _key, _env_state, _turn_batch, _transitions, metrics = collect_rollout_jax(
        jax.random.PRNGKey(seed + 2),
        env_state,
        turn_batch,
        train_state,
        policy,
        cfg,
        stage_view=_stage_view(cfg),
    )
    metric_key = f"opponent_slots_{family}"
    assert float(metrics[metric_key]) > 0.0

    sample_key = jax.random.PRNGKey(seed + 3)
    action = sampler(sample_key, env_state.game, turn_batch, cfg)
    assert bool(jax.numpy.any(action.valid))
