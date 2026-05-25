import jax
import pytest

from src.config import TrainConfig
from src.game.trajectory_shield import apply_trajectory_shield_to_turn_batch_v2
from src.jax.env import batched_reset
from src.jax.policy import build_jax_policy
from src.jax.rollout.collect import collect_rollout_jax
from src.jax.train_state import init_train_state
from src.opponents.jax_actions.builders_v2 import (
    build_opportunistic_action_from_edge_batch,
    build_sniper_action_from_edge_batch,
    build_turtle_action_from_edge_batch,
)
from src.training.curriculum import CurriculumController


def _v2_cfg(*, player_count: int = 2) -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.architecture = "gnn_pointer_v2"
    cfg.task.encoding_version = "v2"
    cfg.task.player_count = player_count
    cfg.task.candidate_count = 4
    cfg.task.max_fleets = 16
    cfg.model.hidden_size = 16
    cfg.model.gnn_k_neighbors = 3
    cfg.model.gnn_message_passing_layers = 1
    cfg.model.max_moves_k = 2
    cfg.training.num_envs = 2
    cfg.training.rollout_steps = 1
    cfg.opponents.mode.opponent = "self"
    cfg.opponents.self_play.enabled = True
    cfg.curriculum.enabled = True
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
    shielded = jax.vmap(
        lambda game_row, batch_row: apply_trajectory_shield_to_turn_batch_v2(
            game_row, batch_row, cfg.task
        )
    )(state.game, batch)
    from src.jax.policy_v2 import edge_action_count

    bucket_mask = shielded.ship_bucket_mask.reshape(
        1, edge_action_count(cfg.task), cfg.task.ship_bucket_count
    )
    action = builder(state.game, shielded.batch, cfg, bucket_mask)
    assert action.valid.shape == (1, cfg.task.max_fleets)
    assert bool(jax.numpy.any(action.valid) or jax.numpy.all(~action.valid))


@pytest.mark.jax
@pytest.mark.parametrize(
    ("family", "metric_key", "expected_total"),
    [
        ("nearest_sniper", "opponent_slots_nearest_sniper", 2.0),
        ("turtle", "opponent_slots_turtle", 2.0),
        ("opportunistic", "opponent_slots_opportunistic", 2.0),
    ],
)
def test_v2_two_player_self_play_scripted_family_slots(family, metric_key, expected_total):
    cfg = _v2_cfg(player_count=2)
    cfg.curriculum.stages = [{"id": family, "opponent_families": {family: 1.0}}]
    reset_keys = jax.random.split(jax.random.PRNGKey(10), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.PRNGKey(11), policy, cfg)
    _key, _env_state, _turn_batch, _transitions, metrics = collect_rollout_jax(
        jax.random.PRNGKey(12),
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
def test_v2_four_player_self_play_sniper_family_slots():
    cfg = _v2_cfg(player_count=4)
    cfg.curriculum.stages = [
        {"id": "nearest_sniper", "opponent_families": {"nearest_sniper": 1.0}}
    ]
    reset_keys = jax.random.split(jax.random.PRNGKey(20), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.PRNGKey(21), policy, cfg)
    _key, _env_state, _turn_batch, _transitions, metrics = collect_rollout_jax(
        jax.random.PRNGKey(22),
        env_state,
        turn_batch,
        train_state,
        policy,
        cfg,
        stage_view=_stage_view(cfg),
    )
    assert float(metrics["opponent_slots_total"]) == 6.0
    assert float(metrics["opponent_slots_nearest_sniper"]) == 6.0
