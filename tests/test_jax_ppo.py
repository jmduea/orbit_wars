import pickle
from types import SimpleNamespace

import jax
import pytest

from src.config import TrainConfig
from src.constants import MAX_PLANETS, MAX_STEPS
from src.jax_env import batched_reset
from src.jax_policy import build_jax_policy
from src.jax_ppo import collect_rollout_jax, init_train_state, ppo_update_jax
from src.jax_train import _sum_metric_dicts, init_rollout_groups


@pytest.mark.parametrize(
    "architecture", ["mlp", "attention", "transformer", "gnn_pointer"]
)
def test_end_to_end_jax_rollout_and_update_smoke(architecture: str):
    cfg = TrainConfig()
    cfg.model.architecture = architecture
    cfg.task.max_fleets = 16
    cfg.task.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.model.max_moves_k = 3
    cfg.training.num_envs = 2
    cfg.training.rollout_steps = 1
    reset_keys = jax.random.split(jax.random.PRNGKey(0), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(jax.random.PRNGKey(1), policy, cfg)
    _key, env_state, turn_batch, transitions, rollout_metrics = collect_rollout_jax(
        jax.random.PRNGKey(2), env_state, turn_batch, train_state, policy, cfg
    )
    next_train_state, metrics = ppo_update_jax(train_state, policy, transitions, cfg)

    assert transitions.self_features.shape[:3] == (
        cfg.training.rollout_steps,
        cfg.training.num_envs,
        MAX_PLANETS,
    )
    assert (
        float(rollout_metrics["env_steps"]) == cfg.training.rollout_steps * cfg.training.num_envs
    )
    assert "total_loss" in metrics
    assert "total_loss_2p" in metrics
    assert float(metrics["loss_sample_count_2p"]) > 0.0
    assert float(metrics["loss_sample_count_4p"]) == 0.0
    assert all(bool(jax.numpy.isfinite(value)) for value in metrics.values())
    assert next_train_state.params is not train_state.params


def test_rollout_microbatching_preserves_full_environment_axis():
    cfg = TrainConfig()
    cfg.task.max_fleets = 16
    cfg.task.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.training.num_envs = 4
    cfg.training.rollout_steps = 1
    cfg.training.rollout_microbatch_envs = 2
    cfg.opponents.mode.opponent = "random"
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(jax.random.PRNGKey(21), policy, cfg)
    _key, groups = init_rollout_groups(jax.random.PRNGKey(22), cfg, policy)
    group = groups[0]

    _key, env_state, turn_batch, transitions, rollout_metrics = group.collect_fn(
        jax.random.PRNGKey(23),
        group.env_state,
        group.turn_batch,
        train_state,
    )

    assert env_state.game.step.shape == (cfg.training.num_envs,)
    assert turn_batch.self_features.shape[:2] == (cfg.training.num_envs, MAX_PLANETS)
    assert transitions.self_features.shape[:3] == (
        cfg.training.rollout_steps,
        cfg.training.num_envs,
        MAX_PLANETS,
    )
    assert (
        float(rollout_metrics["env_steps"]) == cfg.training.rollout_steps * cfg.training.num_envs
    )


def test_rollout_microbatching_requires_even_environment_division():
    cfg = TrainConfig()
    cfg.training.num_envs = 3
    cfg.training.rollout_microbatch_envs = 2
    policy = build_jax_policy(cfg=cfg)

    with pytest.raises(ValueError, match="evenly divide"):
        init_rollout_groups(jax.random.PRNGKey(24), cfg, policy)


def test_rollout_metric_aggregation_recomputes_rate_metrics():
    first_chunk = _metric_chunk(
        episodes_2p=2.0,
        wins_2p=1.0,
        episodes_4p=2.0,
        first_places_4p=1.0,
        placement_4p_sum=4.0,
    )
    second_chunk = _metric_chunk(
        episodes_2p=2.0,
        wins_2p=2.0,
        episodes_4p=2.0,
        first_places_4p=1.0,
        placement_4p_sum=6.0,
    )

    metrics = _sum_metric_dicts([first_chunk, second_chunk])

    assert float(metrics["win_rate_2p"]) == pytest.approx(0.75)
    assert float(metrics["first_place_rate_4p"]) == pytest.approx(0.5)
    assert float(metrics["average_placement_4p"]) == pytest.approx(2.5)


def _metric_chunk(**overrides: float) -> dict[str, jax.Array]:
    values = {
        "env_steps": 1.0,
        "average_reward": 0.0,
        "episode_done": 0.0,
        "episode_reward_mean": 0.0,
        "valid_non_noop_targets_sum": 0.0,
        "valid_non_noop_target_rows": 0.0,
        "only_noop_rows": 0.0,
        "trajectory_shield_original_non_noop_count": 0.0,
        "trajectory_shield_legal_non_noop_count": 0.0,
        "wins_2p": 0.0,
        "episodes_2p": 0.0,
        "first_places_4p": 0.0,
        "episodes_4p": 0.0,
        "placement_4p_sum": 0.0,
        "decision_count": 0.0,
        "noop_count": 0.0,
        "friendly_target_count": 0.0,
        "enemy_target_count": 0.0,
        "neutral_target_count": 0.0,
        "survival_time_sum": 0.0,
        "score_share_sum": 0.0,
        "win_episode_rows": 0.0,
        "loss_episode_rows": 0.0,
        "non_noop_count": 0.0,
        "launched_ship_count": 0.0,
        "launched_ship_total": 0.0,
        "launched_ship_speed_total": 0.0,
        "won_planets_owned_total": 0.0,
        "lost_planets_owned_total": 0.0,
        "won_planets_lost_total": 0.0,
        "lost_planets_lost_total": 0.0,
        "won_planets_taken_total": 0.0,
        "lost_planets_taken_total": 0.0,
        "won_garrisoned_ships_per_planet_total": 0.0,
        "lost_garrisoned_ships_per_planet_total": 0.0,
        "won_planet_diff_total": 0.0,
        "lost_planet_diff_total": 0.0,
        "won_production_diff_total": 0.0,
        "lost_production_diff_total": 0.0,
    }
    values.update(overrides)
    values["episode_done"] = values["episodes_2p"] + values["episodes_4p"]
    return {key: jax.numpy.asarray(value) for key, value in values.items()}


def test_jax_action_builder_allows_fewer_fleet_slots_than_planets():
    from src.jax_ppo import build_action_from_batch, build_random_action_from_batch

    cfg = TrainConfig()
    cfg.task.max_fleets = 4
    cfg.task.candidate_count = 4
    cfg.training.num_envs = 2
    _env_state, turn_batch = batched_reset(
        jax.random.split(jax.random.PRNGKey(42), cfg.training.num_envs), cfg.task
    )
    target = jax.numpy.zeros((cfg.training.num_envs * MAX_PLANETS,), dtype=jax.numpy.int32)
    bucket = jax.numpy.zeros_like(target)

    action = build_action_from_batch(turn_batch, target, bucket, cfg)

    assert action.source_id.shape == (cfg.training.num_envs, cfg.task.max_fleets)
    assert action.valid.shape == (cfg.training.num_envs, cfg.task.max_fleets)

    random_action = build_random_action_from_batch(
        jax.random.PRNGKey(7), turn_batch, cfg
    )

    assert random_action.source_id.shape == (cfg.training.num_envs, cfg.task.max_fleets)
    assert random_action.valid.shape == (cfg.training.num_envs, cfg.task.max_fleets)


def test_jax_action_builder_emits_multiple_launch_slots_per_source():
    from src.jax_ppo import build_action_from_batch

    cfg = TrainConfig()
    cfg.task.max_fleets = 32
    cfg.task.candidate_count = 4
    cfg.training.num_envs = 1
    _env_state, turn_batch = batched_reset(
        jax.random.split(jax.random.PRNGKey(43), cfg.training.num_envs), cfg.task
    )
    target = jax.numpy.ones((cfg.training.num_envs * MAX_PLANETS, 3), dtype=jax.numpy.int32)
    bucket = jax.numpy.ones_like(target)

    action = build_action_from_batch(turn_batch, target, bucket, cfg)

    assert action.source_id.shape == (cfg.training.num_envs, cfg.task.max_fleets)
    assert action.valid.shape == (cfg.training.num_envs, cfg.task.max_fleets)
    assert action.source_id[0, 1] == action.source_id[0, 0]


def test_jax_action_builder_invalid_step_does_not_consume_later_ships():
    from src.jax_ppo import build_action_from_batch

    cfg = TrainConfig()
    cfg.task.max_fleets = MAX_PLANETS * 2
    cfg.task.candidate_count = 4
    cfg.task.ship_bucket_count = 4
    cfg.training.num_envs = 1
    _env_state, turn_batch = batched_reset(
        jax.random.split(jax.random.PRNGKey(44), cfg.training.num_envs), cfg.task
    )
    flat_decision = turn_batch.decision_mask.reshape(-1)
    row_idx = int(jax.numpy.argmax(flat_decision))
    source_ships = float(turn_batch.source_ships.reshape(-1)[row_idx])
    target = jax.numpy.zeros((cfg.training.num_envs * MAX_PLANETS, 2), dtype=jax.numpy.int32)
    bucket = jax.numpy.zeros_like(target)
    target = target.at[row_idx, 0].set(0)
    bucket = bucket.at[row_idx, 0].set(3)
    target = target.at[row_idx, 1].set(1)
    bucket = bucket.at[row_idx, 1].set(3)

    action = build_action_from_batch(turn_batch, target, bucket, cfg)

    fleet_slot = (row_idx % MAX_PLANETS) * 2 + 1
    assert bool(action.valid[0, fleet_slot])
    assert float(action.ships[0, fleet_slot]) == source_ships


def test_jax_checkpoint_roundtrip_restores_resume_metadata(tmp_path):
    from src.jax_train import load_jax_checkpoint, save_jax_checkpoint

    cfg = TrainConfig()
    cfg.task.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.model.max_moves_k = 3
    cfg.training.rollout_steps = 3
    cfg.training.num_envs = 2
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(jax.random.PRNGKey(5), policy, cfg)

    save_jax_checkpoint(
        tmp_path,
        7,
        train_state,
        cfg,
        key=jax.random.PRNGKey(9),
        total_env_steps=42,
        completed_episodes=3,
    )
    loaded_state, key, start_update, total_env_steps, completed_episodes = (
        load_jax_checkpoint(str(tmp_path / "jax_ckpt_000007.pkl"), train_state, cfg)
    )

    assert start_update == 8
    assert total_env_steps == 42
    assert completed_episodes == 3
    assert jax.numpy.array_equal(key, jax.random.PRNGKey(9))
    assert loaded_state.opt_state is not None


def test_jax_checkpoint_rejects_legacy_config_payload(tmp_path):
    from src.jax_train import load_jax_checkpoint

    cfg = TrainConfig()
    cfg.task.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(jax.random.PRNGKey(5), policy, cfg)
    checkpoint_path = tmp_path / "jax_ckpt_legacy.pkl"
    with checkpoint_path.open("wb") as file:
        pickle.dump(
            {
                "params": train_state.params,
                "config": SimpleNamespace(
                    env=SimpleNamespace(candidate_count=4),
                    ppo=SimpleNamespace(total_updates=1),
                ),
            },
            file,
        )

    with pytest.raises(ValueError, match="legacy config fields"):
        load_jax_checkpoint(str(checkpoint_path), train_state, cfg)


def test_collect_rollout_jax_supports_four_player_multi_player_step():
    cfg = TrainConfig()
    cfg.task.player_count = 4
    cfg.task.max_fleets = 16
    cfg.task.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.training.num_envs = 2
    cfg.training.rollout_steps = 1
    cfg.opponents.mode.opponent = "random"
    reset_keys = jax.random.split(jax.random.PRNGKey(10), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(jax.random.PRNGKey(11), policy, cfg)

    _key, env_state, turn_batch, transitions, rollout_metrics = collect_rollout_jax(
        jax.random.PRNGKey(12), env_state, turn_batch, train_state, policy, cfg
    )

    assert transitions.self_features.shape[:3] == (
        cfg.training.rollout_steps,
        cfg.training.num_envs,
        MAX_PLANETS,
    )
    assert transitions.decision_mask.shape == (
        cfg.training.rollout_steps,
        cfg.training.num_envs,
        MAX_PLANETS,
        cfg.model.max_moves_k,
    )
    assert (
        float(rollout_metrics["env_steps"]) == cfg.training.rollout_steps * cfg.training.num_envs
    )




def test_collect_rollout_jax_two_player_static_shapes():
    cfg = TrainConfig()
    cfg.task.player_count = 2
    cfg.task.max_fleets = 16
    cfg.task.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.model.max_moves_k = 3
    cfg.training.num_envs = 3
    cfg.training.rollout_steps = 1
    cfg.opponents.mode.opponent = "random"

    reset_keys = jax.random.split(jax.random.PRNGKey(60), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(jax.random.PRNGKey(61), policy, cfg)

    _key, _env_state, _turn_batch, transitions, metrics = collect_rollout_jax(
        jax.random.PRNGKey(62), env_state, turn_batch, train_state, policy, cfg
    )

    assert transitions.self_features.shape == (
        1,
        3,
        60,
        transitions.self_features.shape[-1],
    )
    assert transitions.decision_mask.shape == (1, 3, 60, cfg.model.max_moves_k)
    assert float(metrics["env_steps"]) == 3.0
def test_assign_learner_players_uses_env_index_and_episode_count():
    from src.jax_env import assign_learner_players

    cfg = TrainConfig()
    cfg.task.player_count = 4
    cfg.task.max_fleets = 16
    cfg.training.num_envs = 5
    reset_keys = jax.random.split(jax.random.PRNGKey(20), cfg.training.num_envs)
    env_state, _turn_batch = batched_reset(reset_keys, cfg.task)

    env_indices = jax.numpy.arange(cfg.training.num_envs, dtype=jax.numpy.int32)
    episode_counts = jax.numpy.array([0, 0, 1, 2, 3], dtype=jax.numpy.int32)
    env_state, turn_batch = assign_learner_players(
        env_state, env_indices, episode_counts, cfg.task, True
    )

    expected = (env_indices + episode_counts) % cfg.task.player_count
    assert jax.numpy.array_equal(env_state.learner_player, expected)
    assert jax.numpy.array_equal(env_state.episode_count, episode_counts)
    assert turn_batch.self_features.shape[:2] == (cfg.training.num_envs, MAX_PLANETS)


def test_collect_rollout_jax_rotates_learner_after_reset_done():
    from src.jax_env import assign_learner_players

    cfg = TrainConfig()
    cfg.task.player_count = 4
    cfg.task.max_fleets = 16
    cfg.task.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.model.max_moves_k = 3
    cfg.training.num_envs = 4
    cfg.training.rollout_steps = 1
    cfg.opponents.mode.opponent = "random"
    reset_keys = jax.random.split(jax.random.PRNGKey(30), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    env_indices = jax.numpy.arange(cfg.training.num_envs, dtype=jax.numpy.int32)
    episode_counts = jax.numpy.zeros((cfg.training.num_envs,), dtype=jax.numpy.int32)
    env_state, turn_batch = assign_learner_players(
        env_state, env_indices, episode_counts, cfg.task, cfg.opponents.mode.alternate_player_sides
    )
    terminal_step = jax.numpy.full(
        (cfg.training.num_envs,), MAX_STEPS - 3, dtype=jax.numpy.int32
    )
    env_state = env_state._replace(game=env_state.game._replace(step=terminal_step))
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(jax.random.PRNGKey(31), policy, cfg)

    _key, env_state, _turn_batch, _transitions, rollout_metrics = collect_rollout_jax(
        jax.random.PRNGKey(32), env_state, turn_batch, train_state, policy, cfg
    )

    expected_episode_counts = jax.numpy.ones((cfg.training.num_envs,), dtype=jax.numpy.int32)
    expected_players = (env_indices + expected_episode_counts) % cfg.task.player_count
    assert float(rollout_metrics["episode_done"]) == cfg.training.num_envs
    assert jax.numpy.array_equal(env_state.episode_count, expected_episode_counts)
    assert jax.numpy.array_equal(env_state.learner_player, expected_players)


def test_collect_rollout_jax_emits_training_scalar_metric_contract():
    from src.jax_train import _BASE_ROLLOUT_SCALAR_KEYS

    cfg = TrainConfig()
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.model.max_moves_k = 3
    cfg.task.candidate_count = 4
    cfg.task.max_fleets = 16
    cfg.training.num_envs = 2
    cfg.training.rollout_steps = 1
    cfg.opponents.mode.opponent = "random"

    reset_keys = jax.random.split(jax.random.PRNGKey(40), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(jax.random.PRNGKey(41), policy, cfg)

    _key, _env_state, _turn_batch, _transitions, rollout_metrics = collect_rollout_jax(
        jax.random.PRNGKey(42), env_state, turn_batch, train_state, policy, cfg
    )

    missing_keys = [
        key for key in _BASE_ROLLOUT_SCALAR_KEYS if key not in rollout_metrics
    ]
    assert missing_keys == []
    assert "avg_reward" not in _BASE_ROLLOUT_SCALAR_KEYS
    assert "episode_reward_sum" not in _BASE_ROLLOUT_SCALAR_KEYS


def test_collect_rollout_jax_logs_trajectory_shield_metrics_and_keeps_k_step_masks():
    cfg = TrainConfig()
    cfg.model.architecture = "gnn_pointer"
    cfg.model.max_moves_k = 3
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.task.candidate_count = 4
    cfg.task.max_fleets = 16
    cfg.training.num_envs = 2
    cfg.training.rollout_steps = 1
    cfg.opponents.mode.opponent = "random"

    reset_keys = jax.random.split(jax.random.PRNGKey(90), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(jax.random.PRNGKey(91), policy, cfg)

    _key, _env_state, _turn_batch, transitions, rollout_metrics = collect_rollout_jax(
        jax.random.PRNGKey(92), env_state, turn_batch, train_state, policy, cfg
    )

    assert transitions.decision_mask.shape[-1] == cfg.model.max_moves_k
    assert jax.numpy.array_equal(
        transitions.decision_mask[..., 0], transitions.decision_mask[..., 1]
    )
    assert transitions.ship_bucket_mask.shape[-3:] == (
        cfg.model.max_moves_k,
        cfg.task.candidate_count,
        cfg.task.ship_bucket_count,
    )
    assert "trajectory_shield_blocked_count" in rollout_metrics
    assert "trajectory_shield_fallback_noop_count" in rollout_metrics
    assert "trajectory_shield_legal_non_noop_count" in rollout_metrics
    assert "trajectory_shield_original_non_noop_count" in rollout_metrics
    assert "trajectory_shield_legal_non_noop_rate" in rollout_metrics
    assert 0.0 <= float(rollout_metrics["trajectory_shield_legal_non_noop_rate"]) <= 1.0


def test_jax_rollout_groups_collect_two_and_four_player_formats_under_jit():
    from src.jax_ppo import concatenate_transition_batches
    from src.jax_train import init_rollout_groups

    cfg = TrainConfig()
    cfg.task.max_fleets = 16
    cfg.task.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.training.num_envs = 4
    cfg.training.rollout_steps = 1
    cfg.opponents.mode.opponent = "random"
    cfg.format.rollout_groups = [
        {"name": "two_player", "player_count": 2, "num_envs": 2},
        {"name": "four_player", "player_count": 4, "num_envs": 2},
    ]
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(jax.random.PRNGKey(41), policy, cfg)
    _key, groups = init_rollout_groups(jax.random.PRNGKey(40), cfg, policy)

    transitions_by_group = []
    for index, group in enumerate(groups):
        _key, _env_state, _turn_batch, transitions, rollout_metrics = group.collect_fn(
            jax.random.PRNGKey(50 + index),
            group.env_state,
            group.turn_batch,
            train_state,
        )
        transitions_by_group.append(transitions)
        assert transitions.self_features.shape[:3] == (
            cfg.training.rollout_steps,
            group.cfg.training.num_envs,
            MAX_PLANETS,
        )
        assert (
            float(rollout_metrics["env_steps"])
            == cfg.training.rollout_steps * group.cfg.training.num_envs
        )

    combined = concatenate_transition_batches(transitions_by_group)

    assert [group.cfg.task.player_count for group in groups] == [2, 4]
    assert set(jax.numpy.unique(combined.player_count).tolist()) == {2, 4}
    assert combined.self_features.shape[:3] == (
        cfg.training.rollout_steps,
        4,
        MAX_PLANETS,
    )
    assert combined.decision_mask.shape == (
        cfg.training.rollout_steps,
        4,
        MAX_PLANETS,
        cfg.model.max_moves_k,
    )


def test_collect_rollout_jax_rotation_covers_all_player_ids_across_envs():
    from src.jax_env import assign_learner_players

    cfg = TrainConfig()
    cfg.task.player_count = 4
    cfg.task.max_fleets = 16
    cfg.task.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.training.num_envs = 4
    cfg.training.rollout_steps = 1
    cfg.opponents.mode.opponent = "random"

    env_indices = jax.numpy.arange(cfg.training.num_envs, dtype=jax.numpy.int32)
    reset_keys = jax.random.split(jax.random.PRNGKey(70), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    env_state, turn_batch = assign_learner_players(
        env_state,
        env_indices,
        jax.numpy.zeros((cfg.training.num_envs,), dtype=jax.numpy.int32),
        cfg.task,
        cfg.opponents.mode.alternate_player_sides,
    )
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(jax.random.PRNGKey(71), policy, cfg)

    _key, env_state, _turn_batch, _transitions, _metrics = collect_rollout_jax(
        jax.random.PRNGKey(72), env_state, turn_batch, train_state, policy, cfg
    )

    assert jax.numpy.array_equal(jax.numpy.sort(env_state.learner_player), jax.numpy.arange(4, dtype=jax.numpy.int32))


def test_ppo_update_jax_accepts_four_player_rollout_transitions():
    cfg = TrainConfig()
    cfg.task.player_count = 4
    cfg.task.max_fleets = 16
    cfg.task.candidate_count = 4
    cfg.model.value_head = "format_routed"
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.training.num_envs = 2
    cfg.training.rollout_steps = 1
    cfg.opponents.mode.opponent = "random"

    reset_keys = jax.random.split(jax.random.PRNGKey(80), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(jax.random.PRNGKey(81), policy, cfg)

    _key, _env_state, _turn_batch, transitions, _metrics = collect_rollout_jax(
        jax.random.PRNGKey(82), env_state, turn_batch, train_state, policy, cfg
    )
    next_train_state, metrics = ppo_update_jax(train_state, policy, transitions, cfg)

    assert "total_loss" in metrics
    assert "total_loss_4p" in metrics
    assert float(metrics["loss_sample_count_2p"]) == 0.0
    assert float(metrics["loss_sample_count_4p"]) > 0.0
    assert next_train_state.params is not train_state.params
