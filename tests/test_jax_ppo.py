"""JAX PPO and rollout integration tests (slow tier unless listed in FAST_JAX_PPO_TESTS).

Ownership:
- ``test_end_to_end_jax_rollout_and_update_smoke`` is the canonical 2p collect→PPO smoke.
- ``test_ppo_update_jax_accepts_four_player_rollout_transitions`` is the canonical 4p
  collect→PPO smoke.
- Multi-update rollout health lives in ``tests/test_jax_rollout.py``.
- Mixed 2p/4p format rollout collection lives in
  ``test_jax_rollout_groups_collect_two_and_four_player_formats_under_jit``.
"""

import pickle
from types import SimpleNamespace

import pytest

import jax


def _configure_rollout_groups(cfg, groups):
    if not groups:
        cfg.training.format_weights = {int(cfg.task.player_count): 1.0}
        return
    active = [group for group in groups if int(group.get("num_envs", 0)) > 0]
    if len(active) == 1:
        group = active[0]
        cfg.training.num_envs = int(group["num_envs"])
        cfg.training.format_weights = {int(group["player_count"]): 1.0}
        return
    total = sum(int(group["num_envs"]) for group in active)
    cfg.training.num_envs = total
    cfg.training.rotate_format_rollouts = False
    cfg.training.format_weights = {
        int(group["player_count"]): int(group["num_envs"]) / float(total)
        for group in active
    }


from src.config import TrainConfig
from src.game.constants import MAX_PLANETS, MAX_STEPS
from src.jax.env import batched_reset
from src.jax.policy import build_jax_policy
from src.jax.ppo_update import ppo_update_jax
from src.jax.rollout.collect import collect_rollout_jax
from src.jax.train import init_rollout_groups, init_train_state
from src.jax.rollout.metrics import rollout_metrics
from src.jax.train.metrics import (
    finalize_cross_chunk_rate_metrics,
    merge_metric_dicts,
    sum_metric_dicts,
)


def test_end_to_end_jax_rollout_and_update_smoke():
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
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

    assert transitions.planet_features.shape[:3] == (
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
    assert turn_batch.planet_features.shape[:2] == (cfg.training.num_envs, MAX_PLANETS)
    assert transitions.planet_features.shape[:3] == (
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

    metrics = sum_metric_dicts([first_chunk, second_chunk])

    assert float(metrics["win_rate_2p"]) == pytest.approx(0.75)
    assert float(metrics["first_place_rate_4p"]) == pytest.approx(0.5)
    assert float(metrics["average_placement_4p"]) == pytest.approx(2.5)


def test_sum_metric_dicts_single_group_finalizes_win_rates() -> None:
    chunk = _metric_chunk(episodes_2p=4.0, wins_2p=3.0, episodes_4p=0.0, first_places_4p=0.0)
    metrics = sum_metric_dicts([chunk])
    assert float(metrics["overall_win_rate"]) == pytest.approx(0.75)
    assert float(metrics["win_rate_2p"]) == pytest.approx(0.75)


def test_rollout_metrics_binary_win_rate_matches_terminal_reward() -> None:
    """Preflight noop gates use binary terminal reward; win rate must follow it."""

    import jax.numpy as jnp

    cfg = TrainConfig()
    cfg.task.player_count = 2
    cfg.reward.terminal_reward_mode = "binary_win"
    cfg.reward.early_terminal_reward_shaping_enabled = False
    cfg.training.rollout_steps = 4
    data = {
        "done": jnp.array([0.0, 0.0, 0.0, 1.0]),
        "reward": jnp.array([0.0, 0.0, 0.0, 1.0]),
        "terminal_is_first": jnp.zeros(4, dtype=jnp.float32),
        "terminal_placement": jnp.zeros(4, dtype=jnp.float32),
        "terminal_survival_time": jnp.zeros(4, dtype=jnp.float32),
        "terminal_score_share": jnp.zeros(4, dtype=jnp.float32),
        "terminal_ship_differential": jnp.zeros(4, dtype=jnp.float32),
        "target_index": jnp.zeros((4, 1), dtype=jnp.float32),
    }
    metrics = rollout_metrics(data=data, cfg=cfg, env_count=1)
    finalized = finalize_cross_chunk_rate_metrics(dict(metrics))
    assert float(finalized["episode_reward_mean"]) == pytest.approx(1.0)
    assert float(finalized["overall_win_rate"]) == pytest.approx(1.0)


def test_merge_metric_dicts_skips_removed_action_target_metrics() -> None:
    lean_chunk = {
        "env_steps": jax.numpy.asarray(8.0),
        "average_reward": jax.numpy.asarray(0.5),
        "episode_done": jax.numpy.asarray(0.0),
        "episode_reward_mean": jax.numpy.asarray(0.0),
        "trajectory_shield_legal_non_noop_count": jax.numpy.asarray(0.0),
        "trajectory_shield_original_non_noop_count": jax.numpy.asarray(0.0),
    }

    merged = merge_metric_dicts([lean_chunk, lean_chunk])

    assert "valid_non_noop_targets_per_row" not in merged
    assert "only_noop_fraction" not in merged
    assert float(merged["trajectory_shield_legal_non_noop_rate"]) == 0.0


def test_merge_metric_dicts_defers_finalize_only_rates():
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

    merged = merge_metric_dicts([first_chunk, second_chunk])
    for key in (
        "win_rate_2p",
        "first_place_rate_4p",
        "average_placement_4p",
        "survival_time",
        "score_share",
    ):
        assert key not in merged

    finalized = finalize_cross_chunk_rate_metrics(dict(merged))
    assert float(finalized["win_rate_2p"]) == pytest.approx(0.75)


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
    from src.opponents.jax_actions.builders import (
        build_action_from_edge_batch,
        build_random_action_from_edge_batch,
    )

    cfg = TrainConfig()
    cfg.task.max_fleets = 4
    cfg.task.candidate_count = 4
    cfg.training.num_envs = 2
    env_state, turn_batch = batched_reset(
        jax.random.split(jax.random.PRNGKey(42), cfg.training.num_envs), cfg.task
    )
    target = jax.numpy.zeros((cfg.training.num_envs, 1), dtype=jax.numpy.int32)
    bucket = jax.numpy.zeros((cfg.training.num_envs, 1), dtype=jax.numpy.int32)

    action = build_action_from_edge_batch(
        env_state.game, turn_batch, target, bucket, cfg
    )

    assert action.source_id.shape == (cfg.training.num_envs, cfg.task.max_fleets)
    assert action.valid.shape == (cfg.training.num_envs, cfg.task.max_fleets)

    random_action = build_random_action_from_edge_batch(
        jax.random.PRNGKey(7), env_state.game, turn_batch, cfg
    )

    assert random_action.source_id.shape == (cfg.training.num_envs, cfg.task.max_fleets)
    assert random_action.valid.shape == (cfg.training.num_envs, cfg.task.max_fleets)


def test_jax_action_builder_emits_multiple_launch_slots_per_source():
    from src.features.registry import edge_k
    from src.opponents.jax_actions.builders import build_action_from_edge_batch

    cfg = TrainConfig()
    cfg.task.max_fleets = 32
    cfg.task.candidate_count = 4
    cfg.training.num_envs = 1
    env_state, turn_batch = batched_reset(
        jax.random.split(jax.random.PRNGKey(43), cfg.training.num_envs), cfg.task
    )
    k = edge_k(cfg.task)
    owned = env_state.game.planets.active & (
        env_state.game.planets.owner == env_state.game.player
    )
    src_row = int(jax.numpy.argmax(owned.astype(jax.numpy.int32)))
    flat_idx = src_row * k
    target = jax.numpy.full((cfg.training.num_envs, 3), flat_idx, dtype=jax.numpy.int32)
    bucket = jax.numpy.ones((cfg.training.num_envs, 3), dtype=jax.numpy.int32)

    action = build_action_from_edge_batch(
        env_state.game, turn_batch, target, bucket, cfg
    )

    assert action.source_id.shape == (cfg.training.num_envs, cfg.task.max_fleets)
    assert action.valid.shape == (cfg.training.num_envs, cfg.task.max_fleets)
    assert action.source_id[0, 1] == action.source_id[0, 0]


def test_jax_action_builder_invalid_step_does_not_consume_later_ships():
    from src.features.registry import edge_k
    from src.opponents.jax_actions.builders import (
        build_action_from_edge_batch,
        noop_edge_index,
        owned_planet_ships,
    )

    cfg = TrainConfig()
    cfg.task.max_fleets = MAX_PLANETS * 2
    cfg.task.candidate_count = 4
    cfg.task.ship_bucket_count = 4
    cfg.training.num_envs = 1
    env_state, turn_batch = batched_reset(
        jax.random.split(jax.random.PRNGKey(44), cfg.training.num_envs), cfg.task
    )
    k = edge_k(cfg.task)
    owned = env_state.game.planets.active[0] & (
        env_state.game.planets.owner[0] == env_state.game.player[0]
    )
    src_row = int(jax.numpy.argmax(owned.astype(jax.numpy.int32)))
    source_ships = float(owned_planet_ships(env_state.game)[0, src_row])
    noop_idx = noop_edge_index(cfg.task)
    valid_edge = src_row * k
    target = jax.numpy.zeros((cfg.training.num_envs, 2), dtype=jax.numpy.int32)
    bucket = jax.numpy.zeros((cfg.training.num_envs, 2), dtype=jax.numpy.int32)
    target = target.at[0, 0].set(noop_idx)
    bucket = bucket.at[0, 0].set(3)
    target = target.at[0, 1].set(valid_edge)
    bucket = bucket.at[0, 1].set(3)

    action = build_action_from_edge_batch(
        env_state.game, turn_batch, target, bucket, cfg
    )

    assert bool(action.valid[0, 1])
    assert float(action.ships[0, 1]) == source_ships


def test_jax_checkpoint_roundtrip_restores_resume_metadata(tmp_path):
    from src.jax.train import load_jax_checkpoint, save_jax_checkpoint

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
    from src.jax.train import load_jax_checkpoint

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

    assert transitions.planet_features.shape[:3] == (
        cfg.training.rollout_steps,
        cfg.training.num_envs,
        MAX_PLANETS,
    )
    assert transitions.target_index.shape == (
        cfg.training.rollout_steps,
        cfg.training.num_envs,
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

    assert transitions.planet_features.shape == (
        1,
        3,
        60,
        transitions.planet_features.shape[-1],
    )
    assert transitions.target_index.shape == (1, 3, cfg.model.max_moves_k)
    assert float(metrics["env_steps"]) == 3.0
def test_assign_learner_players_uses_env_index_and_episode_count():
    from src.jax.env import assign_learner_players

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
    assert turn_batch.planet_features.shape[:2] == (cfg.training.num_envs, MAX_PLANETS)


def test_collect_rollout_jax_rotates_learner_after_reset_done():
    from src.jax.env import assign_learner_players

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
    from src.jax.rollout.metric_contract import ROLLOUT_ALLOWED_SCALAR_KEYS
    from src.telemetry.metric_registry import rollout_compute_scalar_keys

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

    expected_keys = rollout_compute_scalar_keys(cfg)
    missing_keys = [key for key in expected_keys if key not in rollout_metrics]
    assert missing_keys == []
    disabled_keys = sorted(
        key for key in rollout_metrics if key not in expected_keys
    )
    assert disabled_keys == []
    unexpected_keys = sorted(
        key for key in rollout_metrics if key not in ROLLOUT_ALLOWED_SCALAR_KEYS
    )
    assert unexpected_keys == []


def test_collect_rollout_jax_logs_trajectory_shield_metrics_and_keeps_k_step_masks():
    from src.features.registry import edge_k
    from src.game.constants import MAX_PLANETS

    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.model.max_moves_k = 3
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.task.candidate_count = 4
    cfg.task.max_fleets = 16
    cfg.training.num_envs = 2
    cfg.training.rollout_steps = 1
    cfg.opponents.mode.opponent = "random"
    cfg.telemetry.metric_groups.trajectory_shield_debug = True

    reset_keys = jax.random.split(jax.random.PRNGKey(90), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(jax.random.PRNGKey(91), policy, cfg)

    _key, _env_state, _turn_batch, transitions, rollout_metrics = collect_rollout_jax(
        jax.random.PRNGKey(92), env_state, turn_batch, train_state, policy, cfg
    )

    assert transitions.target_index.shape[-1] == cfg.model.max_moves_k
    k = edge_k(cfg.task)
    assert transitions.ship_bucket_mask.shape[-3:] == (
        MAX_PLANETS,
        k,
        cfg.task.ship_bucket_count,
    )
    assert "trajectory_shield_blocked_count" in rollout_metrics
    assert "trajectory_shield_fallback_noop_count" in rollout_metrics
    assert "trajectory_shield_legal_non_noop_count" in rollout_metrics
    assert "trajectory_shield_original_non_noop_count" in rollout_metrics
    assert "trajectory_shield_legal_non_noop_rate" in rollout_metrics
    assert 0.0 <= float(rollout_metrics["trajectory_shield_legal_non_noop_rate"]) <= 1.0


def test_jax_rollout_groups_collect_two_and_four_player_formats_under_jit():
    from src.jax.ppo_update import concatenate_transition_batches
    from src.jax.train import init_rollout_groups

    cfg = TrainConfig()
    cfg.task.max_fleets = 16
    cfg.task.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.training.num_envs = 4
    cfg.training.rollout_steps = 1
    cfg.opponents.mode.opponent = "random"
    _configure_rollout_groups(cfg, [
        {"name": "two_player", "player_count": 2, "num_envs": 2},
        {"name": "four_player", "player_count": 4, "num_envs": 2},
    ])
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
        assert transitions.planet_features.shape[:3] == (
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
    assert combined.planet_features.shape[:3] == (
        cfg.training.rollout_steps,
        4,
        MAX_PLANETS,
    )
    assert combined.target_index.shape == (
        cfg.training.rollout_steps,
        4,
        cfg.model.max_moves_k,
    )


def test_collect_rollout_jax_rotation_covers_all_player_ids_across_envs():
    from src.jax.env import assign_learner_players

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
