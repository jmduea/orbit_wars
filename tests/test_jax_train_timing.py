from types import SimpleNamespace

from src.jax.train import _active_group_indices, _build_per_format_timing_metrics


def _rollout_group(player_count: int):
    return SimpleNamespace(
        cfg=SimpleNamespace(task=SimpleNamespace(player_count=player_count))
    )


def test_active_group_indices_runs_all_formats_by_default():
    groups = [_rollout_group(2), _rollout_group(4)]
    indices = _active_group_indices(groups, {2: 0.5, 4: 0.5}, update=1)
    assert indices == [0, 1]


def test_active_group_indices_rotate_selects_one_format_per_update():
    groups = [_rollout_group(2), _rollout_group(4)]
    first = _active_group_indices(
        groups,
        {2: 0.5, 4: 0.5},
        update=1,
        rotate_format_rollouts=True,
    )
    second = _active_group_indices(
        groups,
        {2: 0.5, 4: 0.5},
        update=51,
        rotate_format_rollouts=True,
    )
    assert first == [0]
    assert second == [1]


def test_build_per_format_timing_metrics_is_deterministic():
    metrics = _build_per_format_timing_metrics(
        {
            2: {"seconds": 2.0, "env_steps": 20.0, "samples": 200.0},
            4: {"seconds": 4.0, "env_steps": 40.0, "samples": 800.0},
        },
        update_seconds=10.0,
        rollout_seconds=6.0,
        ppo_seconds=3.0,
        include_per_format=True,
    )

    assert metrics["update_time_rollout_fraction"] == 0.6
    assert metrics["update_time_ppo_fraction"] == 0.3
    assert metrics["rollout_seconds_2p"] == 2.0
    assert metrics["rollout_seconds_4p"] == 4.0
    assert metrics["env_steps_per_sec_2p"] == 2.0
    assert metrics["env_steps_per_sec_4p"] == 4.0
    assert metrics["rollout_env_steps_per_sec_2p"] == 10.0
    assert metrics["rollout_env_steps_per_sec_4p"] == 10.0
    assert metrics["samples_per_sec_2p"] == 20.0
    assert metrics["samples_per_sec_4p"] == 80.0
    assert metrics["rollout_samples_per_sec_2p"] == 100.0
    assert metrics["rollout_samples_per_sec_4p"] == 200.0


def test_build_per_format_timing_metrics_skips_per_format_by_default():
    metrics = _build_per_format_timing_metrics(
        {
            2: {"seconds": 2.0, "env_steps": 20.0, "samples": 200.0},
            4: {"seconds": 4.0, "env_steps": 40.0, "samples": 800.0},
        },
        update_seconds=10.0,
        rollout_seconds=6.0,
        ppo_seconds=3.0,
    )

    assert metrics == {
        "update_time_rollout_fraction": 0.6,
        "update_time_ppo_fraction": 0.3,
    }


def test_build_per_format_timing_metrics_emits_inactive_format_zeros():
    metrics = _build_per_format_timing_metrics(
        {2: {"seconds": 1.0, "env_steps": 10.0, "samples": 30.0}},
        update_seconds=5.0,
        rollout_seconds=1.0,
        ppo_seconds=1.0,
        include_per_format=True,
    )

    assert metrics["rollout_seconds_4p"] == 0.0
    assert metrics["env_steps_per_sec_4p"] == 0.0
    assert metrics["rollout_env_steps_per_sec_4p"] == 0.0
    assert metrics["samples_per_sec_4p"] == 0.0
    assert metrics["rollout_samples_per_sec_4p"] == 0.0

def test_rollout_metrics_for_update_record_omits_disabled_groups() -> None:
    from src.jax.train_telemetry import rollout_metrics_for_update_record

    cfg = SimpleNamespace(model=SimpleNamespace(max_moves_k=4))
    rollout_scalars = {
        "win_rate_2p": 0.5,
        "stop_rate": 0.1,
        "mean_active_launches_per_turn": 2.0,
    }

    metrics = rollout_metrics_for_update_record(rollout_scalars, cfg)

    assert "win_rate_2p" not in metrics
    assert metrics["stop_rate"] == 0.1
    assert metrics["mean_active_launches_per_turn"] == 2.0
    assert metrics["stop_utilization_ratio"] == 0.5
    assert "trajectory_shield_blocked_count" not in metrics
