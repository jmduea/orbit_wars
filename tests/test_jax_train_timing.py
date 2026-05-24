from src.jax_train import _build_per_format_timing_metrics


def test_build_per_format_timing_metrics_is_deterministic():
    metrics = _build_per_format_timing_metrics(
        {
            2: {"seconds": 2.0, "env_steps": 20.0, "samples": 200.0},
            4: {"seconds": 4.0, "env_steps": 40.0, "samples": 800.0},
        },
        update_seconds=10.0,
        rollout_seconds=6.0,
        ppo_seconds=3.0,
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


def test_build_per_format_timing_metrics_emits_inactive_format_zeros():
    metrics = _build_per_format_timing_metrics(
        {2: {"seconds": 1.0, "env_steps": 10.0, "samples": 30.0}},
        update_seconds=5.0,
        rollout_seconds=1.0,
        ppo_seconds=1.0,
    )

    assert metrics["rollout_seconds_4p"] == 0.0
    assert metrics["env_steps_per_sec_4p"] == 0.0
    assert metrics["rollout_env_steps_per_sec_4p"] == 0.0
    assert metrics["samples_per_sec_4p"] == 0.0
    assert metrics["rollout_samples_per_sec_4p"] == 0.0