"""Telemetry metric definitions for the debug group."""

from __future__ import annotations

from src.telemetry.metric_definition import MetricDefinition, metric


_DEBUG_BY_NAME: dict[str, MetricDefinition] = {
    "policy_loss_2p": metric("policy_loss_2p", "debug", "PPO policy loss for 2-player samples."),
    "value_loss_2p": metric("value_loss_2p", "debug", "PPO value loss for 2-player samples."),
    "entropy_2p": metric("entropy_2p", "debug", "Action entropy for 2-player samples."),
    "approx_kl_2p": metric("approx_kl_2p", "debug", "Approximate KL for 2-player samples."),
    "approx_kl_v2_2p": metric(
        "approx_kl_v2_2p",
        "debug",
        "Schulman-style approximate KL for 2-player samples.",
    ),
    "total_loss_2p": metric("total_loss_2p", "debug", "Weighted PPO loss for 2-player samples."),
    "loss_sample_count_2p": metric(
        "loss_sample_count_2p",
        "debug",
        "Learner decision samples contributing to 2-player PPO loss diagnostics.",
        rollout_scalar_role="chunk_only",
    ),
    "policy_loss_4p": metric("policy_loss_4p", "debug", "PPO policy loss for 4-player samples."),
    "value_loss_4p": metric("value_loss_4p", "debug", "PPO value loss for 4-player samples."),
    "entropy_4p": metric("entropy_4p", "debug", "Action entropy for 4-player samples."),
    "approx_kl_4p": metric("approx_kl_4p", "debug", "Approximate KL for 4-player samples."),
    "approx_kl_v2_4p": metric(
        "approx_kl_v2_4p",
        "debug",
        "Schulman-style approximate KL for 4-player samples.",
    ),
    "total_loss_4p": metric("total_loss_4p", "debug", "Weighted PPO loss for 4-player samples."),
    "loss_sample_count_4p": metric(
        "loss_sample_count_4p",
        "debug",
        "Learner decision samples contributing to 4-player PPO loss diagnostics.",
        rollout_scalar_role="chunk_only",
    ),
    "rollout_seconds_2p": metric(
        "rollout_seconds_2p",
        "debug",
        "Wall-clock seconds spent collecting 2-player rollout groups.",
    ),
    "rollout_seconds_4p": metric(
        "rollout_seconds_4p",
        "debug",
        "Wall-clock seconds spent collecting 4-player rollout groups.",
    ),
    "env_steps_per_sec_2p": metric(
        "env_steps_per_sec_2p",
        "debug",
        "2-player environment steps processed per second over the full update.",
    ),
    "env_steps_per_sec_4p": metric(
        "env_steps_per_sec_4p",
        "debug",
        "4-player environment steps processed per second over the full update.",
    ),
    "rollout_env_steps_per_sec_2p": metric(
        "rollout_env_steps_per_sec_2p",
        "debug",
        "2-player environment steps processed per second during 2-player rollout collection.",
    ),
    "rollout_env_steps_per_sec_4p": metric(
        "rollout_env_steps_per_sec_4p",
        "debug",
        "4-player environment steps processed per second during 4-player rollout collection.",
    ),
    "samples_per_sec_2p": metric(
        "samples_per_sec_2p",
        "debug",
        "2-player learner decision samples processed per second over the full update.",
    ),
    "samples_per_sec_4p": metric(
        "samples_per_sec_4p",
        "debug",
        "4-player learner decision samples processed per second over the full update.",
    ),
    "rollout_samples_per_sec_2p": metric(
        "rollout_samples_per_sec_2p",
        "debug",
        "2-player learner decision samples processed per second during 2-player rollout collection.",
    ),
    "rollout_samples_per_sec_4p": metric(
        "rollout_samples_per_sec_4p",
        "debug",
        "4-player learner decision samples processed per second during 4-player rollout collection.",
    ),
    "update_time_rollout_fraction": metric(
        "update_time_rollout_fraction",
        "debug",
        "Fraction of update wall time spent collecting rollouts.",
    ),
    "update_time_ppo_fraction": metric(
        "update_time_ppo_fraction",
        "debug",
        "Fraction of update wall time spent in PPO optimization.",
    ),
    "mean_ships_per_launch": metric(
        "mean_ships_per_launch",
        "debug",
        "Mean ship count across emitted fleet launches in the rollout.",
        rollout_scalar_role="finalized_rate",
    ),
    "debug_step_mask_sum": metric(
        "debug_step_mask_sum",
        "debug",
        "Sum of active PPO step masks in the update minibatch.",
    ),
    "debug_old_log_prob_finite": metric(
        "debug_old_log_prob_finite",
        "debug",
        "Whether all stored old log-probabilities are finite.",
    ),
    "debug_returns_finite": metric(
        "debug_returns_finite",
        "debug",
        "Whether all computed returns are finite.",
    ),
    "debug_advantages_finite": metric(
        "debug_advantages_finite",
        "debug",
        "Whether all computed advantages are finite.",
    ),
    "debug_ship_bucket_mask_any_min": metric(
        "debug_ship_bucket_mask_any_min",
        "debug",
        "Minimum per-step ship-bucket mask count in the update batch.",
    ),
    "debug_ship_bucket_mask_all_false": metric(
        "debug_ship_bucket_mask_all_false",
        "debug",
        "Count of rows whose ship-bucket mask is entirely false.",
    ),
    "debug_source_mask_all_false": metric(
        "debug_source_mask_all_false",
        "debug",
        "Count of rows whose source mask is entirely false.",
    ),
    "debug_active_launch_all_false_bucket": metric(
        "debug_active_launch_all_false_bucket",
        "debug",
        "Count of active launch rows with an all-false ship bucket.",
    ),
}


def debug_metric_definitions() -> tuple[MetricDefinition, ...]:
    return tuple(_DEBUG_BY_NAME[name] for name in _DEBUG_BY_NAME)
