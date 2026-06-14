"""Telemetry metric definitions for the losses group."""

from __future__ import annotations

from src.telemetry.metric_definition import MetricDefinition, metric


_LOSSES_BY_NAME: dict[str, MetricDefinition] = {
    "policy_loss": metric(
        "policy_loss",
        "losses",
        "Mean PPO policy loss across minibatches.",
    ),
    "value_loss": metric(
        "value_loss",
        "losses",
        "Mean PPO value loss across minibatches.",
    ),
    "entropy": metric("entropy", "losses", "Mean action entropy across minibatches."),
    "entropy_stop": metric(
        "entropy_stop",
        "losses",
        "Mean stop-head entropy for factorized pointer decoders.",
    ),
    "entropy_move": metric(
        "entropy_move",
        "losses",
        "Mean source/target/ship entropy for factorized pointer decoders.",
    ),
    "approx_kl": metric(
        "approx_kl",
        "losses",
        "Unweighted mean over PPO minibatches of mean(old_log_prob - new_log_prob); "
        "can diverge with epochs>1 and many inner steps.",
    ),
    "approx_kl_v2": metric(
        "approx_kl_v2",
        "losses",
        "Schulman-style approximate KL using clipped importance ratios.",
    ),
    "approx_kl_first_minibatch": metric(
        "approx_kl_first_minibatch",
        "losses",
        "approx_kl on the first minibatch before any optimizer step (parity sentinel).",
        protected=True,
    ),
    "approx_kl_last_minibatch": metric(
        "approx_kl_last_minibatch",
        "losses",
        "approx_kl on the final minibatch after all inner optimizer steps.",
    ),
    "approx_kl_v2_first_minibatch": metric(
        "approx_kl_v2_first_minibatch",
        "losses",
        "approx_kl_v2 on the first minibatch before any optimizer step.",
    ),
    "approx_kl_v2_last_minibatch": metric(
        "approx_kl_v2_last_minibatch",
        "losses",
        "approx_kl_v2 on the final minibatch after all inner optimizer steps.",
    ),
    "log_ratio_abs_mean": metric(
        "log_ratio_abs_mean",
        "losses",
        "Mean absolute log-probability delta between rollout and replay.",
    ),
    "log_ratio_abs_max_last_minibatch": metric(
        "log_ratio_abs_max_last_minibatch",
        "losses",
        "Max absolute log-probability delta on the final PPO minibatch.",
    ),
    "log_ratio_abs_max": metric(
        "log_ratio_abs_max",
        "losses",
        "Max absolute log-probability delta across all PPO minibatches.",
    ),
    "log_ratio_abs_max_minibatch": metric(
        "log_ratio_abs_max_minibatch",
        "losses",
        "Minibatch index containing the max absolute log-probability delta.",
    ),
    "old_log_prob_min_at_log_ratio_abs_max": metric(
        "old_log_prob_min_at_log_ratio_abs_max",
        "losses",
        "Minimum active old log-probability in the max log-ratio minibatch.",
    ),
    "old_log_prob_max_at_log_ratio_abs_max": metric(
        "old_log_prob_max_at_log_ratio_abs_max",
        "losses",
        "Maximum active old log-probability in the max log-ratio minibatch.",
    ),
    "new_log_prob_min_at_log_ratio_abs_max": metric(
        "new_log_prob_min_at_log_ratio_abs_max",
        "losses",
        "Minimum active new log-probability in the max log-ratio minibatch.",
    ),
    "new_log_prob_max_at_log_ratio_abs_max": metric(
        "new_log_prob_max_at_log_ratio_abs_max",
        "losses",
        "Maximum active new log-probability in the max log-ratio minibatch.",
    ),
    "old_log_prob_at_neg100_frac_at_log_ratio_abs_max": metric(
        "old_log_prob_at_neg100_frac_at_log_ratio_abs_max",
        "losses",
        "Fraction of active old log-probs <= -100 in the max log-ratio minibatch.",
    ),
    "new_log_prob_at_neg100_frac_at_log_ratio_abs_max": metric(
        "new_log_prob_at_neg100_frac_at_log_ratio_abs_max",
        "losses",
        "Fraction of active new log-probs <= -100 in the max log-ratio minibatch.",
    ),
    "source_log_prob_min_at_log_ratio_abs_max": metric(
        "source_log_prob_min_at_log_ratio_abs_max",
        "losses",
        "Minimum active source log-probability in the max log-ratio minibatch.",
    ),
    "target_log_prob_min_at_log_ratio_abs_max": metric(
        "target_log_prob_min_at_log_ratio_abs_max",
        "losses",
        "Minimum active target log-probability in the max log-ratio minibatch.",
    ),
    "ship_log_prob_min_at_log_ratio_abs_max": metric(
        "ship_log_prob_min_at_log_ratio_abs_max",
        "losses",
        "Minimum active ship log-probability in the max log-ratio minibatch.",
    ),
    "importance_ratio_mean": metric(
        "importance_ratio_mean",
        "losses",
        "Mean clipped importance ratio exp(clip(new_log_prob - old_log_prob)).",
    ),
    "clip_fraction": metric(
        "clip_fraction",
        "losses",
        "Fraction of masked steps where the importance ratio exceeds clip_coef.",
    ),
    "parity_logprob_delta_abs_mean": metric(
        "parity_logprob_delta_abs_mean",
        "losses",
        "Pre-update mean |replay_log_prob - stored_old_log_prob| on the first minibatch.",
        protected=True,
    ),
    "parity_logprob_delta_abs_max": metric(
        "parity_logprob_delta_abs_max",
        "losses",
        "Pre-update max |replay_log_prob - stored_old_log_prob| on the first minibatch.",
        protected=True,
    ),
    "parity_old_log_prob_finite": metric(
        "parity_old_log_prob_finite",
        "losses",
        "Whether stored old log-probs are finite on the first minibatch parity slice.",
    ),
    "parity_new_log_prob_finite": metric(
        "parity_new_log_prob_finite",
        "losses",
        "Whether replay log-probs are finite on the first minibatch parity slice.",
    ),
    "total_loss": metric("total_loss", "losses", "Final weighted PPO loss used for optimization."),
    "minibatches": metric("minibatches", "losses", "Minibatch count used in the PPO update."),
}


def losses_metric_definitions() -> tuple[MetricDefinition, ...]:
    return tuple(_LOSSES_BY_NAME[name] for name in _LOSSES_BY_NAME)
