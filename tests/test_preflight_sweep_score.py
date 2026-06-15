from __future__ import annotations

import pytest

from src.config.schema import TelemetryConfig, TrainConfig, WandBConfig
from src.jax.train.sweep_score import (
    PREFLIGHT_SWEEP_SCORE_INELIGIBLE,
    EntropyTrendTracker,
    MetricWindowTracker,
    WinRateTrendTracker,
    collect_preflight_sweep_metrics,
    is_preflight_sweep,
    preflight_sweep_score,
)


def test_preflight_sweep_score_returns_delta_when_floors_pass() -> None:
    assert preflight_sweep_score(
        win_rate_delta=0.08,
        approx_kl=0.1,
        entropy=0.01,
    ) == pytest.approx(0.08)


def test_preflight_sweep_score_accepts_recovery_delta() -> None:
    assert preflight_sweep_score(
        win_rate_delta=0.0,
        win_rate_recovery_delta=0.12,
        approx_kl=0.1,
        entropy=0.01,
    ) == pytest.approx(0.12)


def test_preflight_sweep_score_rejects_entropy_collapse() -> None:
    assert (
        preflight_sweep_score(
            win_rate_delta=0.12,
            win_rate_recovery_delta=0.20,
            approx_kl=0.1,
            entropy=0.01,
            entropy_retention_ratio=0.10,
        )
        == PREFLIGHT_SWEEP_SCORE_INELIGIBLE
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"win_rate_delta": None, "approx_kl": 0.1, "entropy": 0.01},
        {"win_rate_delta": 0.08, "approx_kl": None, "entropy": 0.01},
        {"win_rate_delta": 0.08, "approx_kl": 0.1, "entropy": None},
        {"win_rate_delta": 0.02, "approx_kl": 0.1, "entropy": 0.01},
        {"win_rate_delta": 0.08, "approx_kl": 0.2, "entropy": 0.01},
        {"win_rate_delta": 0.08, "approx_kl": 0.1, "entropy": 1.0e-5},
    ],
)
def test_preflight_sweep_score_ineligible(kwargs: dict[str, float | None]) -> None:
    assert preflight_sweep_score(**kwargs) == PREFLIGHT_SWEEP_SCORE_INELIGIBLE


def test_is_preflight_sweep_detects_tag() -> None:
    cfg = TrainConfig(
        telemetry=TelemetryConfig(wandb=WandBConfig(tags=["preflight", "gates_2_3"]))
    )
    assert is_preflight_sweep(cfg) is True

    assert is_preflight_sweep(TrainConfig()) is False


def test_is_preflight_sweep_ignores_legacy_ssot_tag() -> None:
    cfg = TrainConfig(
        telemetry=TelemetryConfig(
            wandb=WandBConfig(tags=["ssot_preflight", "gates_2_3"])
        )
    )
    assert is_preflight_sweep(cfg) is False


def test_collect_preflight_sweep_metrics_populates_score() -> None:
    win_rate_trend = WinRateTrendTracker(window=3)
    approx_kl_window = MetricWindowTracker(window=3)
    entropy_window = MetricWindowTracker(window=3)
    entropy_trend = EntropyTrendTracker(window=3)
    for rate in (0.40, 0.45, 0.50, 0.55, 0.60):
        win_rate_trend.observe(rate)
    for kl in (0.05, 0.06, 0.07):
        approx_kl_window.observe(kl)
    for ent in (0.02, 0.03, 0.04):
        entropy_window.observe(ent)
        entropy_trend.observe(ent)

    metrics = collect_preflight_sweep_metrics(
        win_rate_trend=win_rate_trend,
        approx_kl_window=approx_kl_window,
        entropy_window=entropy_window,
        overall_win_rate=0.65,
        metrics_host={"approx_kl": 0.07, "entropy": 0.04},
        entropy_trend=entropy_trend,
    )

    assert "win_rate_delta_10" in metrics
    assert "win_rate_recovery_delta_10" in metrics
    assert "win_rate_window_mean_10" in metrics
    assert "win_rate_best_window_mean_10" in metrics
    assert "approx_kl_window_mean" in metrics
    assert "entropy_window_mean" in metrics
    assert "entropy_delta_10" in metrics
    assert "entropy_retention_ratio_10" in metrics
    assert metrics["preflight_sweep_score"] > 0.0


def test_collect_preflight_sweep_metrics_rejects_entropy_collapse() -> None:
    win_rate_trend = WinRateTrendTracker(window=3)
    approx_kl_window = MetricWindowTracker(window=3)
    entropy_window = MetricWindowTracker(window=3)
    entropy_trend = EntropyTrendTracker(window=3)
    for rate in (0.20, 0.25, 0.30, 0.55, 0.60):
        win_rate_trend.observe(rate)
    for kl in (0.01, 0.01, 0.01):
        approx_kl_window.observe(kl)
    for ent in (1.0, 1.0, 1.0, 0.1, 0.1):
        entropy_window.observe(ent)
        entropy_trend.observe(ent)

    metrics = collect_preflight_sweep_metrics(
        win_rate_trend=win_rate_trend,
        approx_kl_window=approx_kl_window,
        entropy_window=entropy_window,
        overall_win_rate=0.65,
        metrics_host={"approx_kl": 0.01, "entropy": 0.1},
        entropy_trend=entropy_trend,
    )

    assert metrics["entropy_retention_ratio_10"] < 0.25
    assert metrics["preflight_sweep_score"] == PREFLIGHT_SWEEP_SCORE_INELIGIBLE


def test_collect_preflight_sweep_metrics_recovers_from_lucky_first_window() -> None:
    win_rate_trend = WinRateTrendTracker(window=3)
    approx_kl_window = MetricWindowTracker(window=3)
    entropy_window = MetricWindowTracker(window=3)
    for rate in (1.0, 1.0, 1.0, 0.7, 0.7, 0.7, 1.0, 1.0):
        win_rate_trend.observe(rate)
    for kl in (0.01, 0.01):
        approx_kl_window.observe(kl)
    for ent in (0.8, 0.8):
        entropy_window.observe(ent)

    metrics = collect_preflight_sweep_metrics(
        win_rate_trend=win_rate_trend,
        approx_kl_window=approx_kl_window,
        entropy_window=entropy_window,
        overall_win_rate=1.0,
        metrics_host={"approx_kl": 0.01, "entropy": 0.8},
    )

    assert metrics["win_rate_delta_10"] == pytest.approx(0.0)
    assert metrics["win_rate_recovery_delta_10"] == pytest.approx(0.3)
    assert metrics["preflight_sweep_score"] == pytest.approx(0.3)
