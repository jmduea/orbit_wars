from __future__ import annotations

import pytest

from src.config.schema import TelemetryConfig, TrainConfig, WandBConfig
from src.jax.train.sweep_score import (
    PREFLIGHT_SWEEP_SCORE_INELIGIBLE,
    MetricWindowTracker,
    WinRateTrendTracker,
    collect_ssot_preflight_sweep_metrics,
    is_ssot_preflight_sweep,
    preflight_sweep_score,
)


def test_preflight_sweep_score_returns_delta_when_floors_pass() -> None:
    assert preflight_sweep_score(
        win_rate_delta=0.08,
        approx_kl=0.1,
        entropy=0.01,
    ) == pytest.approx(0.08)


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


def test_is_ssot_preflight_sweep_detects_tag() -> None:
    cfg = TrainConfig(
        telemetry=TelemetryConfig(
            wandb=WandBConfig(tags=["ssot_preflight", "gates_2_3"])
        )
    )
    assert is_ssot_preflight_sweep(cfg) is True

    cfg_preflight_tag = TrainConfig(
        telemetry=TelemetryConfig(wandb=WandBConfig(tags=["preflight"]))
    )
    assert is_ssot_preflight_sweep(cfg_preflight_tag) is True

    assert is_ssot_preflight_sweep(TrainConfig()) is False


def test_collect_ssot_preflight_sweep_metrics_populates_score() -> None:
    win_rate_trend = WinRateTrendTracker(window=3)
    approx_kl_window = MetricWindowTracker(window=3)
    entropy_window = MetricWindowTracker(window=3)
    for rate in (0.40, 0.45, 0.50, 0.55, 0.60):
        win_rate_trend.observe(rate)
    for kl in (0.05, 0.06, 0.07):
        approx_kl_window.observe(kl)
    for ent in (0.02, 0.03, 0.04):
        entropy_window.observe(ent)

    metrics = collect_ssot_preflight_sweep_metrics(
        win_rate_trend=win_rate_trend,
        approx_kl_window=approx_kl_window,
        entropy_window=entropy_window,
        overall_win_rate=0.65,
        metrics_host={"approx_kl": 0.07, "entropy": 0.04},
    )

    assert "win_rate_delta_10" in metrics
    assert "approx_kl_window_mean" in metrics
    assert "entropy_window_mean" in metrics
    assert metrics["preflight_sweep_score"] > 0.0
