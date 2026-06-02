from __future__ import annotations

import pytest
from src.jax.train.sweep_score import (
    PLANET_FLOW_SWEEP_SCORE_INELIGIBLE,
    WinRateTrendTracker,
    planet_flow_sweep_score,
)


def _eligible_inputs(*, win_rate_delta: float = 0.12) -> dict[str, float | None]:
    return {
        "win_rate_delta": win_rate_delta,
        "mean_active_launches_per_turn": 0.2,
        "planet_flow_demanded_mass_sum": 500.0,
        "planet_flow_emitted_launch_count": 100.0,
        "entropy": 0.05,
        "approx_kl": 0.02,
    }


def test_planet_flow_sweep_score_returns_delta_when_floors_pass() -> None:
    score = planet_flow_sweep_score(**_eligible_inputs(win_rate_delta=0.12))

    assert score == pytest.approx(0.12)


def test_planet_flow_sweep_score_is_ineligible_without_win_rate_delta() -> None:
    score = planet_flow_sweep_score(**{**_eligible_inputs(), "win_rate_delta": None})

    assert score == PLANET_FLOW_SWEEP_SCORE_INELIGIBLE


def test_planet_flow_sweep_score_is_ineligible_when_launches_collapse() -> None:
    inputs = _eligible_inputs()
    inputs["mean_active_launches_per_turn"] = 0.0
    score = planet_flow_sweep_score(**inputs)

    assert score == PLANET_FLOW_SWEEP_SCORE_INELIGIBLE


def test_planet_flow_sweep_score_is_ineligible_when_post_mask_unreachable_high() -> None:
    score = planet_flow_sweep_score(
        **_eligible_inputs(),
        planet_flow_unreachable_demand_rate=0.25,
        max_post_mask_unreachable_rate=0.05,
    )

    assert score == PLANET_FLOW_SWEEP_SCORE_INELIGIBLE


def test_win_rate_trend_tracker_matches_first_and_last_window_means() -> None:
    tracker = WinRateTrendTracker(window=3)
    for value in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        tracker.observe(value)

    assert tracker.win_rate_delta() == pytest.approx(0.3)
