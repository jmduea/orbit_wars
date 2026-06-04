"""SSOT preflight W&B shortlist ranking."""

from __future__ import annotations

from src.jax.planet_flow_shortlist import ShortlistRunInput
from src.jax.ssot_preflight_shortlist import (
    build_ssot_shortlist_report,
    evaluate_ssot_shortlist_run,
    rank_ssot_eligible_entries,
)
from src.jax.train.sweep_score import SSOT_PREFLIGHT_SWEEP_SCORE_INELIGIBLE


def test_evaluate_ssot_shortlist_run_eligible_when_gates_pass() -> None:
    run = ShortlistRunInput(
        run_id="abc",
        name="run-1",
        summary={
            "win_rate_delta_10": 0.08,
            "approx_kl_window_mean": 0.05,
            "entropy_window_mean": 0.01,
        },
        config={},
    )
    entry = evaluate_ssot_shortlist_run(run)
    assert entry["eligible"] is True
    assert float(entry["ssot_preflight_sweep_score"]) > 0.0


def test_evaluate_ssot_shortlist_run_ineligible_when_kl_high() -> None:
    run = ShortlistRunInput(
        run_id="abc",
        name="run-1",
        summary={
            "win_rate_delta_10": 0.08,
            "approx_kl_window_mean": 0.99,
            "entropy_window_mean": 0.01,
        },
        config={},
    )
    entry = evaluate_ssot_shortlist_run(run)
    assert entry["eligible"] is False
    assert entry["ssot_preflight_sweep_score"] == SSOT_PREFLIGHT_SWEEP_SCORE_INELIGIBLE


def test_rank_ssot_eligible_entries_prefers_higher_score() -> None:
    ranked = rank_ssot_eligible_entries(
        [
            {"eligible": True, "ssot_preflight_sweep_score": 0.06, "win_rate_delta_10": 0.06},
            {"eligible": True, "ssot_preflight_sweep_score": 0.09, "win_rate_delta_10": 0.09},
        ]
    )
    assert ranked[0]["ssot_preflight_sweep_score"] == 0.09


def test_build_ssot_shortlist_report_winner_is_top_eligible() -> None:
    runs = [
        ShortlistRunInput(
            run_id="low",
            name="low",
            summary={
                "win_rate_delta_10": 0.06,
                "approx_kl_window_mean": 0.05,
                "entropy_window_mean": 0.01,
            },
            config={},
        ),
        ShortlistRunInput(
            run_id="high",
            name="high",
            summary={
                "win_rate_delta_10": 0.12,
                "approx_kl_window_mean": 0.05,
                "entropy_window_mean": 0.01,
            },
            config={},
        ),
    ]
    report = build_ssot_shortlist_report(runs, sweep_id="sweep-1", limit=1)
    assert report["winner"] is not None
    assert report["winner"]["run_id"] == "high"
    assert len(report["eligible"]) == 1
