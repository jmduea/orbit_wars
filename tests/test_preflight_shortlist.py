"""Preflight W&B shortlist ranking."""

from __future__ import annotations

from src.jax.planet_flow_shortlist import ShortlistRunInput
from src.jax.preflight_shortlist import (
    build_preflight_shortlist_report,
    evaluate_preflight_shortlist_run,
    rank_preflight_eligible_entries,
)
from src.jax.train.sweep_score import PREFLIGHT_SWEEP_SCORE_INELIGIBLE


def test_evaluate_preflight_shortlist_run_eligible_when_gates_pass() -> None:
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
    entry = evaluate_preflight_shortlist_run(run)
    assert entry["eligible"] is True
    assert float(entry["preflight_sweep_score"]) > 0.0


def test_evaluate_preflight_shortlist_run_eligible_via_recovery_delta() -> None:
    run = ShortlistRunInput(
        run_id="recovery",
        name="recovery",
        summary={
            "win_rate_delta_10": 0.0,
            "win_rate_recovery_delta_10": 0.12,
            "approx_kl_window_mean": 0.05,
            "entropy_window_mean": 0.01,
        },
        config={},
    )
    entry = evaluate_preflight_shortlist_run(run)
    assert entry["eligible"] is True
    assert entry["preflight_sweep_score"] == 0.12


def test_evaluate_preflight_shortlist_run_ineligible_when_entropy_collapses() -> None:
    run = ShortlistRunInput(
        run_id="entropy",
        name="entropy",
        summary={
            "win_rate_delta_10": 0.12,
            "win_rate_recovery_delta_10": 0.20,
            "approx_kl_window_mean": 0.05,
            "entropy_window_mean": 0.01,
            "entropy_retention_ratio_10": 0.10,
        },
        config={},
    )
    entry = evaluate_preflight_shortlist_run(run)
    assert entry["eligible"] is False
    assert entry["preflight_sweep_score"] == PREFLIGHT_SWEEP_SCORE_INELIGIBLE
    assert any(
        "entropy_retention_ratio_10" in reason for reason in entry["guardrail_reasons"]
    )


def test_evaluate_preflight_shortlist_run_ineligible_when_kl_high() -> None:
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
    entry = evaluate_preflight_shortlist_run(run)
    assert entry["eligible"] is False
    assert entry["preflight_sweep_score"] == PREFLIGHT_SWEEP_SCORE_INELIGIBLE


def test_rank_preflight_eligible_entries_prefers_higher_score() -> None:
    ranked = rank_preflight_eligible_entries(
        [
            {
                "eligible": True,
                "preflight_sweep_score": 0.06,
                "win_rate_delta_10": 0.06,
            },
            {
                "eligible": True,
                "preflight_sweep_score": 0.09,
                "win_rate_delta_10": 0.09,
            },
        ]
    )
    assert ranked[0]["preflight_sweep_score"] == 0.09


def test_build_preflight_shortlist_report_winner_is_top_eligible() -> None:
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
    report = build_preflight_shortlist_report(runs, sweep_id="sweep-1", limit=1)
    assert report["winner"] is not None
    assert report["winner"]["run_id"] == "high"
    assert len(report["eligible"]) == 1
