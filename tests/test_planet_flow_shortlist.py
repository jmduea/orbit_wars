from __future__ import annotations

from src.jax.planet_flow_shortlist import (
    ShortlistRunInput,
    build_shortlist_report,
    evaluate_shortlist_run,
    hydra_overrides_from_config,
    rank_eligible_entries,
)
from src.jax.train.sweep_score import PREFLIGHT_SWEEP_SCORE_INELIGIBLE


def _summary(**kwargs: float) -> dict[str, object]:
    base: dict[str, object] = {
        "win_rate_delta_10": 0.12,
        "approx_kl_window_mean": 0.05,
        "entropy_window_mean": 0.02,
        "mean_active_launches_per_turn": 0.2,
        "planet_flow_demanded_mass_sum": 500.0,
        "planet_flow_emitted_launch_count": 100.0,
        "planet_flow_unreachable_demand_rate": 0.0,
    }
    base.update(kwargs)
    return base


def test_high_window_kl_marks_ineligible_despite_strong_trend() -> None:
    entry = evaluate_shortlist_run(
        ShortlistRunInput(
            run_id="winner",
            name="winner",
            summary=_summary(approx_kl_window_mean=0.5, win_rate_delta_10=0.2),
            config={"training": {"lr": 0.0001}},
        )
    )

    assert entry["eligible"] is False
    assert entry["preflight_sweep_score"] == PREFLIGHT_SWEEP_SCORE_INELIGIBLE
    assert any("approx_kl" in str(reason) for reason in entry["guardrail_reasons"])


def test_rank_prefers_lower_kl_on_equal_win_rate_delta() -> None:
    low_kl = {
        "eligible": True,
        "win_rate_delta_10": 0.1,
        "approx_kl_window_mean": 0.04,
        "entropy_window_mean": 0.02,
        "mean_active_launches_per_turn": 0.2,
    }
    high_kl = {
        "eligible": True,
        "win_rate_delta_10": 0.1,
        "approx_kl_window_mean": 0.08,
        "entropy_window_mean": 0.02,
        "mean_active_launches_per_turn": 0.2,
    }

    ranked = rank_eligible_entries([high_kl, low_kl])

    assert ranked[0]["approx_kl_window_mean"] == 0.04


def test_missing_window_summary_fields_audit() -> None:
    entry = evaluate_shortlist_run(
        ShortlistRunInput(
            run_id="stale",
            name="stale",
            summary={"overall_win_rate": 0.9},
            config={},
        )
    )

    assert entry["eligible"] is False
    assert any("missing" in str(reason) for reason in entry["guardrail_reasons"])


def test_hydra_overrides_round_trip_ppo_keys() -> None:
    overrides = hydra_overrides_from_config(
        {
            "training": {
                "lr": 0.00003,
                "clip_coef": 0.12,
                "ent_coef": 0.001,
                "epochs": 2,
                "vf_coef": 1.0,
                "max_grad_norm": 0.8,
                "update_chunk_rows": 2048,
            }
        }
    )

    assert "training.lr=" in overrides[0]
    assert "training.clip_coef=0.12" in overrides
    assert "training.update_chunk_rows=2048" in overrides


def test_build_shortlist_report_splits_eligible_and_audit() -> None:
    report = build_shortlist_report(
        [
            ShortlistRunInput(
                run_id="good",
                name="good",
                summary=_summary(),
                config={},
            ),
            ShortlistRunInput(
                run_id="bad",
                name="bad",
                summary=_summary(approx_kl_window_mean=0.9),
                config={},
            ),
        ],
        sweep_id="test_sweep",
    )

    assert len(report["eligible"]) == 1
    assert report["eligible"][0]["run_id"] == "good"
    assert len(report["audit"]) == 1
    assert report["winner"]["run_id"] == "good"
