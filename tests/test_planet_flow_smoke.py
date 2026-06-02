from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from src.jax.planet_flow_smoke import run_planet_flow_noop_smoke

from src.jax.preflight import GateEvaluation, PreflightVerdict


def _shortlist(tmp_path: Path, eligible: list[dict[str, object]]) -> Path:
    path = tmp_path / "shortlist.json"
    entries = [{**entry, "eligible": True} for entry in eligible]
    path.write_text(
        json.dumps({"eligible": entries, "audit": []}),
        encoding="utf-8",
    )
    return path


def test_smoke_empty_eligible_raises(tmp_path: Path) -> None:
    path = _shortlist(tmp_path, [])

    with pytest.raises(ValueError, match="no eligible"):
        run_planet_flow_noop_smoke(
            path,
            top_k=1,
            repo_root=tmp_path,
            dry_run=True,
        )


def test_smoke_marks_pass_when_beat_noop_verified(tmp_path: Path) -> None:
    shortlist = _shortlist(
        tmp_path,
        [
            {
                "run_id": "cfg_a",
                "train_overrides": ["training.lr=0.00003"],
            }
        ],
    )
    verified = GateEvaluation(
        gate_id="beat_noop",
        verdict=PreflightVerdict.VERIFIED,
        reasons=(),
        campaign="planet_flow_noop_smoke_cfg_a",
        run_dir=str(tmp_path / "run"),
        log_path="/tmp/log.jsonl",
        checkpoint_path=None,
        window_overall_win_rate=0.5,
        window_launches=0.2,
        win_rate_first_window=0.2,
        win_rate_delta=0.1,
        best_rolling_win_rate=0.5,
        curriculum_promotions=0,
        evaluation_mode="learning_signal",
    )

    with (
        patch("src.jax.planet_flow_smoke.run_ow_train"),
        patch("src.jax.planet_flow_smoke.latest_run_dir", return_value=tmp_path / "run"),
        patch(
            "src.jax.planet_flow_smoke.evaluate_gate_records",
            return_value=verified,
        ),
        patch("src.jax.planet_flow_smoke.read_jsonl_records", return_value=[{"update": 1}]),
    ):
        report = run_planet_flow_noop_smoke(
            shortlist,
            top_k=1,
            output_root=tmp_path,
            repo_root=tmp_path,
            dry_run=False,
        )

    assert report["any_passed"] is True
    assert report["recommended_for_learn_proof"]["run_id"] == "cfg_a"


def test_smoke_fails_on_kl_gate_reasons(tmp_path: Path) -> None:
    shortlist = _shortlist(
        tmp_path,
        [{"run_id": "cfg_b", "train_overrides": []}],
    )

    with (
        patch("src.jax.planet_flow_smoke.run_ow_train"),
        patch("src.jax.planet_flow_smoke.latest_run_dir", return_value=tmp_path / "run"),
        patch(
            "src.jax.planet_flow_smoke.evaluate_gate_records",
            return_value=GateEvaluation(
                gate_id="beat_noop",
                verdict=PreflightVerdict.NOT_VERIFIED,
                reasons=("approx_kl 0.5000 > 0.1500",),
                campaign="planet_flow_noop_smoke_cfg_b",
                run_dir=str(tmp_path / "run"),
                log_path="/tmp/log.jsonl",
                checkpoint_path=None,
                window_overall_win_rate=0.5,
                window_launches=0.2,
                win_rate_first_window=0.2,
                win_rate_delta=0.1,
                best_rolling_win_rate=0.5,
                curriculum_promotions=0,
                evaluation_mode="learning_signal",
            ),
        ),
        patch("src.jax.planet_flow_smoke.read_jsonl_records", return_value=[{"update": 1}]),
    ):
        report = run_planet_flow_noop_smoke(
            shortlist,
            top_k=1,
            output_root=tmp_path,
            repo_root=tmp_path,
            dry_run=False,
        )

    assert report["any_passed"] is False
    assert "approx_kl" in report["results"][0]["gate"]["reasons"][0]
