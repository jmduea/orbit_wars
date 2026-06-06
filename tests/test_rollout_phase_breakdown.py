"""Tests for ``ow benchmark rollout-phase-breakdown`` JSONL extract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.jax.rollout.phase_timing_report import (
    extract_rollout_phase_breakdown_from_input,
    extract_rollout_phase_breakdown_from_records,
    format_rollout_phase_breakdown,
)


def _record(update: int, **kwargs: float) -> dict[str, object]:
    base = {
        "update": update,
        "rollout_seconds": 10.0,
        "rollout_phase_policy_seconds": 4.0,
        "rollout_phase_opponent_seconds": 2.0,
        "rollout_phase_env_step_seconds": 3.0,
        "rollout_phase_reset_seconds": 0.5,
        "rollout_phase_post_step_seconds": 0.5,
        "rollout_phase_measured_total_seconds": 10.0,
        "rollout_phase_policy_fraction": 0.4,
        "rollout_phase_opponent_fraction": 0.2,
        "rollout_phase_env_step_fraction": 0.3,
        "rollout_phase_reset_fraction": 0.05,
        "rollout_phase_post_step_fraction": 0.05,
    }
    base.update(kwargs)
    return base


def test_extract_rollout_phase_breakdown_averages_window() -> None:
    records = [_record(1), _record(2), _record(3), _record(4)]
    payload = extract_rollout_phase_breakdown_from_records(records)
    assert payload["measured_updates"] == 2
    assert payload["updates_in_window"] == [3, 4]
    phases = payload["phases"]
    assert isinstance(phases, dict)
    assert phases["policy"]["seconds_mean"] == pytest.approx(4.0)
    assert phases["env_step"]["fraction_mean"] == pytest.approx(0.3)


def test_extract_rollout_phase_breakdown_requires_timing_metrics() -> None:
    with pytest.raises(ValueError, match="rollout-phase-profile"):
        extract_rollout_phase_breakdown_from_records([{"update": 5, "rollout_seconds": 1.0}])


def test_extract_rollout_phase_breakdown_from_profile_json(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "warmup": 2,
                "max_measured_update": 20,
                "per_update_records": [_record(3), _record(4)],
            }
        ),
        encoding="utf-8",
    )
    payload = extract_rollout_phase_breakdown_from_input(path)
    assert payload["measured_updates"] == 2
    assert payload["source_path"] == str(path)


def test_format_rollout_phase_breakdown_includes_phases(tmp_path: Path) -> None:
    payload = extract_rollout_phase_breakdown_from_records([_record(3), _record(4)])
    payload["log_path"] = str(tmp_path / "run_jax.jsonl")
    text = format_rollout_phase_breakdown(payload)
    assert "policy" in text
    assert "env_step" in text
    assert "measured total" in text
