"""Tests for offline rollout phase profile helpers."""

from __future__ import annotations

import pytest

from src.jax.rollout.phase_timing_report import (
    extract_rollout_phase_breakdown_from_records,
)
from src.jax.rollout_phase_profile import resolve_profile_overrides


def test_admission_profile_quick_geometry_by_default() -> None:
    overrides = resolve_profile_overrides(
        preset="admission",
        extra_overrides=("task=map_pool",),
        updates=5,
        quick=True,
    )
    assert "training=smoke" in overrides
    assert "training.rollout_steps=256" not in overrides
    assert "task=map_pool" in overrides


def test_admission_profile_full_geometry_opt_in() -> None:
    overrides = resolve_profile_overrides(
        preset="admission",
        extra_overrides=("task=map_pool",),
        updates=5,
        quick=False,
    )
    assert "training=2p4p_32_split" in overrides
    assert "training.rollout_steps=256" in overrides
    assert "task.candidate_count=3" in overrides
    assert "opponents=noop_only" in overrides
    assert "task=map_pool" in overrides
    assert "training.total_updates=5" in overrides
    assert "telemetry=rollout_phase_timing" not in overrides


def test_profile_breakdown_uses_measured_window() -> None:
    records = [
        {
            "update": 3,
            "rollout_seconds": 10.0,
            "rollout_phase_policy_seconds": 6.0,
            "rollout_phase_opponent_seconds": 1.0,
            "rollout_phase_env_step_seconds": 2.0,
            "rollout_phase_reset_seconds": 0.5,
            "rollout_phase_post_step_seconds": 0.5,
            "rollout_phase_measured_total_seconds": 10.0,
            "rollout_phase_policy_fraction": 0.6,
            "rollout_phase_opponent_fraction": 0.1,
            "rollout_phase_env_step_fraction": 0.2,
            "rollout_phase_reset_fraction": 0.05,
            "rollout_phase_post_step_fraction": 0.05,
        }
    ]
    payload = extract_rollout_phase_breakdown_from_records(records)
    assert payload["measured_updates"] == 1
    assert payload["phases"]["policy"]["fraction_mean"] == pytest.approx(0.6)
