"""Tests for offline rollout phase profile helpers."""

from __future__ import annotations

import pytest

import jax
from src.config import compose_hydra_train_config
from src.jax.policy import build_jax_policy
from src.jax.rollout.phase_timing_report import (
    extract_rollout_phase_breakdown_from_records,
)
from src.jax.rollout_phase_profile import (
    _maybe_seed_historical_snapshots,
    resolve_profile_overrides,
)
from src.jax.train import init_train_state
from src.jax.train.snapshots import init_historical_snapshot_pool
from src.training.curriculum import CurriculumController


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


@pytest.mark.jax
def test_maybe_seed_historical_snapshots_for_production_mix() -> None:
    cfg = compose_hydra_train_config(
        ["opponents=default", "curriculum=default", "training.total_updates=1"]
    )
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.PRNGKey(0), policy, cfg)
    pool = init_historical_snapshot_pool(
        train_state.params, cfg.opponents.snapshot.pool_size
    )
    curriculum = CurriculumController(cfg.curriculum, cfg.opponents.snapshot)
    assert not bool(jax.device_get(pool.valid_mask).any())

    seeded = _maybe_seed_historical_snapshots(
        pool, train_state.params, cfg, curriculum, seed_snapshots=1
    )
    assert bool(jax.device_get(seeded.valid_mask).any())


def test_maybe_seed_historical_snapshots_skips_without_historical_weight() -> None:
    cfg = compose_hydra_train_config(
        [
            "curriculum=scripted_heavy",
            "opponents=base",
            "opponents.self_play.enabled=true",
            "opponents.snapshot.pool_size=2",
            "opponents.snapshot.interval_updates=10",
            "training.total_updates=1",
        ]
    )
    policy = build_jax_policy(cfg)
    train_state = init_train_state(jax.random.PRNGKey(1), policy, cfg)
    pool = init_historical_snapshot_pool(train_state.params, 2)
    curriculum = CurriculumController(cfg.curriculum, cfg.opponents.snapshot)

    seeded = _maybe_seed_historical_snapshots(
        pool, train_state.params, cfg, curriculum, seed_snapshots=1
    )
    assert seeded is pool
    assert not bool(jax.device_get(seeded.valid_mask).any())
