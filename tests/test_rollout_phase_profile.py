"""Tests for offline rollout phase profile helpers."""

from __future__ import annotations

import pytest

import jax
from src.benchmark.rollout_phase_profile import (
    _maybe_seed_historical_snapshots,
    compose_profile_config,
    resolve_profile_overrides,
)
from src.config import compose_hydra_train_config
from src.jax.policy import build_jax_policy
from src.jax.rollout.phase_timing_report import (
    extract_rollout_phase_breakdown_from_records,
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
    assert "curriculum=noop_only" in overrides
    assert "task=map_pool" in overrides
    assert "training.total_updates=5" in overrides
    assert "telemetry=rollout_phase_timing" not in overrides


def test_profile_config_composes_production_mix_overrides() -> None:
    cfg = compose_profile_config(
        preset="admission",
        extra_overrides=("curriculum=production_mix",),
        updates=20,
        quick=False,
    )

    assert cfg.opponents.dispatch == "self"
    assert cfg.curriculum.enabled is True
    assert cfg.training.rollout_steps == 256
    assert cfg.training.total_updates == 20


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


def test_profile_breakdown_includes_opponent_subphase_details() -> None:
    records = [
        {
            "update": 3,
            "rollout_seconds": 10.0,
            "rollout_phase_policy_seconds": 2.0,
            "rollout_phase_opponent_seconds": 6.0,
            "rollout_phase_opponent_sample_seconds": 4.0,
            "rollout_phase_opponent_encode_seconds": 2.0,
            "rollout_phase_env_step_seconds": 1.0,
            "rollout_phase_reset_seconds": 0.5,
            "rollout_phase_post_step_seconds": 0.5,
            "rollout_phase_measured_total_seconds": 10.0,
            "rollout_phase_policy_fraction": 0.2,
            "rollout_phase_opponent_fraction": 0.6,
            "rollout_phase_opponent_sample_fraction": 0.4,
            "rollout_phase_opponent_encode_fraction": 0.2,
            "rollout_phase_env_step_fraction": 0.1,
            "rollout_phase_reset_fraction": 0.05,
            "rollout_phase_post_step_fraction": 0.05,
        }
    ]
    payload = extract_rollout_phase_breakdown_from_records(records)
    details = payload["opponent_details"]
    assert details["opponent_sample"]["fraction_mean"] == pytest.approx(0.4)
    assert details["opponent_encode"]["fraction_mean"] == pytest.approx(0.2)


@pytest.mark.jax
def test_maybe_seed_historical_snapshots_for_production_mix() -> None:
    cfg = compose_hydra_train_config(["curriculum=self_play_staged", "training.total_updates=1"])
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
