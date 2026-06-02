"""Tests for per-model preflight PPO profile registry."""

from __future__ import annotations

import json

from src.jax.preflight import _gate_specs
from src.jax.preflight_calibration import calibration_train_overrides
from src.jax.preflight_profiles import (
    default_profiles_path,
    ppo_overrides_for_model,
)


def test_ppo_overrides_for_transformer_factorized_small() -> None:
    overrides = ppo_overrides_for_model("transformer_factorized_small")
    assert "training.lr=0.0003" in overrides
    assert "training.epochs=2" in overrides


def test_gate_specs_includes_profile_ppo_overrides() -> None:
    spec = _gate_specs("transformer_factorized_small")["beat_noop"]
    assert "training.lr=0.0003" in spec.train_overrides
    assert "training=2p_16" in spec.train_overrides
    assert "opponents=noop_only" in spec.train_overrides


def test_calibration_train_overrides_includes_profile(tmp_path) -> None:
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(
        json.dumps(
            {
                "models": {
                    "transformer_factorized_small": {
                        "ppo_overrides": ["training.lr=0.0001"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    overrides = calibration_train_overrides(
        "noop_only",
        seed=42,
        total_updates=200,
        profiles_path=profile_path,
    )
    assert "training.lr=0.0001" in overrides


def test_default_profiles_path_exists() -> None:
    path = default_profiles_path()
    assert path.is_file()
