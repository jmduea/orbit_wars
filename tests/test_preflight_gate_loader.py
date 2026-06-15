"""Tests for YAML-authoritative preflight gate loading."""

from __future__ import annotations

from src.jax.preflight import _gate_specs
from src.jax.preflight_gate_loader import build_gate_spec, load_gate_yaml


def test_load_gate_yaml_includes_train_section() -> None:
    recipe = load_gate_yaml("beat_noop")
    assert recipe["gate_id"] == "beat_noop"
    train = recipe.get("train")
    assert isinstance(train, dict)
    assert "default" in train
    assert "planet_flow_target_heatmap" in train


def test_build_gate_spec_uses_curriculum_only_beat_noop_overrides() -> None:
    spec = build_gate_spec("beat_noop", model="transformer_factorized_small")
    assert spec.gate_id == "beat_noop"
    assert "model=transformer_factorized_small" in spec.train_overrides
    assert "curriculum=noop_only" in spec.train_overrides
    assert "training.total_updates=200" in spec.train_overrides
    assert spec.min_win_rate_delta is not None


def test_gate_specs_round_trip_all_gates() -> None:
    specs = _gate_specs("transformer_factorized_small")
    assert set(specs) == {"beat_noop", "beat_random", "curriculum_staged"}
    curriculum = specs["curriculum_staged"]
    assert curriculum.require_curriculum_promotion is True
    assert curriculum.min_win_rate_delta is None
    assert "model=transformer_factorized" in curriculum.train_overrides


def test_planet_flow_gate_spec_from_yaml() -> None:
    spec = build_gate_spec("beat_random", model="planet_flow_target_heatmap")
    assert "model=planet_flow_target_heatmap" in spec.train_overrides
    assert spec.require_planet_flow_control_metrics is True
