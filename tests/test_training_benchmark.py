from __future__ import annotations

from src.jax.training_benchmark import (
    PRIMARY_E2E_OVERRIDES,
    TrainingBenchmarkResult,
    default_benchmark_updates,
    resolve_benchmark_overrides,
    training_benchmark_payload,
)


def test_training_benchmark_payload_includes_r3_metrics() -> None:
    payload = training_benchmark_payload(
        TrainingBenchmarkResult(
            label="unit",
            overrides=("task=shield_cheap",),
            updates=5,
            warmup=2,
            measured_updates=3,
            seconds_total=30.0,
            seconds_per_update_mean=10.0,
            compile_seconds_to_update_3=12.5,
            devices=("cuda:0",),
            default_backend="gpu",
            num_envs=32,
            rollout_steps=128,
            update_metric_means={},
            rollout_metric_means={},
            env_steps=9600,
            samples=12000,
            env_steps_per_sec=320.0,
            samples_per_sec=400.0,
        )
    )

    assert payload["env_steps_per_sec"] == 320.0
    assert payload["samples_per_sec"] == 400.0
    assert payload["seconds_per_update_mean"] == 10.0
    assert payload["compile_seconds_to_update_3"] == 12.5
    assert payload["env_steps"] == 9600
    assert payload["samples"] == 12000


def test_samples_per_sec_matches_totals_over_measured_seconds() -> None:
    total_seconds = 18.0
    samples = 5400
    payload = training_benchmark_payload(
        TrainingBenchmarkResult(
            label="ratio",
            overrides=(),
            updates=5,
            warmup=2,
            measured_updates=3,
            seconds_total=total_seconds,
            seconds_per_update_mean=total_seconds / 3,
            compile_seconds_to_update_3=None,
            devices=(),
            default_backend="gpu",
            num_envs=16,
            rollout_steps=64,
            update_metric_means={},
            rollout_metric_means={},
            env_steps=3072,
            samples=samples,
            env_steps_per_sec=3072 / total_seconds,
            samples_per_sec=samples / total_seconds,
        )
    )

    assert payload["samples_per_sec"] == payload["samples"] / payload["seconds_total"]


def test_primary_preset_resolves_shield_cheap_overrides() -> None:
    overrides = resolve_benchmark_overrides(preset="primary", overrides=None)
    assert "task=shield_cheap" in overrides
    assert "model=transformer_factorized" in overrides
    assert overrides == list(PRIMARY_E2E_OVERRIDES)


def test_primary_preset_default_updates_is_twenty() -> None:
    assert default_benchmark_updates(preset="primary") == 20
    assert default_benchmark_updates(preset=None) == 30
