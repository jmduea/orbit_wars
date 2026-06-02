from __future__ import annotations

from pathlib import Path

import pytest

from src.jax.training_benchmark import (
    DEFAULT_E2E_WITHIN_PCT,
    E2E_THROUGHPUT_GATE,
    aggregate_e2e_run_payloads,
    build_e2e_baseline_artifact,
    compare_e2e_throughput_to_baseline,
    derive_e2e_pass_band,
    load_e2e_baseline,
    validate_e2e_baseline_artifact,
)


def _sample_runs() -> list[dict[str, object]]:
    return [
        {
            "env_steps_per_sec": 4000.0,
            "samples_per_sec": 8000.0,
            "seconds_per_update_mean": 1.0,
        },
        {
            "env_steps_per_sec": 4100.0,
            "samples_per_sec": 8200.0,
            "seconds_per_update_mean": 0.98,
        },
        {
            "env_steps_per_sec": 3900.0,
            "samples_per_sec": 7800.0,
            "seconds_per_update_mean": 1.02,
        },
    ]


def test_derive_pass_band_floors_and_ceilings_from_mean() -> None:
    aggregate = aggregate_e2e_run_payloads(_sample_runs())
    pass_band = derive_e2e_pass_band(aggregate, within_pct=10.0)

    assert pass_band["within_pct"] == 10.0
    assert pass_band["floors"]["env_steps_per_sec"] == pytest.approx(4000.0 * 0.9)
    assert pass_band["ceilings"]["seconds_per_update_mean"] == pytest.approx(
        aggregate["seconds_per_update_mean"]["mean"] * 1.1
    )


def test_validate_baseline_requires_three_runs_and_gate() -> None:
    aggregate = aggregate_e2e_run_payloads(_sample_runs())
    valid = {
        "gate": E2E_THROUGHPUT_GATE,
        "runs": _sample_runs(),
        "aggregate": aggregate,
    }
    assert validate_e2e_baseline_artifact(valid) == []

    invalid = dict(valid)
    invalid["runs"] = _sample_runs()[:2]
    errors = validate_e2e_baseline_artifact(invalid)
    assert any("at least 3" in item for item in errors)


def test_compare_fails_when_env_steps_below_floor() -> None:
    pass_band = derive_e2e_pass_band(
        {"env_steps_per_sec": {"mean": 4000.0, "stddev": 0.0}},
        within_pct=10.0,
    )
    passed, failures = compare_e2e_throughput_to_baseline(
        {
            "env_steps_per_sec": 3500.0,
            "samples_per_sec": 9000.0,
            "seconds_per_update_mean": 0.9,
        },
        pass_band=pass_band,
    )

    assert passed is False
    assert any("env_steps_per_sec" in item for item in failures)


def test_compare_passes_when_all_metrics_within_band() -> None:
    aggregate = aggregate_e2e_run_payloads(_sample_runs())
    pass_band = derive_e2e_pass_band(aggregate, within_pct=DEFAULT_E2E_WITHIN_PCT)
    measured = {
        key: float(aggregate[key]["mean"])
        for key in ("env_steps_per_sec", "samples_per_sec", "seconds_per_update_mean")
    }
    passed, failures = compare_e2e_throughput_to_baseline(
        measured,
        pass_band=pass_band,
    )

    assert passed is True
    assert failures == []


def test_committed_e2e_baseline_artifact_is_valid() -> None:
    baseline_path = Path("docs/benchmarks/launch-hygiene-e2e-baseline.json")
    loaded = load_e2e_baseline(baseline_path)
    errors = validate_e2e_baseline_artifact(loaded)
    assert errors == [], f"committed baseline invalid: {errors}"
    assert loaded["gate"] == E2E_THROUGHPUT_GATE
    assert len(loaded["runs"]) >= 3
    assert "pass_band" in loaded


def test_load_baseline_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="baseline artifact not found"):
        load_e2e_baseline(missing)


def test_build_baseline_artifact_round_trip(tmp_path: Path) -> None:
    runs = _sample_runs()
    artifact = build_e2e_baseline_artifact(
        commit_sha="79162a2088160b8ed05c3e3a050e064c7f6c9556",
        merge_topology_notes="PR #163 merge; first parent pre-hygiene.",
        co_landing_commits=["ce6714b"],
        run_date="2026-06-01",
        device={
            "default_backend": "gpu",
            "devices": ["cuda:0"],
            "jax_version": "0.10.0",
        },
        primary_profile={
            "preset": "primary",
            "overrides": ["task=shield_cheap"],
            "updates": 20,
            "warmup": 2,
            "seed": 42,
        },
        runs=runs,
        operator_example="make test-launch-hygiene-e2e-throughput",
    )
    path = tmp_path / "baseline.json"
    path.write_text(__import__("json").dumps(artifact, indent=2) + "\n", encoding="utf-8")
    loaded = load_e2e_baseline(path)
    assert loaded["gate"] == E2E_THROUGHPUT_GATE
    assert len(loaded["runs"]) == 3
