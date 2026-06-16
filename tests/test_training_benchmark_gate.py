from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.benchmark.training import (
    DEFAULT_E2E_WITHIN_PCT,
    E2E_THROUGHPUT_GATE,
    aggregate_e2e_run_payloads,
    build_e2e_baseline_artifact,
    compare_e2e_throughput_to_baseline,
    derive_e2e_pass_band,
    load_e2e_baseline,
    resolve_e2e_measured_for_gate,
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


def test_compare_passes_when_samples_below_floor_but_env_steps_ok() -> None:
    pass_band = derive_e2e_pass_band(
        {"env_steps_per_sec": {"mean": 4000.0, "stddev": 0.0}},
        within_pct=10.0,
    )
    passed, failures = compare_e2e_throughput_to_baseline(
        {
            "env_steps_per_sec": 4000.0,
            "samples_per_sec": 100.0,
            "seconds_per_update_mean": 1.0,
        },
        pass_band=pass_band,
    )

    assert passed is True
    assert failures == []
    assert "samples_per_sec" not in pass_band.get("floors", {})


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


def test_resolve_measured_for_gate_uses_aggregate_means_for_repeats() -> None:
    runs = _sample_runs()
    aggregate = aggregate_e2e_run_payloads(runs)
    measured = resolve_e2e_measured_for_gate(
        repeats=3,
        run_payloads=runs,
        aggregate=aggregate,
    )

    assert measured == {
        key: float(aggregate[key]["mean"])
        for key in ("env_steps_per_sec", "samples_per_sec", "seconds_per_update_mean")
    }


def test_resolve_measured_for_gate_uses_first_run_when_single_repeat() -> None:
    runs = _sample_runs()
    measured = resolve_e2e_measured_for_gate(
        repeats=1,
        run_payloads=runs,
        aggregate=None,
    )

    assert measured == {
        "env_steps_per_sec": 4000.0,
        "samples_per_sec": 8000.0,
        "seconds_per_update_mean": 1.0,
    }


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
    gap = loaded.get("gap_assessment")
    assert gap is not None
    assert (
        gap.get("ablation_artifact") == "docs/benchmarks/launch-hygiene-ablation.json"
    )


_ABLATION_VERDICT_RANK = {"VERIFIED": 2, "NOT_VERIFIED": 1, "INCONCLUSIVE": 0}
_ABLATION_REQUIRED_ARM_KEYS = (
    "label",
    "commit_sha",
    "launch_hygiene",
    "learn_proof",
    "throughput_e2e",
)
_LEARN_PROOF_KEYS = ("artifact", "model", "through", "verdict", "stages")


def test_committed_launch_hygiene_ablation_artifact() -> None:
    """Committed learner-ablation snapshot (assessed_date in JSON).

    Re-run arms per docs/operator-runbook.md, then refresh
    docs/benchmarks/launch-hygiene-ablation.json from captured learn-proof outputs.
    """
    path = Path("docs/benchmarks/launch-hygiene-ablation.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["gate"] == "launch_hygiene_learner_ablation"
    for field in (
        "hot_path_status",
        "phase_b_status",
        "thresholds_source",
        "tier2_status",
    ):
        value = payload.get(field)
        assert isinstance(value, str) and value.strip(), f"missing or empty {field}"
    assert Path(payload["thresholds_source"]).is_file()

    arms = payload["arms"]
    assert isinstance(arms, dict) and arms
    winner = payload["winner"]
    assert winner in arms

    missing_artifacts: list[str] = []
    for arm_key, arm in arms.items():
        assert isinstance(arm, dict)
        for key in _ABLATION_REQUIRED_ARM_KEYS:
            assert key in arm, f"{arm_key} missing {key}"
        learn_proof = arm["learn_proof"]
        for key in _LEARN_PROOF_KEYS:
            assert key in learn_proof, f"{arm_key}.learn_proof missing {key}"
        verdict = learn_proof["verdict"]
        assert verdict in _ABLATION_VERDICT_RANK
        artifact_path = Path(learn_proof["artifact"])
        assert not artifact_path.is_absolute(), (
            f"{arm_key} learn_proof.artifact must be repo-relative"
        )
        if artifact_path.is_file():
            ref = json.loads(artifact_path.read_text(encoding="utf-8"))
            assert ref.get("verdict") == verdict
        else:
            missing_artifacts.append(f"{arm_key}:{artifact_path}")

    if missing_artifacts:
        pytest.skip(
            "learn-proof artifacts missing (capture per docs/operator-runbook.md "
            f"learner ablation table): {', '.join(missing_artifacts)}"
        )

    winner_verdict = arms[winner]["learn_proof"]["verdict"]
    for arm_key, arm in arms.items():
        if arm_key == winner:
            continue
        other = arm["learn_proof"]["verdict"]
        assert _ABLATION_VERDICT_RANK[winner_verdict] >= _ABLATION_VERDICT_RANK[other]

    assert isinstance(payload.get("decision"), str) and payload["decision"].strip()


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
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    loaded = load_e2e_baseline(path)
    assert loaded["gate"] == E2E_THROUGHPUT_GATE
    assert len(loaded["runs"]) == 3


_ANCHOR_SHA = "79162a2088160b8ed05c3e3a050e064c7f6c9556"
_MANIFEST_VERDICT = frozenset(
    {"admit", "admit_stack", "reject", "pending", "deferred", "operator_skip"}
)
_MANIFEST_PHASE = frozenset({"env_parity", "learning"})
_MANIFEST_INTEGRATION_STATUS = frozenset({"building", "ready_for_main"})
_MANIFEST_BASELINE_GATE_KEYS = frozenset({"throughput_e2e", "learn_proof", "parity"})
_MANIFEST_REQUIRED_TOP_LEVEL = (
    "manifest_id",
    "assessed_date",
    "baseline_sha",
    "baseline_branch",
    "baseline_gates",
    "criterion",
    "candidates",
    "integration_state",
    "decision",
)


def test_committed_cherry_pick_manifest_artifact() -> None:
    """Committed cherry-pick manifest scaffold (assessed_date in JSON).

    Populate baseline_gates and candidates[] after U1/U4–U5 gate captures per
    docs/solutions/workflow-issues/nuclear-cherry-pick-manifest-baseline-integration.md.
    """
    path = Path("docs/benchmarks/cherry-pick-manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for field in _MANIFEST_REQUIRED_TOP_LEVEL:
        assert field in payload, f"missing top-level field {field}"
    assert payload["baseline_sha"] == _ANCHOR_SHA
    baseline_path = Path("docs/benchmarks/launch-hygiene-e2e-baseline.json")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["commit_sha"] == payload["baseline_sha"]

    baseline_gates = payload["baseline_gates"]
    assert isinstance(baseline_gates, dict)
    assert set(baseline_gates) == _MANIFEST_BASELINE_GATE_KEYS
    for gate_key, gate in baseline_gates.items():
        assert isinstance(gate, dict), f"baseline_gates.{gate_key} must be object"
        verdict = gate.get("verdict")
        assert verdict in _MANIFEST_VERDICT, (
            f"baseline_gates.{gate_key}.verdict invalid: {verdict!r}"
        )
        artifact = gate.get("artifact")
        if artifact is not None:
            artifact_path = Path(str(artifact))
            assert not artifact_path.is_absolute(), (
                f"baseline_gates.{gate_key}.artifact must be repo-relative"
            )
    throughput = baseline_gates["throughput_e2e"]
    assert throughput.get("preset") == "tier2_primary"
    assert throughput.get("source") == str(baseline_path)

    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    for idx, candidate in enumerate(candidates):
        assert isinstance(candidate, dict), f"candidates[{idx}] must be object"
        phase = candidate.get("phase")
        if phase is not None:
            assert phase in _MANIFEST_PHASE, f"candidates[{idx}].phase invalid"
        verdict = candidate.get("verdict")
        if verdict is not None:
            assert verdict in _MANIFEST_VERDICT, (
                f"candidates[{idx}].verdict invalid: {verdict!r}"
            )
        for block_key in ("throughput_e2e", "learn_proof", "parity"):
            block = candidate.get(block_key)
            if not isinstance(block, dict):
                continue
            block_verdict = block.get("verdict")
            if block_verdict is not None:
                assert block_verdict in _MANIFEST_VERDICT
            artifact = block.get("artifact")
            if artifact is not None:
                artifact_path = Path(str(artifact))
                assert not artifact_path.is_absolute()

    integration = payload["integration_state"]
    assert isinstance(integration, dict)
    assert integration.get("branch") in {
        "throughput-baseline-integration",
        "optimize/opponent-rollout-throughput",
    }
    status = integration.get("integration_status")
    assert status in _MANIFEST_INTEGRATION_STATUS, (
        f"integration_status invalid: {status!r}"
    )
    ordered = integration.get("ordered_shas")
    assert isinstance(ordered, list)
    assert isinstance(payload.get("decision"), str) and payload["decision"].strip()
