"""Extract launch-hygiene throughput from training JSONL without a second GPU run."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Mapping, Sequence

from src.benchmark.jsonl_window import (
    ThroughputWindow,
    default_throughput_window,
    record_float,
    record_update,
)
from src.benchmark.training import (
    DEFAULT_E2E_WITHIN_PCT,
    E2E_THROUGHPUT_METRICS,
    compare_e2e_throughput_to_baseline,
    load_e2e_baseline,
    resolve_e2e_measured_for_gate,
    resolve_e2e_pass_band,
)
from src.jax.preflight import read_jsonl_records

ADMISSION_THROUGHPUT_GATE = "admission_throughput"


def _env_steps_for_record(record: Mapping[str, object]) -> float | None:
    direct = record_float(record, "env_steps")
    if direct is not None:
        return direct
    update_seconds = record_float(record, "update_seconds")
    env_steps_per_sec = record_float(record, "env_steps_per_sec")
    if update_seconds is not None and env_steps_per_sec is not None:
        return env_steps_per_sec * update_seconds
    return None


def extract_throughput_from_records(
    records: Sequence[Mapping[str, object]],
    *,
    window: ThroughputWindow | None = None,
) -> dict[str, object]:
    """Aggregate throughput metrics from per-update JSONL rows."""

    resolved_window = window or default_throughput_window()
    selected: list[Mapping[str, object]] = []
    for record in records:
        update = record_update(record)
        if update is None or not resolved_window.includes(update):
            continue
        if record_float(record, "update_seconds") is None:
            continue
        selected.append(record)

    if not selected:
        raise ValueError(
            "no per-update timing rows in window "
            f"updates {resolved_window.first_update}–"
            f"{resolved_window.max_measured_update}"
        )

    seconds_total = 0.0
    env_steps_total = 0.0
    samples_total = 0.0
    update_seconds_values: list[float] = []
    rollout_seconds_values: list[float] = []
    ppo_seconds_values: list[float] = []

    for record in selected:
        update_seconds = record_float(record, "update_seconds")
        assert update_seconds is not None
        seconds_total += update_seconds
        update_seconds_values.append(update_seconds)
        env_steps = _env_steps_for_record(record)
        if env_steps is not None:
            env_steps_total += env_steps
        samples = record_float(record, "samples")
        if samples is not None:
            samples_total += samples
        rollout_seconds = record_float(record, "rollout_seconds")
        if rollout_seconds is not None:
            rollout_seconds_values.append(rollout_seconds)
        ppo_seconds = record_float(record, "ppo_seconds")
        if ppo_seconds is not None:
            ppo_seconds_values.append(ppo_seconds)

    measured_updates = len(selected)
    payload: dict[str, object] = {
        "gate": ADMISSION_THROUGHPUT_GATE,
        "warmup": resolved_window.warmup,
        "max_measured_update": resolved_window.max_measured_update,
        "measured_updates": measured_updates,
        "updates_in_window": sorted(record_update(record) for record in selected),
        "seconds_total": seconds_total,
        "seconds_per_update_mean": seconds_total / measured_updates,
        "env_steps": int(env_steps_total),
        "samples": int(samples_total),
        "env_steps_per_sec": env_steps_total / max(seconds_total, 1e-9),
        "samples_per_sec": samples_total / max(seconds_total, 1e-9),
    }
    if rollout_seconds_values:
        payload["rollout_seconds_per_update_mean"] = statistics.fmean(
            rollout_seconds_values
        )
    if ppo_seconds_values:
        payload["ppo_seconds_per_update_mean"] = statistics.fmean(ppo_seconds_values)
    return payload


def extract_throughput_from_log(
    log_path: Path,
    *,
    window: ThroughputWindow | None = None,
) -> dict[str, object]:
    records = read_jsonl_records(log_path)
    payload = extract_throughput_from_records(records, window=window)
    payload["log_path"] = str(log_path)
    return payload


def measured_for_baseline_gate(payload: Mapping[str, object]) -> dict[str, float]:
    return resolve_e2e_measured_for_gate(
        repeats=1,
        run_payloads=[dict(payload)],
        aggregate=None,
    )


def apply_baseline_comparison(
    payload: dict[str, object],
    *,
    baseline_path: Path,
    within_pct: float | None,
) -> tuple[dict[str, object], bool]:
    baseline = load_e2e_baseline(baseline_path)
    measured = measured_for_baseline_gate(payload)
    if not measured:
        missing = [key for key in E2E_THROUGHPUT_METRICS if key not in payload]
        raise ValueError(
            "throughput payload missing baseline metrics: " + ", ".join(missing)
        )
    pass_band = resolve_e2e_pass_band(baseline, within_pct=within_pct)
    passed, failures = compare_e2e_throughput_to_baseline(
        measured,
        pass_band=pass_band,
    )
    payload["baseline_path"] = str(baseline_path)
    payload["pass_band_applied"] = pass_band
    payload["measured_for_gate"] = measured
    payload["gate_passed"] = passed
    if failures:
        payload["gate_failures"] = failures
    return payload, passed


def default_within_pct_for_assert(
    *,
    baseline_path: Path | None,
    assert_within_pct: float | None,
) -> float | None:
    if baseline_path is None:
        return None
    if assert_within_pct is not None:
        return float(assert_within_pct)
    return DEFAULT_E2E_WITHIN_PCT


def throughput_verdict_from_payload(payload: Mapping[str, object]) -> str:
    """Map throughput extract + optional baseline compare to a gate verdict."""

    if payload.get("gate_passed") is True:
        return "VERIFIED"
    if payload.get("gate_passed") is False:
        return "NOT_VERIFIED"
    if payload.get("baseline_path") is None:
        return "INCONCLUSIVE"
    return "NOT_VERIFIED"


def run_throughput_gate(
    log_path: Path,
    *,
    baseline_path: Path | None = None,
    within_pct: float | None = None,
    window: ThroughputWindow | None = None,
) -> tuple[dict[str, object], int]:
    """Extract throughput from a gate JSONL and optionally compare to baseline."""

    payload = extract_throughput_from_log(log_path, window=window)
    resolved_within_pct = default_within_pct_for_assert(
        baseline_path=baseline_path,
        assert_within_pct=within_pct,
    )
    if baseline_path is not None:
        payload, passed = apply_baseline_comparison(
            payload,
            baseline_path=baseline_path,
            within_pct=resolved_within_pct,
        )
        payload["verdict"] = throughput_verdict_from_payload(payload)
        return payload, 0 if passed else 1
    payload["verdict"] = "INCONCLUSIVE"
    return payload, 0
