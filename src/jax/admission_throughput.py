"""Extract launch-hygiene throughput from training JSONL without a second GPU run."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from src.jax.preflight import read_jsonl_records
from src.benchmark.training import (
    DEFAULT_E2E_WITHIN_PCT,
    E2E_THROUGHPUT_METRICS,
    compare_e2e_throughput_to_baseline,
    load_e2e_baseline,
    resolve_e2e_measured_for_gate,
    resolve_e2e_pass_band,
)

ADMISSION_THROUGHPUT_GATE = "admission_throughput"
DEFAULT_WARMUP = 2
# Matches ``ow benchmark training --updates`` (measured rows after warmup).
DEFAULT_MEASURED_UPDATE_COUNT = 20


@dataclass(frozen=True, slots=True)
class ThroughputWindow:
    """Measured update window after JIT warmup (launch hygiene convention)."""

    warmup: int
    max_measured_update: int

    @property
    def first_update(self) -> int:
        return self.warmup + 1

    @classmethod
    def from_training_benchmark(
        cls,
        *,
        warmup: int = DEFAULT_WARMUP,
        measured_update_count: int = DEFAULT_MEASURED_UPDATE_COUNT,
    ) -> ThroughputWindow:
        """Build window aligned with ``run_training_benchmark`` (--updates = measured count)."""

        return cls(warmup=warmup, max_measured_update=warmup + measured_update_count)

    def includes(self, update: int) -> bool:
        return self.first_update <= update <= self.max_measured_update


def resolve_log_path_from_input(path: Path) -> tuple[Path, Path | None]:
    """Resolve a jax jsonl path from a log file or gate-result JSON."""

    if not path.is_file():
        raise FileNotFoundError(f"input not found: {path}")
    if path.name.endswith("_jax.jsonl") or path.suffix == ".jsonl":
        return path, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    stage = payload.get("stage")
    if isinstance(stage, dict):
        log_path = stage.get("log_path")
        if isinstance(log_path, str) and log_path:
            return Path(log_path), path
    log_path = payload.get("log_path")
    if isinstance(log_path, str) and log_path:
        return Path(log_path), path
    raise ValueError(
        f"no log_path in gate result {path}; pass a *_jax.jsonl path directly"
    )


def _record_update(record: Mapping[str, object]) -> int | None:
    update = record.get("update")
    if isinstance(update, int):
        return update
    if isinstance(update, float) and update.is_integer():
        return int(update)
    return None


def _record_float(record: Mapping[str, object], key: str) -> float | None:
    value = record.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def _env_steps_for_record(record: Mapping[str, object]) -> float | None:
    direct = _record_float(record, "env_steps")
    if direct is not None:
        return direct
    update_seconds = _record_float(record, "update_seconds")
    env_steps_per_sec = _record_float(record, "env_steps_per_sec")
    if update_seconds is not None and env_steps_per_sec is not None:
        return env_steps_per_sec * update_seconds
    return None


def extract_throughput_from_records(
    records: Sequence[Mapping[str, object]],
    *,
    window: ThroughputWindow | None = None,
) -> dict[str, object]:
    """Aggregate throughput metrics from per-update JSONL rows."""

    resolved_window = window or ThroughputWindow.from_training_benchmark()
    selected: list[Mapping[str, object]] = []
    for record in records:
        update = _record_update(record)
        if update is None or not resolved_window.includes(update):
            continue
        if _record_float(record, "update_seconds") is None:
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
        update_seconds = _record_float(record, "update_seconds")
        assert update_seconds is not None
        seconds_total += update_seconds
        update_seconds_values.append(update_seconds)
        env_steps = _env_steps_for_record(record)
        if env_steps is not None:
            env_steps_total += env_steps
        samples = _record_float(record, "samples")
        if samples is not None:
            samples_total += samples
        rollout_seconds = _record_float(record, "rollout_seconds")
        if rollout_seconds is not None:
            rollout_seconds_values.append(rollout_seconds)
        ppo_seconds = _record_float(record, "ppo_seconds")
        if ppo_seconds is not None:
            ppo_seconds_values.append(ppo_seconds)

    measured_updates = len(selected)
    payload: dict[str, object] = {
        "gate": ADMISSION_THROUGHPUT_GATE,
        "warmup": resolved_window.warmup,
        "max_measured_update": resolved_window.max_measured_update,
        "measured_updates": measured_updates,
        "updates_in_window": sorted(_record_update(record) for record in selected),
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


def _baseline_rollout_geometry(
    baseline: Mapping[str, object],
) -> tuple[int, int] | None:
    runs = baseline.get("runs")
    if not isinstance(runs, list) or not runs:
        return None
    first = runs[0]
    if not isinstance(first, dict):
        return None
    rollout_steps = first.get("rollout_steps")
    num_envs = first.get("num_envs")
    if isinstance(rollout_steps, int) and isinstance(num_envs, int):
        return rollout_steps, num_envs
    return None


def _measured_env_steps_per_update(payload: Mapping[str, object]) -> float | None:
    env_steps = payload.get("env_steps")
    measured_updates = payload.get("measured_updates")
    if not isinstance(env_steps, int | float):
        return None
    if not isinstance(measured_updates, int) or measured_updates <= 0:
        return None
    return float(env_steps) / measured_updates


def validate_throughput_baseline_geometry(
    payload: Mapping[str, object],
    baseline: Mapping[str, object],
    *,
    tolerance: float = 0.10,
) -> list[str]:
    """Reject apples-to-oranges baseline compares (e.g. 500-step baseline vs 256-step gate)."""

    runs = baseline.get("runs")
    baseline_geometry = _baseline_rollout_geometry(baseline)
    measured_per_update = _measured_env_steps_per_update(payload)
    if isinstance(runs, list) and runs and baseline_geometry is None:
        return [
            "throughput baseline runs missing rollout_steps/num_envs geometry metadata"
        ]
    if baseline_geometry is None or measured_per_update is None:
        return []
    rollout_steps, num_envs = baseline_geometry
    expected_per_update = rollout_steps * num_envs
    if expected_per_update <= 0:
        return []
    ratio = measured_per_update / expected_per_update
    if abs(ratio - 1.0) <= tolerance:
        return []
    return [
        "throughput geometry mismatch: measured "
        f"~{measured_per_update:.0f} env-steps/update vs baseline recipe "
        f"{rollout_steps} rollout_steps × {num_envs} envs "
        f"(={expected_per_update}); ratio {ratio:.2f} outside "
        f"±{tolerance * 100:.0f}% — recapture baseline with "
        "`ow benchmark training --preset admission` (and matching --overrides) "
        "or align gate --train-overrides"
    ]


def apply_baseline_comparison(
    payload: dict[str, object],
    *,
    baseline_path: Path,
    within_pct: float | None,
) -> tuple[dict[str, object], bool]:
    baseline = load_e2e_baseline(baseline_path)
    geometry_failures = validate_throughput_baseline_geometry(payload, baseline)
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
    if geometry_failures:
        passed = False
        failures = [*geometry_failures, *failures]
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
