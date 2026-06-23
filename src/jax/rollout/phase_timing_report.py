"""Extract rollout phase timing breakdown from training JSONL (opt-in telemetry)."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from src.benchmark.jsonl_window import (
    ThroughputWindow,
    record_float,
    record_update,
    resolve_log_path_from_input,
)
from src.jax.preflight_calibration import read_jsonl_records

PHASE_NAMES: tuple[str, ...] = (
    "policy",
    "opponent",
    "env_step",
    "reset",
    "post_step",
)
OPPONENT_DETAIL_NAMES: tuple[str, ...] = (
    "opponent_sample",
    "opponent_encode",
)

PHASE_SECOND_KEYS: tuple[str, ...] = tuple(
    f"rollout_phase_{name}_seconds" for name in PHASE_NAMES
)
PHASE_FRACTION_KEYS: tuple[str, ...] = tuple(
    f"rollout_phase_{name}_fraction" for name in PHASE_NAMES
)
OPPONENT_DETAIL_SECOND_KEYS: tuple[str, ...] = tuple(
    f"rollout_phase_{name}_seconds" for name in OPPONENT_DETAIL_NAMES
)
OPPONENT_DETAIL_FRACTION_KEYS: tuple[str, ...] = tuple(
    f"rollout_phase_{name}_fraction" for name in OPPONENT_DETAIL_NAMES
)
MEASURED_TOTAL_KEY = "rollout_phase_measured_total_seconds"


@dataclass(frozen=True, slots=True)
class PhaseTimingWindow:
    """Update window for steady-state phase averages (launch hygiene convention)."""

    warmup: int
    max_measured_update: int

    @property
    def first_update(self) -> int:
        return self.warmup + 1

    def includes(self, update: int) -> bool:
        return self.first_update <= update <= self.max_measured_update


def _window_from_throughput(
    window: ThroughputWindow | PhaseTimingWindow | None,
) -> PhaseTimingWindow:
    if window is None:
        return PhaseTimingWindow(
            warmup=2,
            max_measured_update=22,
        )
    if isinstance(window, PhaseTimingWindow):
        return window
    return PhaseTimingWindow(
        warmup=window.warmup,
        max_measured_update=window.max_measured_update,
    )


def _has_phase_timing(record: Mapping[str, object]) -> bool:
    return record_float(record, PHASE_SECOND_KEYS[0]) is not None


def extract_rollout_phase_breakdown_from_records(
    records: Sequence[Mapping[str, object]],
    *,
    window: ThroughputWindow | PhaseTimingWindow | None = None,
) -> dict[str, object]:
    """Aggregate per-phase rollout seconds and fractions from JSONL rows."""

    resolved = _window_from_throughput(window)
    selected: list[Mapping[str, object]] = []
    for record in records:
        update = record_update(record)
        if update is None or not resolved.includes(update):
            continue
        if not _has_phase_timing(record):
            continue
        selected.append(record)

    if not selected:
        raise ValueError(
            "no rollout phase timing rows in window "
            f"updates {resolved.first_update}–{resolved.max_measured_update}. "
            "Run `ow benchmark rollout-phase-profile` (integration worktree), "
            "then `ow benchmark rollout-phase-breakdown` on the output JSONL."
        )

    phase_seconds: dict[str, list[float]] = {name: [] for name in PHASE_NAMES}
    phase_fractions: dict[str, list[float]] = {name: [] for name in PHASE_NAMES}
    opponent_detail_seconds: dict[str, list[float]] = {
        name: [] for name in OPPONENT_DETAIL_NAMES
    }
    opponent_detail_fractions: dict[str, list[float]] = {
        name: [] for name in OPPONENT_DETAIL_NAMES
    }
    measured_totals: list[float] = []
    rollout_seconds: list[float] = []

    for record in selected:
        for name, sec_key, frac_key in zip(
            PHASE_NAMES, PHASE_SECOND_KEYS, PHASE_FRACTION_KEYS, strict=True
        ):
            sec = record_float(record, sec_key)
            frac = record_float(record, frac_key)
            if sec is not None:
                phase_seconds[name].append(sec)
            if frac is not None:
                phase_fractions[name].append(frac)
        for name, sec_key, frac_key in zip(
            OPPONENT_DETAIL_NAMES,
            OPPONENT_DETAIL_SECOND_KEYS,
            OPPONENT_DETAIL_FRACTION_KEYS,
            strict=True,
        ):
            sec = record_float(record, sec_key)
            frac = record_float(record, frac_key)
            if sec is not None:
                opponent_detail_seconds[name].append(sec)
            if frac is not None:
                opponent_detail_fractions[name].append(frac)
        measured = record_float(record, MEASURED_TOTAL_KEY)
        if measured is not None:
            measured_totals.append(measured)
        rollout_s = record_float(record, "rollout_seconds")
        if rollout_s is not None:
            rollout_seconds.append(rollout_s)

    def _mean(values: list[float]) -> float:
        return statistics.fmean(values) if values else 0.0

    phases_payload: dict[str, object] = {}
    for name in PHASE_NAMES:
        phases_payload[name] = {
            "seconds_mean": _mean(phase_seconds[name]),
            "fraction_mean": _mean(phase_fractions[name]),
        }
    opponent_details_payload: dict[str, object] = {}
    for name in OPPONENT_DETAIL_NAMES:
        opponent_details_payload[name] = {
            "seconds_mean": _mean(opponent_detail_seconds[name]),
            "fraction_mean": _mean(opponent_detail_fractions[name]),
        }

    measured_mean = _mean(measured_totals)
    rollout_mean = _mean(rollout_seconds)
    payload: dict[str, object] = {
        "warmup": resolved.warmup,
        "max_measured_update": resolved.max_measured_update,
        "measured_updates": len(selected),
        "updates_in_window": sorted(
            u for u in (record_update(record) for record in selected) if u is not None
        ),
        "phases": phases_payload,
        "opponent_details": opponent_details_payload,
        "rollout_phase_measured_total_seconds_mean": measured_mean,
        "rollout_seconds_mean": rollout_mean,
        "rollout_seconds_gap_mean": rollout_mean - measured_mean,
    }
    return payload


def compare_rollout_phase_breakdowns(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    min_opponent_drop_points: float | None = None,
) -> dict[str, object]:
    """Compare opponent phase share, expressed in percentage points."""

    baseline_fraction = _phase_fraction_mean(baseline, "opponent")
    candidate_fraction = _phase_fraction_mean(candidate, "opponent")
    drop_points = (baseline_fraction - candidate_fraction) * 100.0
    payload: dict[str, object] = {
        "baseline_opponent_fraction": baseline_fraction,
        "candidate_opponent_fraction": candidate_fraction,
        "opponent_fraction_drop_points": drop_points,
    }
    if min_opponent_drop_points is not None:
        minimum = float(min_opponent_drop_points)
        payload["min_opponent_drop_points"] = minimum
        payload["passed"] = drop_points >= minimum
    return payload


def _phase_fraction_mean(payload: Mapping[str, object], name: str) -> float:
    phases = payload.get("phases")
    if not isinstance(phases, Mapping):
        return 0.0
    entry = phases.get(name)
    if not isinstance(entry, Mapping):
        return 0.0
    return float(entry.get("fraction_mean", 0.0))


def extract_rollout_phase_breakdown_from_log(
    log_path: Path,
    *,
    window: ThroughputWindow | PhaseTimingWindow | None = None,
) -> dict[str, object]:
    records = read_jsonl_records(log_path)
    payload = extract_rollout_phase_breakdown_from_records(records, window=window)
    payload["log_path"] = str(log_path)
    return payload


def _profile_records_from_payload(
    payload: Mapping[str, object],
) -> list[dict[str, object]] | None:
    raw = payload.get("per_update_records")
    if not isinstance(raw, list) or not raw:
        return None
    records: list[dict[str, object]] = []
    for item in raw:
        if isinstance(item, dict):
            records.append(dict(item))
    return records or None


def extract_rollout_phase_breakdown_from_input(
    path: Path,
    *,
    window: ThroughputWindow | PhaseTimingWindow | None = None,
) -> dict[str, object]:
    """Load phase rows from profile JSON, gate JSON (via log_path), or JSONL."""

    if not path.is_file():
        raise FileNotFoundError(f"input not found: {path}")
    if path.name.endswith("_jax.jsonl") or path.suffix == ".jsonl":
        return extract_rollout_phase_breakdown_from_log(path, window=window)

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")

    profile_records = _profile_records_from_payload(payload)
    if profile_records is not None:
        resolved_window = window
        if resolved_window is None:
            warmup = payload.get("warmup")
            max_update = payload.get("max_measured_update")
            resolved_window = PhaseTimingWindow(
                warmup=int(warmup) if isinstance(warmup, int | float) else 2,
                max_measured_update=(
                    int(max_update) if isinstance(max_update, int | float) else 20
                ),
            )
        breakdown = extract_rollout_phase_breakdown_from_records(
            profile_records,
            window=resolved_window,
        )
        breakdown["source_path"] = str(path)
        breakdown["preset"] = payload.get("preset")
        return breakdown

    log_path, gate_result_path = resolve_log_path_from_input(path)
    breakdown = extract_rollout_phase_breakdown_from_log(log_path, window=window)
    if gate_result_path is not None:
        breakdown["gate_result_path"] = str(gate_result_path)
    return breakdown


def resolve_input_to_log_path(path: Path) -> tuple[Path, Path | None]:
    return resolve_log_path_from_input(path)


def format_rollout_phase_breakdown(payload: Mapping[str, object]) -> str:
    """Human-readable table for terminal output."""

    lines = [
        (
            f"Rollout phase breakdown "
            f"(updates {payload['warmup'] + 1}–{payload['max_measured_update']}, "
            f"n={payload['measured_updates']})"
        ),
        f"  source: {payload.get('log_path') or payload.get('source_path', '(unknown)')}",
        "",
        f"  {'Phase':<12} {'Mean s':>8}  {'Share':>7}",
        f"  {'-' * 12} {'-' * 8}  {'-' * 7}",
    ]
    phases = payload.get("phases")
    if isinstance(phases, dict):
        for name in PHASE_NAMES:
            entry = phases.get(name)
            if not isinstance(entry, dict):
                continue
            sec = float(entry.get("seconds_mean", 0.0))
            frac = float(entry.get("fraction_mean", 0.0))
            lines.append(f"  {name:<12} {sec:8.3f}  {frac * 100:6.1f}%")
    opponent_details = payload.get("opponent_details")
    if isinstance(opponent_details, dict):
        lines.extend(
            [
                "",
                f"  {'Opponent detail':<16} {'Mean s':>8}  {'Share':>7}",
                f"  {'-' * 16} {'-' * 8}  {'-' * 7}",
            ]
        )
        for name in OPPONENT_DETAIL_NAMES:
            entry = opponent_details.get(name)
            if not isinstance(entry, dict):
                continue
            sec = float(entry.get("seconds_mean", 0.0))
            frac = float(entry.get("fraction_mean", 0.0))
            lines.append(f"  {name:<16} {sec:8.3f}  {frac * 100:6.1f}%")
    measured = float(payload.get("rollout_phase_measured_total_seconds_mean", 0.0))
    rollout = float(payload.get("rollout_seconds_mean", 0.0))
    gap = float(payload.get("rollout_seconds_gap_mean", 0.0))
    lines.extend(
        [
            "",
            f"  measured total (instrumented): {measured:.3f}s",
            f"  rollout_seconds (wall):        {rollout:.3f}s",
            f"  gap (host/sync overhead):      {gap:.3f}s",
        ]
    )
    comparison = payload.get("comparison")
    if isinstance(comparison, Mapping):
        baseline = float(comparison.get("baseline_opponent_fraction", 0.0)) * 100.0
        candidate = float(comparison.get("candidate_opponent_fraction", 0.0)) * 100.0
        drop = float(comparison.get("opponent_fraction_drop_points", 0.0))
        lines.extend(
            [
                "",
                "  comparison:",
                f"    baseline opponent share:  {baseline:.1f}%",
                f"    candidate opponent share: {candidate:.1f}%",
                f"    drop:                     {drop:.1f} points",
            ]
        )
        if "min_opponent_drop_points" in comparison:
            minimum = float(comparison.get("min_opponent_drop_points", 0.0))
            passed = bool(comparison.get("passed", False))
            status = "pass" if passed else "fail"
            lines.append(f"    minimum drop:             {minimum:.1f} points ({status})")
    return "\n".join(lines)
