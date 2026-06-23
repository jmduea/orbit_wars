"""Shared JSONL windowing helpers for benchmark post-hoc extractors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

DEFAULT_WARMUP = 2
# Matches ``ow benchmark training --updates`` (measured rows after warmup).
DEFAULT_MEASURED_UPDATE_COUNT = 20
DEFAULT_MAX_MEASURED_UPDATE = DEFAULT_WARMUP + DEFAULT_MEASURED_UPDATE_COUNT


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


def default_throughput_window() -> ThroughputWindow:
    return ThroughputWindow(
        warmup=DEFAULT_WARMUP,
        max_measured_update=DEFAULT_MAX_MEASURED_UPDATE,
    )


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


def record_update(record: Mapping[str, object]) -> int | None:
    update = record.get("update")
    if isinstance(update, int):
        return update
    if isinstance(update, float) and update.is_integer():
        return int(update)
    return None


def record_float(record: Mapping[str, object], key: str) -> float | None:
    value = record.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None
