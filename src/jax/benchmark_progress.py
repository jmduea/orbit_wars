"""stderr progress lines for long ``ow benchmark`` subprocess workflows."""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone


def emit_benchmark_progress(message: str) -> None:
    """Print a flushed progress line to stderr (safe when stdout carries JSON)."""
    print(message, file=sys.stderr, flush=True)


def emit_benchmark_progress_ts(message: str) -> None:
    """Progress line prefixed with an ISO timestamp (UTC)."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    emit_benchmark_progress(f"[{stamp}] {message}")


def total_updates_from_overrides(overrides: list[str] | tuple[str, ...]) -> int | None:
    """Best-effort parse of ``training.total_updates`` from Hydra override strings."""
    for item in overrides:
        match = re.fullmatch(r"training\.total_updates=(\d+)", str(item).strip())
        if match:
            return int(match.group(1))
    return None
