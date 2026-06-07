"""``ow benchmark rollout-phase-breakdown`` — rollout cost itemization from JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.benchmark.jsonl_window import ThroughputWindow
from src.jax.rollout.phase_timing_report import (
    compare_rollout_phase_breakdowns,
    extract_rollout_phase_breakdown_from_input,
    format_rollout_phase_breakdown,
)


def run_rollout_phase_breakdown_cli(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    try:
        window = ThroughputWindow(
            warmup=int(args.warmup),
            max_measured_update=int(args.max_measured_update),
        )
        payload = extract_rollout_phase_breakdown_from_input(input_path, window=window)
        baseline_path = Path(args.baseline) if args.baseline is not None else None
        min_drop = args.min_opponent_drop_points
        if baseline_path is None and min_drop is not None:
            raise ValueError("--min-opponent-drop-points requires --baseline")
        if baseline_path is not None:
            baseline = extract_rollout_phase_breakdown_from_input(
                baseline_path, window=window
            )
            payload["comparison"] = compare_rollout_phase_breakdowns(
                baseline,
                payload,
                min_opponent_drop_points=min_drop,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_rollout_phase_breakdown(payload))
    return 0
