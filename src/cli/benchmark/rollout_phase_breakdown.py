"""``ow benchmark rollout-phase-breakdown`` — rollout cost itemization from JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.jax.rollout.phase_timing_report import (
    PhaseTimingWindow,
    extract_rollout_phase_breakdown_from_input,
    format_rollout_phase_breakdown,
)


def run_rollout_phase_breakdown_cli(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    try:
        window = PhaseTimingWindow(
            warmup=int(args.warmup),
            max_measured_update=int(args.max_measured_update),
        )
        payload = extract_rollout_phase_breakdown_from_input(input_path, window=window)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_rollout_phase_breakdown(payload))
    return 0
