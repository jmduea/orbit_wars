"""``ow benchmark admission-throughput`` — throughput from gate JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.jax.admission_throughput import (
    ThroughputWindow,
    resolve_log_path_from_input,
    run_throughput_gate,
)


def run_admission_throughput_cli(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    try:
        log_path, gate_result_path = resolve_log_path_from_input(input_path)
        window = ThroughputWindow(
            warmup=int(args.warmup),
            max_measured_update=int(args.max_measured_update),
        )
        payload, exit_code = run_throughput_gate(
            log_path,
            baseline_path=args.baseline,
            within_pct=args.assert_within_pct,
            window=window,
        )
        if gate_result_path is not None:
            payload["gate_result_path"] = str(gate_result_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if exit_code != 0:
        # pyrefly: ignore [not-iterable]
        for reason in payload.get("gate_failures", []):
            print(str(reason), file=sys.stderr)

    print(json.dumps(payload, indent=2))
    return exit_code
