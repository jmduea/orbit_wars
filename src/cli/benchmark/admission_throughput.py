"""``ow benchmark admission-throughput`` — throughput from gate JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.jax.admission_throughput import (
    ThroughputWindow,
    apply_baseline_comparison,
    default_within_pct_for_assert,
    extract_throughput_from_log,
    resolve_log_path_from_input,
)


def run_admission_throughput_cli(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    try:
        log_path, gate_result_path = resolve_log_path_from_input(input_path)
        window = ThroughputWindow(
            warmup=int(args.warmup),
            max_measured_update=int(args.max_measured_update),
        )
        payload = extract_throughput_from_log(log_path, window=window)
        if gate_result_path is not None:
            payload["gate_result_path"] = str(gate_result_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    within_pct = default_within_pct_for_assert(
        baseline_path=args.baseline,
        assert_within_pct=args.assert_within_pct,
    )
    if args.baseline is not None:
        try:
            payload, passed = apply_baseline_comparison(
                payload,
                baseline_path=args.baseline,
                within_pct=within_pct,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if not passed:
            for reason in payload.get("gate_failures", []):
                print(str(reason), file=sys.stderr)
            print(json.dumps(payload, indent=2))
            return 1

    print(json.dumps(payload, indent=2))
    return 0
