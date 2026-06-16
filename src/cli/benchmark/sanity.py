"""``ow benchmark sanity`` command."""

from __future__ import annotations

import argparse
import json

from src.cli.benchmark.common import _git_head_sha, _init_benchmark_runtime


def run_sanity_cli(args: argparse.Namespace) -> int:
    import jax
    from src.benchmark.training import (
        WORKSTATION_VALIDATION_OVERRIDES,
        compose_benchmark_config,
        run_training_benchmark,
        training_benchmark_payload,
    )
    from src.jax.preflight import (
        PreflightVerdict,
        compare_repro_snapshots,
        write_report,
    )

    _init_benchmark_runtime()

    overrides = (
        list(args.overrides)
        if args.overrides is not None
        else list(WORKSTATION_VALIDATION_OVERRIDES)
    )
    cfg = compose_benchmark_config(overrides)
    snapshot_updates = frozenset({args.compare_update})
    first = run_training_benchmark(
        cfg,
        label="sanity_repro_a",
        overrides=tuple(overrides),
        warmup=args.warmup,
        updates=args.updates,
        snapshot_updates=snapshot_updates,
    )
    second = run_training_benchmark(
        cfg,
        label="sanity_repro_b",
        overrides=tuple(overrides),
        warmup=args.warmup,
        updates=args.updates,
        snapshot_updates=snapshot_updates,
    )
    verdict, reasons = compare_repro_snapshots(
        training_benchmark_payload(first),
        training_benchmark_payload(second),
        update=args.compare_update,
    )
    report: dict[str, object] = {
        "gate": "sanity_repro",
        "commit_sha": _git_head_sha(),
        "jax_version": jax.__version__,
        "verdict": verdict.value,
        "reasons": list(reasons),
        "compare_update": args.compare_update,
        "run_a": training_benchmark_payload(first),
        "run_b": training_benchmark_payload(second),
    }
    write_report(args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if verdict == PreflightVerdict.VERIFIED else 1
