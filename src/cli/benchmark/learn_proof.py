"""``ow benchmark learn-proof`` composer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.cli.benchmark.common import LEARN_PROOF_PRIMITIVES, REPO_ROOT, _git_head_sha
from src.cli.benchmark.tournament_proof import run_tournament_proof_cli

def _learn_proof_primitive_payload() -> dict[str, object]:
    return {
        "workflow": "ow benchmark learn-proof",
        "prefer_primitives": True,
        "primitives": list(LEARN_PROOF_PRIMITIVES),
        "gate_list_command": "uv run ow benchmark gate --list",
    }


def _resolve_learn_proof_gates(args: argparse.Namespace) -> tuple[str, ...]:
    from src.jax.preflight import GATE_ORDER

    if args.steps:
        if args.gate is not None or args.through is not None:
            raise SystemExit("Use only one of --steps, --gate, or --through.")
        requested = tuple(
            item.strip()
            for item in str(args.steps).split(",")
            if item.strip()
        )
        if not requested:
            raise SystemExit("--steps requires at least one gate id.")
        unknown = [gate_id for gate_id in requested if gate_id not in GATE_ORDER]
        if unknown:
            raise SystemExit(
                f"Unknown learn-proof step(s): {', '.join(unknown)} "
                f"(expected subset of {', '.join(GATE_ORDER)})"
            )
        return tuple(gate_id for gate_id in GATE_ORDER if gate_id in requested)
    if args.gate is not None and args.through is not None:
        raise SystemExit("Use only one of --gate or --through.")
    if args.gate is not None:
        return (str(args.gate),)
    through = args.through or "beat_random"
    stop_index = GATE_ORDER.index(through)
    return GATE_ORDER[: stop_index + 1]


def run_learn_proof_cli(args: argparse.Namespace) -> int:
    """Thin composer over gate-run and tournament-proof primitives."""

    from src.jax.preflight import (
        GATE_ORDER,
        PreflightVerdict,
        gate_evaluation_to_dict,
        run_preflight_gate,
        write_report,
    )

    if args.print_primitives:
        print(json.dumps(_learn_proof_primitive_payload(), indent=2))
        return 0

    if args.eval_checkpoint is not None:
        return run_tournament_proof_cli(args)

    selected_gates = _resolve_learn_proof_gates(args)

    extra_train_overrides = tuple(args.train_overrides)
    started = __import__("time").perf_counter()
    evaluations = []
    overall_verdict = PreflightVerdict.VERIFIED
    for gate_id in selected_gates:
        gate_model = (
            "transformer_factorized"
            if gate_id == "curriculum_staged" and args.model != "planet_flow_target_heatmap"
            else args.model
        )
        evaluation = run_preflight_gate(
            gate_id,
            model=gate_model,
            output_root=args.output_root,
            repo_root=REPO_ROOT,
            dry_run=args.dry_run,
            thresholds_path=args.thresholds_path,
            profiles_path=args.profile_path,
            extra_train_overrides=extra_train_overrides,
        )
        evaluations.append(evaluation)
        if evaluation.verdict != PreflightVerdict.VERIFIED:
            overall_verdict = evaluation.verdict
            break
    stages = [gate_evaluation_to_dict(item) for item in evaluations]

    report: dict[str, object] = {
        "gate": "learn_proof",
        "commit_sha": _git_head_sha(),
        "seconds_total": __import__("time").perf_counter() - started,
        "verdict": overall_verdict.value,
        "through": args.through or args.gate or ",".join(selected_gates),
        "steps": list(selected_gates),
        "model": args.model,
        "gate_order": list(GATE_ORDER),
        "stages": stages,
        **_learn_proof_primitive_payload(),
    }
    write_report(args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if overall_verdict == PreflightVerdict.VERIFIED else 1

