"""``ow benchmark tournament proof`` command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.cli.benchmark.common import REPO_ROOT, _git_head_sha


def run_tournament_proof_cli(args: argparse.Namespace) -> int:
    from src.artifacts.docker_validation import (
        docker_gate_passed,
        run_submit_valid_docker_gate,
    )
    from src.artifacts.tournament.unified.ladder import run_unified_ladder
    from src.artifacts.tournament.unified.spec import load_unified_tournament_spec
    from src.jax.preflight import PreflightVerdict, write_report
    from src.jax.preflight_calibration import (
        default_calibration_json_path,
    )

    checkpoint = Path(args.eval_checkpoint)
    if not checkpoint.is_file():
        print(f"missing checkpoint: {checkpoint}", file=sys.stderr)
        return 1

    thresholds_path = args.thresholds_path or default_calibration_json_path(REPO_ROOT)
    spec = load_unified_tournament_spec(thresholds_path)
    has_unified_section = False
    if thresholds_path.is_file():
        payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
        has_unified_section = isinstance(payload.get("unified_tournament"), dict)

    output_dir = (
        args.output_root
        / "campaigns"
        / args.campaign
        / "evaluations"
        / "preflight_win_proof_unified"
    )
    docker_output_dir = output_dir / "docker_validation"

    if args.dry_run:
        if "4p_challenger_vs_baselines" in spec.stage1.formats:
            stage1_count = (
                len(spec.stage1.opponents)
                * len(spec.stage1.seeds)
                * spec.stage1.games_per_pair
                + len(spec.stage1.seeds) * spec.stage1.games_per_pair
            )
        else:
            stage1_count = (
                len(spec.stage1.opponents)
                * len(spec.stage1.seeds)
                * spec.stage1.games_per_pair
            )
        plan = {
            "gate": "win_proof",
            "verdict": PreflightVerdict.INCONCLUSIVE.value,
            "dry_run": True,
            "unified": True,
            "submit_valid_order": ["docker_validation", "unified_tournament_ladder"],
            "docker_output_dir": str(docker_output_dir),
            "enforcement": spec.enforcement,
            "needs_calibration": spec.needs_calibration,
            "stage1": {
                "opponents": list(spec.stage1.opponents),
                "seeds": list(spec.stage1.seeds),
                "games_per_pair": spec.stage1.games_per_pair,
                "scheduled_matches": stage1_count,
                "floors": dict(spec.stage1.floors),
            },
            "stage2": {
                "seeds": list(spec.stage2.seeds),
                "games_per_pair": spec.stage2.games_per_pair,
                "blocking_reason": spec.blocking_reason,
            },
            "output_dir": str(output_dir),
        }
        write_report(args.out, plan)
        print(json.dumps(plan, indent=2))
        return 0

    if spec.needs_calibration and not has_unified_section:
        report = {
            "gate": "win_proof",
            "commit_sha": _git_head_sha(),
            "verdict": PreflightVerdict.INCONCLUSIVE.value,
            "reasons": ["missing unified_tournament section in calibration JSON"],
            "checkpoint": str(checkpoint),
            "thresholds_path": str(thresholds_path),
            "evaluation_mode": "unified_tournament",
        }
        write_report(args.out, report)
        print(json.dumps(report, indent=2))
        return 1

    docker_manifest: dict[str, object] = {}
    try:
        docker_manifest = run_submit_valid_docker_gate(
            checkpoint_path=checkpoint,
            output_dir=docker_output_dir,
            repo_root=REPO_ROOT,
        )
    except (OSError, RuntimeError) as exc:
        report = {
            "gate": "win_proof",
            "commit_sha": _git_head_sha(),
            "verdict": PreflightVerdict.NOT_VERIFIED.value,
            "reasons": [f"docker_validation_failed: {exc}"],
            "checkpoint": str(checkpoint),
            "evaluation_mode": "submit_valid_docker_gate",
            "docker_output_dir": str(docker_output_dir),
            "tournament_skipped": True,
            "tournament_skipped_reason": "docker_validation_failed",
        }
        write_report(args.out, report)
        print(json.dumps(report, indent=2))
        return 1

    if not docker_gate_passed(docker_manifest):
        report = {
            "gate": "win_proof",
            "commit_sha": _git_head_sha(),
            "verdict": PreflightVerdict.NOT_VERIFIED.value,
            "reasons": ["docker_validation_failed"],
            "checkpoint": str(checkpoint),
            "evaluation_mode": "submit_valid_docker_gate",
            "docker_manifest": docker_manifest,
            "docker_output_dir": str(docker_output_dir),
            "tournament_skipped": True,
            "tournament_skipped_reason": "docker_validation_failed",
        }
        write_report(args.out, report)
        print(json.dumps(report, indent=2))
        return 1

    verdict = run_unified_ladder(
        checkpoint,
        spec,
        output_dir,
        campaign=args.campaign,
        output_root=args.output_root,
    )

    preflight_verdict = PreflightVerdict.VERIFIED
    reasons: list[str] = []
    if not verdict.passed:
        if spec.enforcement:
            preflight_verdict = PreflightVerdict.NOT_VERIFIED
        else:
            preflight_verdict = PreflightVerdict.INCONCLUSIVE
        reasons.append(verdict.reason)

    report = {
        "gate": "win_proof",
        "commit_sha": _git_head_sha(),
        "verdict": preflight_verdict.value,
        "reasons": reasons,
        "docker_validation_ok": True,
        "docker_output_dir": str(docker_output_dir),
        "docker_manifest": docker_manifest,
        "unified_verdict": verdict.to_dict(),
        "unified_verdict_path": str(output_dir / "unified_verdict.json"),
        "checkpoint": str(checkpoint),
        "evaluation_mode": "unified_tournament",
        "enforcement": spec.enforcement,
        "submit_valid_order": ["docker_validation", "unified_tournament_ladder"],
    }
    write_report(args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if preflight_verdict == PreflightVerdict.VERIFIED else 1
