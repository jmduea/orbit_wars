"""``ow eval`` CLI for tournament evaluation, artifact worker, and Kaggle submit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.artifacts.kaggle_submission import (
    DEFAULT_COMPETITION,
    package_checkpoint_submission,
    submit_competition_package,
)
from src.artifacts.tournament.eval import run_tournament
from src.artifacts.tournament.promotion import promote_from_tournament, top_passing_row
from src.artifacts.tournament.resolve import (
    ShortlistResolveResult,
    agent_from_checkpoint,
    resolve_promoted_agent,
    resolve_shortlist_agents,
    run_context_for_agent,
    validate_agents_feature_compatible,
)
from src.artifacts.worker_runner import resolve_run_worker_dirs, run_optional_job_worker
from src.config.schema import TournamentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluation, artifact jobs, and Kaggle submission (ow eval).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    tournament = subparsers.add_parser(
        "tournament",
        help="Evaluate checkpoints head-to-head in Kaggle env.",
    )
    _add_tournament_arguments(tournament)

    worker = subparsers.add_parser(
        "worker",
        help="Process queued optional artifact jobs for a run.",
    )
    worker.add_argument(
        "--run",
        type=Path,
        default=None,
        help="Campaign run directory (uses run/queue/optional_jobs and run/evaluations).",
    )
    worker.add_argument(
        "--queue-dir",
        type=Path,
        default=None,
        help="Optional job queue directory (overrides --run queue path).",
    )
    worker.add_argument(
        "--result-root",
        type=Path,
        default=None,
        help="Evaluations output root (defaults to run/evaluations when --run is set).",
    )
    worker.add_argument(
        "--watch",
        action="store_true",
        help="Poll the queue until idle-exit-seconds (default: process once and exit).",
    )
    worker.add_argument(
        "--retry-failed",
        action="store_true",
        help="Include failed jobs (explicit retry workflow).",
    )
    worker.add_argument(
        "--recover-running",
        action="store_true",
        help="Also pick up jobs left in running status by a dead worker.",
    )
    worker.add_argument("--poll-seconds", type=float, default=5.0)
    worker.add_argument("--idle-exit-seconds", type=float, default=None)

    submit = subparsers.add_parser(
        "submit",
        help="Package a checkpoint and submit submission.tar.gz to Kaggle.",
    )
    submit.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to package (required unless --package is set).",
    )
    submit.add_argument(
        "--package",
        type=Path,
        default=None,
        help="Existing submission.tar.gz to upload (skip packaging).",
    )
    submit.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/kaggle_submission"),
        help="Directory for packaging output when building from --checkpoint.",
    )
    submit.add_argument(
        "--message",
        "-m",
        default=None,
        help="Kaggle submission message (default: derived from checkpoint name).",
    )
    submit.add_argument(
        "--competition",
        default=DEFAULT_COMPETITION,
        help=f"Kaggle competition slug (default: {DEFAULT_COMPETITION}).",
    )
    submit.add_argument(
        "--validate-docker",
        action="store_true",
        help="Run local Kaggle Docker validation before submitting.",
    )
    submit.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the kaggle competitions submit command without uploading.",
    )
    submit.add_argument(
        "--quiet",
        action="store_true",
        help="Pass -q to kaggle competitions submit.",
    )

    package = subparsers.add_parser(
        "package",
        help="Build submission.tar.gz from a checkpoint (optional Docker validation).",
    )
    package.add_argument("--checkpoint", required=True, type=Path)
    package.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/kaggle_submission"),
        help="Directory for submission.tar.gz and staging artifacts.",
    )
    package.add_argument(
        "--validate-docker",
        action="store_true",
        help="Run Kaggle Docker validation after packaging (requires Docker).",
    )

    return parser


def _add_tournament_arguments(tournament: argparse.ArgumentParser) -> None:
    tournament.add_argument("--campaign", default="scratch", help="Campaign slug for outputs.")
    tournament.add_argument(
        "--output-root",
        default="outputs",
        help="Output root containing campaigns/.",
    )
    tournament.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit tournament output directory.",
    )
    tournament.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        dest="checkpoints",
        type=Path,
        help="Checkpoint path (repeatable).",
    )
    tournament.add_argument(
        "--shortlist",
        type=Path,
        default=None,
        help="W&B shortlist JSON with optional checkpoint_path fields.",
    )
    tournament.add_argument("--limit", type=int, default=5, help="Max shortlist candidates.")
    tournament.add_argument(
        "--vs-promoted",
        action="store_true",
        help="Include campaign promoted incumbent in head-to-head matches.",
    )
    tournament.add_argument(
        "--promote",
        action="store_true",
        help="Promote top gate-passing candidate to campaign current_best.",
    )
    tournament.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved agents and exit without running matches.",
    )
    tournament.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated env seeds.")
    tournament.add_argument("--games-per-pair", type=int, default=1)
    tournament.add_argument("--max-steps", type=int, default=500)
    tournament.add_argument(
        "--formats",
        default="2p_vs_baseline,2p_head_to_head",
        help="Comma-separated tournament formats.",
    )
    tournament.add_argument(
        "--write-replays",
        action="store_true",
        help="Write HTML replay artifacts for each match.",
    )
    tournament.add_argument(
        "--per-step-seconds",
        type=float,
        default=1.0,
        help="Per-agent action latency budget (Kaggle submission parity).",
    )
    tournament.add_argument(
        "--overage-budget-seconds",
        type=float,
        default=60.0,
        help="Cumulative overage allowed above per-step budget before aborting.",
    )


def _parse_csv_ints(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def _parse_csv_strings(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _default_output_dir(campaign: str, output_root: str) -> Path:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        Path(output_root)
        / "campaigns"
        / campaign
        / "evaluations"
        / f"tournament_{stamp}"
    )


def _collect_candidates(args: argparse.Namespace) -> tuple[list, ShortlistResolveResult | None]:
    candidates: list = []
    shortlist_result: ShortlistResolveResult | None = None
    for checkpoint in args.checkpoints:
        candidates.append(agent_from_checkpoint(checkpoint))
    if args.shortlist is not None:
        cache_dir = Path(args.output_root) / "cache" / "wandb-artifacts"
        shortlist_result = resolve_shortlist_agents(
            args.shortlist,
            limit=int(args.limit),
            wandb_cache_dir=cache_dir,
        )
        if shortlist_result.errors:
            for message in shortlist_result.errors:
                print(f"shortlist: {message}", file=sys.stderr)
        candidates.extend(shortlist_result.agents)
    return candidates, shortlist_result


def run_tournament_cli(args: argparse.Namespace) -> int:
    candidates, shortlist_result = _collect_candidates(args)
    if not candidates:
        if shortlist_result is not None and shortlist_result.errors:
            raise SystemExit(
                "No shortlist candidates resolved. Provide --checkpoint paths, "
                "checkpoint_path in shortlist JSON, or W&B artifact access."
            )
        raise SystemExit("Provide --checkpoint and/or --shortlist with resolvable paths.")

    validate_agents_feature_compatible(candidates)

    cfg = candidates[0].cfg
    cfg.output.campaign = str(args.campaign)
    cfg.output.root = str(args.output_root)
    cfg.artifacts.tournament = TournamentConfig(
        enabled=True,
        seeds=_parse_csv_ints(args.seeds),
        games_per_pair=int(args.games_per_pair),
        max_steps=int(args.max_steps),
        baselines=["sniper"],
        formats=_parse_csv_strings(args.formats),
        write_replays=bool(args.write_replays),
        per_step_seconds=float(args.per_step_seconds),
        overage_budget_seconds=float(args.overage_budget_seconds),
    )

    incumbent = None
    if args.vs_promoted:
        incumbent = resolve_promoted_agent(str(args.campaign), str(args.output_root))

    if args.dry_run:
        payload = {
            "candidates": [candidate.agent_id for candidate in candidates],
            "incumbent": incumbent.agent_id if incumbent is not None else None,
            "formats": cfg.artifacts.tournament.formats,
            "seeds": cfg.artifacts.tournament.seeds,
            "shortlist_skipped": list(shortlist_result.skipped)
            if shortlist_result is not None
            else [],
        }
        print(json.dumps(payload, indent=2))
        return 0

    output_dir = args.output_dir or _default_output_dir(str(args.campaign), str(args.output_root))
    result = run_tournament(
        tuple(candidates),
        cfg=cfg.artifacts.tournament,
        output_dir=output_dir,
        incumbent=incumbent,
        promotion_gates=cfg.artifacts.promotion.tournament,
    )
    print(
        json.dumps(
            json.loads((output_dir / "leaderboard.json").read_text())["rows"], indent=2
        )
    )

    if args.promote:
        passing = top_passing_row(result)
        if passing is None:
            print("No candidate passed tournament gates; promotion skipped.", file=sys.stderr)
            return 1
        promoted_agent = next(
            agent for agent in candidates if agent.agent_id == passing.agent_id
        )
        cfg.artifacts.promotion.enabled = True
        cfg.artifacts.promotion.strategy = "tournament"
        context = run_context_for_agent(
            promoted_agent,
            campaign=str(args.campaign),
            output_root=str(args.output_root),
        )
        attempt = promote_from_tournament(
            cfg,
            context,
            row=passing,
            tournament=result,
        )
        if not attempt.promoted:
            print(f"Promotion failed: {attempt.reason}", file=sys.stderr)
            return 1
        print(f"Promoted {passing.agent_id} -> {attempt.promoted_manifest_path}")
    return 0


def _resolve_worker_dirs(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.queue_dir is not None:
        queue_dir = args.queue_dir.resolve()
        result_root = (
            args.result_root.resolve() if args.result_root is not None else None
        )
        return queue_dir, result_root
    if args.run is None:
        raise SystemExit("Provide --run or --queue-dir.")
    queue_dir, evaluations_dir = resolve_run_worker_dirs(args.run)
    if args.result_root is not None:
        evaluations_dir = args.result_root.resolve()
    return queue_dir, evaluations_dir


def run_worker_cli(args: argparse.Namespace) -> int:
    from scripts import run_artifact_worker

    queue_dir, result_root = _resolve_worker_dirs(args)
    if not queue_dir.is_dir():
        raise SystemExit(f"Queue directory does not exist: {queue_dir}")
    return run_optional_job_worker(
        queue_dir,
        run_artifact_worker._process_job,
        run_artifact_worker._write_status,
        result_root=result_root,
        once=not bool(args.watch),
        poll_seconds=float(args.poll_seconds),
        idle_exit_seconds=args.idle_exit_seconds,
        recover_running=bool(args.recover_running),
        retry_failed=bool(args.retry_failed),
    )


def run_package_cli(args: argparse.Namespace) -> int:
    package_path = package_checkpoint_submission(
        args.checkpoint.resolve(),
        args.output_dir.resolve(),
        validate_docker=bool(args.validate_docker),
    )
    print(f"package_path={package_path}")
    if not args.validate_docker:
        print(
            "docker_validation=skipped (packaging only; does not prove competition compatibility)",
            file=sys.stderr,
        )
    return 0


def run_submit_cli(args: argparse.Namespace) -> int:
    if args.package is not None:
        package_path = args.package.resolve()
        default_message = package_path.name
    elif args.checkpoint is not None:
        checkpoint_path = args.checkpoint.resolve()
        default_message = checkpoint_path.name
        package_path = package_checkpoint_submission(
            checkpoint_path,
            args.output_dir.resolve(),
            validate_docker=bool(args.validate_docker),
        )
        print(f"package_path={package_path}")
    else:
        raise SystemExit(
            "Provide --checkpoint to package or --package to upload an existing tarball."
        )

    message = args.message or f"ow eval submit {default_message}"
    submit_competition_package(
        package_path,
        message,
        competition=str(args.competition),
        quiet=bool(args.quiet),
        dry_run=bool(args.dry_run),
    )
    if not args.dry_run:
        print(f"submitted={package_path} competition={args.competition}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "tournament":
        return run_tournament_cli(args)
    if args.command == "worker":
        return run_worker_cli(args)
    if args.command == "package":
        return run_package_cli(args)
    if args.command == "submit":
        return run_submit_cli(args)
    raise SystemExit(f"Unknown eval command: {args.command!r}")
