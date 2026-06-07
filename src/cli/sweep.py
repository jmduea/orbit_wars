"""Unified sweep orchestration for W&B and Kaggle backends."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KAGGLE_SWEEP = REPO_ROOT / "conf" / "wandb_sweep" / "2p_only_throughput.yaml"


def print_sweep_help() -> None:
    print(
        "ow sweep — create and inspect W&B / Kaggle sweeps\n\n"
        "Subcommands:\n"
        "  create    Register a sweep (--backend wandb|kaggle)\n"
        "  status    Inspect sweep state (W&B API)\n"
        "  list      List recent sweeps for a project (W&B API)\n"
        "  cancel    Cancel active runs in a W&B sweep\n\n"
        "Examples:\n"
        "  uv run ow sweep create --backend wandb --yaml outputs/_meta/sweeps/2p_only_throughput.yaml\n"
        "  uv run ow make wandb_sweep=shield_cheap_history\n"
        "  uv run ow sweep create --backend wandb --make wandb_sweep=shield_cheap_history\n"
        "  uv run ow sweep create --backend kaggle --sweep-yaml conf/wandb_sweep/2p_only_throughput.yaml --dry-run\n"
        "  uv run ow sweep status --backend wandb --sweep-id <id> --project orbit_wars\n\n"
        "Deprecated: bare `wandb sweep` and `ow train kaggle launch --create-sweep` "
        "(still work; prefer `ow sweep create`).\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified W&B and Kaggle sweep orchestration.")
    subparsers = parser.add_subparsers(dest="command")

    create = subparsers.add_parser("create", help="Register a sweep on W&B or Kaggle worker path.")
    create.add_argument(
        "--backend",
        choices=("wandb", "kaggle"),
        required=True,
        help="Sweep backend to target.",
    )
    create.add_argument(
        "--yaml",
        type=Path,
        default=None,
        help="W&B sweep YAML path (wandb backend).",
    )
    create.add_argument(
        "--make",
        default=None,
        metavar="OVERRIDE",
        help="Hydra override passed to `ow make` to generate sweep YAML first.",
    )
    create.add_argument(
        "--sweep-yaml",
        type=Path,
        default=DEFAULT_KAGGLE_SWEEP,
        help="Packaged sweep YAML for kaggle backend (default conf/wandb_sweep/2p_only_throughput.yaml).",
    )
    create.add_argument("--project", default="orbit_wars")
    create.add_argument("--entity", default=None)
    create.add_argument("--dry-run", action="store_true")

    status = subparsers.add_parser("status", help="Inspect a W&B sweep by id.")
    status.add_argument("--backend", choices=("wandb",), default="wandb")
    status.add_argument("--sweep-id", required=True)
    status.add_argument("--project", default="orbit_wars")
    status.add_argument("--entity", default=None)

    list_cmd = subparsers.add_parser("list", help="List recent W&B sweeps.")
    list_cmd.add_argument("--backend", choices=("wandb",), default="wandb")
    list_cmd.add_argument("--project", default="orbit_wars")
    list_cmd.add_argument("--entity", default=None)
    list_cmd.add_argument("--limit", type=int, default=10)

    cancel = subparsers.add_parser(
        "cancel",
        help="Cancel active/pending runs in a W&B sweep (operator teardown).",
    )
    cancel.add_argument("--backend", choices=("wandb",), default="wandb")
    cancel.add_argument("--sweep-id", required=True)
    cancel.add_argument("--project", default="orbit_wars")
    cancel.add_argument("--entity", default=None)
    cancel.add_argument("--dry-run", action="store_true")

    return parser


def _resolve_wandb_yaml(args: argparse.Namespace) -> Path:
    if args.yaml is not None:
        return args.yaml
    if args.make:
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "make_wandb_sweep.py"), args.make],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(proc.stderr or proc.stdout, file=sys.stderr)
            raise SystemExit(proc.returncode)
        for line in proc.stdout.splitlines():
            if line.startswith("Wrote "):
                return Path(line.removeprefix("Wrote ").strip())
        raise SystemExit("make_wandb_sweep did not report output path")
    raise SystemExit("wandb backend requires --yaml or --make OVERRIDE")


def run_create_cli(args: argparse.Namespace) -> int:
    if args.backend == "wandb":
        yaml_path = _resolve_wandb_yaml(args)
        if not yaml_path.is_file():
            print(f"sweep YAML not found: {yaml_path}", file=sys.stderr)
            return 1
        if args.dry_run:
            payload = {"backend": "wandb", "yaml": str(yaml_path), "dry_run": True}
            print(json.dumps(payload, indent=2))
            return 0
        cmd = ["uv", "run", "wandb", "sweep", str(yaml_path)]
        print(
            "Note: prefer `ow sweep create --backend wandb`; bare `wandb sweep` is deprecated.",
            file=sys.stderr,
        )
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        return int(proc.returncode)

    from src.orchestration.kaggle_runner import load_sweep_config
    from src.orchestration.wandb_sweeps import (
        add_population_metadata,
        create_sweep,
        resolve_wandb_group_from_sweep,
    )

    loaded = load_sweep_config(args.sweep_yaml)
    group = resolve_wandb_group_from_sweep(loaded, sweep_yaml_path=args.sweep_yaml)
    sweep = add_population_metadata(loaded, group=group, tags=("kaggle", "population"))
    if args.dry_run:
        print(
            json.dumps(
                {"backend": "kaggle", "would_create_sweep": sweep, "dry_run": True},
                indent=2,
            )
        )
        return 0
    print(
        "Note: prefer `ow sweep create --backend kaggle`; "
        "`ow train kaggle launch --create-sweep` is deprecated.",
        file=sys.stderr,
    )
    sweep_id = create_sweep(sweep, project=args.project, entity=args.entity)
    print(json.dumps({"backend": "kaggle", "sweep_id": sweep_id}, indent=2))
    return 0


def run_status_cli(args: argparse.Namespace) -> int:
    import wandb  # type: ignore

    api = wandb.Api()
    entity = args.entity
    path = f"{entity}/{args.project}" if entity else args.project
    sweep = api.sweep(f"{path}/{args.sweep_id}")
    payload = {
        "backend": args.backend,
        "sweep_id": args.sweep_id,
        "project": args.project,
        "entity": entity,
        "state": getattr(sweep, "state", None),
        "run_count": len(getattr(sweep, "runs", []) or []),
    }
    print(json.dumps(payload, indent=2))
    return 0


def run_list_cli(args: argparse.Namespace) -> int:
    import wandb  # type: ignore

    api = wandb.Api()
    entity = args.entity
    path = f"{entity}/{args.project}" if entity else args.project
    sweeps = api.project(path).sweeps()
    rows = []
    for sweep in list(sweeps)[: max(int(args.limit), 1)]:
        rows.append(
            {
                "id": getattr(sweep, "id", None),
                "name": getattr(sweep, "name", None),
                "state": getattr(sweep, "state", None),
            }
        )
    print(json.dumps({"backend": args.backend, "sweeps": rows}, indent=2))
    return 0


def run_cancel_cli(args: argparse.Namespace) -> int:
    if args.backend != "wandb":
        print(f"sweep cancel unsupported for backend {args.backend!r}", file=sys.stderr)
        return 2

    entity = args.entity
    path = f"{entity}/{args.project}" if entity else args.project
    sweep_path = f"{path}/{args.sweep_id}"
    if args.dry_run:
        print(
            json.dumps(
                {
                    "backend": args.backend,
                    "sweep_id": args.sweep_id,
                    "project": args.project,
                    "entity": entity,
                    "sweep_path": sweep_path,
                    "dry_run": True,
                    "action": "would_cancel_active_runs",
                },
                indent=2,
            )
        )
        return 0

    import wandb  # type: ignore

    api = wandb.Api()
    sweep = api.sweep(sweep_path)
    cancelled: list[str] = []
    skipped: list[str] = []
    for run in getattr(sweep, "runs", []) or []:
        state = str(getattr(run, "state", "") or "")
        run_id = str(getattr(run, "id", "") or "")
        if state.lower() in {"running", "pending", "queued"}:
            run.cancel()
            cancelled.append(run_id)
        elif run_id:
            skipped.append(run_id)
    payload = {
        "backend": args.backend,
        "sweep_id": args.sweep_id,
        "project": args.project,
        "entity": entity,
        "sweep_state": getattr(sweep, "state", None),
        "cancelled_run_ids": cancelled,
        "skipped_run_ids": skipped[:20],
        "cancelled_count": len(cancelled),
    }
    print(json.dumps(payload, indent=2))
    return 0 if cancelled or not getattr(sweep, "runs", None) else 1


def main(argv: list[str] | None = None) -> int:
    if not argv:
        print_sweep_help()
        return 0
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        print_sweep_help()
        return 0
    match args.command:
        case "create":
            return run_create_cli(args)
        case "status":
            return run_status_cli(args)
        case "list":
            return run_list_cli(args)
        case "cancel":
            return run_cancel_cli(args)
        case _:
            parser.error(f"unknown sweep command: {args.command!r}")
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
