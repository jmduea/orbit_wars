#!/usr/bin/env python3
"""Launch and inspect W&B-first Kaggle population workers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.orchestration.kaggle_cli import KaggleCli, KaggleKernelRef
from src.orchestration.kernel_package import render_kernel_package
from src.orchestration.population import AcceleratorPreference
from src.orchestration.wandb_sweeps import (
    add_population_metadata,
    create_sweep,
    load_sweep_config,
    shortlist_from_api,
)

DEFAULT_SWEEP = REPO_ROOT / "conf/sweeps/wandb/kaggle_population_mvp.yaml"
DEFAULT_WORK_DIR = REPO_ROOT / "outputs/kaggle_population/kernel"
WORKER_SOURCE = REPO_ROOT / "scripts/kaggle_worker_entry.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    _add_package_args(prepare)
    prepare.add_argument("--sweep-id", default=None)

    launch = subparsers.add_parser("launch")
    _add_package_args(launch)
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--sweep-id", default=None)
    launch.add_argument("--create-sweep", action="store_true")
    launch.add_argument("--project", default="orbit_wars")
    launch.add_argument("--entity", default=None)
    launch.add_argument("--timeout-seconds", type=int, default=43200)
    launch.add_argument("--accelerator", action="append", dest="accelerators")

    status = subparsers.add_parser("status")
    status.add_argument("kernel_ref")

    sync = subparsers.add_parser("sync-output")
    sync.add_argument("kernel_ref")
    sync.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/kaggle_population/synced")
    sync.add_argument("--force", action="store_true")

    shortlist = subparsers.add_parser("shortlist")
    shortlist.add_argument("--project", default="orbit_wars")
    shortlist.add_argument("--entity", default=None)
    shortlist.add_argument("--sweep-id", required=True)
    shortlist.add_argument("--limit", type=int, default=10)
    shortlist.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/kaggle_population/shortlist.json")

    return parser.parse_args()


def _add_package_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--kernel-id", default=_default_kernel_id())
    parser.add_argument("--title", default="Orbit Wars W&B Population Worker")
    parser.add_argument("--sweep-yaml", type=Path, default=DEFAULT_SWEEP)


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        package = _prepare(args, sweep_id=args.sweep_id)
        print(json.dumps({"package_dir": str(package.package_dir)}, indent=2))
        return
    if args.command == "launch":
        sweep_id = args.sweep_id
        if args.create_sweep:
            sweep = add_population_metadata(
                load_sweep_config(args.sweep_yaml),
                group="kaggle_population_mvp",
                tags=("kaggle", "population"),
            )
            if args.dry_run:
                sweep_id = "dry-run-sweep"
                print(json.dumps({"would_create_sweep": sweep}, indent=2, sort_keys=True))
            else:
                sweep_id = create_sweep(sweep, project=args.project, entity=args.entity)
        package = _prepare(args, sweep_id=sweep_id)
        accelerators = tuple(args.accelerators or AcceleratorPreference().accelerator_ids)
        _launch(args, package.package_dir, accelerators)
        return
    if args.command == "status":
        status = KaggleCli().status(KaggleKernelRef.parse(args.kernel_ref))
        print(json.dumps({"ref": str(status.ref), "status": status.normalized, "raw": status.raw}, indent=2))
        return
    if args.command == "sync-output":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = KaggleCli().output(
            KaggleKernelRef.parse(args.kernel_ref),
            args.output_dir,
            force=args.force,
        )
        print(result.stdout or result.stderr)
        raise SystemExit(result.returncode)
    if args.command == "shortlist":
        rows = shortlist_from_api(
            project=args.project,
            entity=args.entity,
            sweep_id=args.sweep_id,
            limit=args.limit,
        )
        payload = [
            {
                "run_id": row.run_id,
                "name": row.name,
                "state": row.state,
                "checkpoint_artifact": row.checkpoint_artifact,
                "metrics": dict(row.metrics),
                "score": row.score,
            }
            for row in rows
        ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return


def _prepare(args: argparse.Namespace, *, sweep_id: str | None):
    env = {
        "WANDB_SWEEP_ID": sweep_id or "",
        "WANDB_SWEEP_YAML": str(args.sweep_yaml),
    }
    return render_kernel_package(
        package_dir=args.work_dir,
        kernel_id=args.kernel_id,
        title=args.title,
        worker_source=WORKER_SOURCE,
        env=env,
        repo_root=REPO_ROOT,
    )


def _launch(args: argparse.Namespace, package_dir: Path, accelerators: tuple[str, ...]) -> None:
    cli = KaggleCli()
    for accelerator in accelerators:
        with _worker_env_override(package_dir, {"KAGGLE_ACCELERATOR_ID": accelerator}):
            if args.dry_run:
                print(
                    " ".join(
                        [
                            "kaggle",
                            "kernels",
                            "push",
                            "-p",
                            str(package_dir),
                            "--accelerator",
                            accelerator,
                            "--timeout",
                            str(args.timeout_seconds),
                        ]
                    )
                )
                return
            result = cli.push(
                package_dir,
                accelerator=accelerator,
                timeout_seconds=args.timeout_seconds,
            )
            if result.returncode == 0:
                print(result.stdout)
                return
            print(result.stderr)
    raise SystemExit("No accelerator launch attempt succeeded.")


@contextmanager
def _worker_env_override(package_dir: Path, values: dict[str, str]) -> Iterator[None]:
    env_path = package_dir / "worker-env.json"
    payload = {}
    if env_path.exists():
        payload = json.loads(env_path.read_text(encoding="utf-8"))
    payload.update(values)
    env_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    try:
        yield
    finally:
        for key in values:
            payload.pop(key, None)
        env_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _default_kernel_id() -> str:
    owner = os.environ.get("KAGGLE_USERNAME", "replace-me")
    return f"{owner}/orbit-wars-wandb-population"


if __name__ == "__main__":
    main()
