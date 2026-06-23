"""``ow benchmark map-pool`` — offline bake profile, batch bake, validate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from src.benchmark.git_utils import git_head_sha

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_VERSION = "map_pool_bake_v1"
DEFAULT_MAX_EXTRAPOLATED_SECS = 1800.0


def _git_head_sha() -> str | None:
    return git_head_sha(REPO_ROOT)


def build_map_pool_parser(
    parent: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = parent.add_parser(
        "map-pool",
        help="Offline map-pool profile, batch bake, and validate.",
    )
    sub = parser.add_subparsers(dest="map_pool_command", required=True)

    profile = sub.add_parser(
        "profile",
        help="Time single-map offline bake (planets + comet waves) over sample seeds.",
    )
    profile.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of seeds to profile (default: 10).",
    )
    profile.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First integer seed for the profile sweep (default: 0).",
    )
    profile.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Write profile JSON (mean secs/map, success rate, extrapolated batch).",
    )

    bake = sub.add_parser(
        "bake",
        help="Bake a batch of map-pool entries to .npz + manifest sidecar.",
    )
    bake.add_argument("--count", type=int, required=True, help="Pool size (map count).")
    bake.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/jax_map_pool"),
        help="Output directory for <label>.npz and manifest.",
    )
    bake.add_argument(
        "--label", required=True, help="Artifact label (e.g. default_v1)."
    )
    bake.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Prior profile JSON from map-pool profile (R1 gate).",
    )
    bake.add_argument(
        "--accept-extrapolated-secs",
        type=float,
        default=None,
        metavar="SECS",
        help=(
            "Operator override: accept batch extrapolated wall time up to SECS "
            f"(default cap without override: {DEFAULT_MAX_EXTRAPOLATED_SECS:.0f}s)."
        ),
    )
    bake.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First integer seed for deterministic bake stream.",
    )

    validate = sub.add_parser(
        "validate",
        help="Validate a stacked map-pool .npz artifact.",
    )
    validate.add_argument("--pool", type=Path, required=True, help="Path to .npz pool.")

    return parser


def _profile_seeds(repeats: int, seed_start: int) -> list[int]:
    return [seed_start + i for i in range(max(int(repeats), 1))]


def run_profile_cli(args: argparse.Namespace) -> int:
    from src.jax.map_pool.bake import MapPoolBakeError, bake_one_entry

    seeds = _profile_seeds(args.repeats, args.seed_start)
    timings: list[float] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    for seed in seeds:
        t0 = time.perf_counter()
        try:
            bake_one_entry(seed)
        except MapPoolBakeError as exc:
            failures.append({"seed": seed, "error": str(exc)})
        else:
            timings.append(time.perf_counter() - t0)

    elapsed = time.perf_counter() - started
    success_count = len(timings)
    mean_secs = sum(timings) / success_count if timings else None
    payload: dict[str, Any] = {
        "command": "map-pool profile",
        "generator_version": GENERATOR_VERSION,
        "commit_sha": _git_head_sha(),
        "repeats": len(seeds),
        "seed_start": int(args.seed_start),
        "seeds": seeds,
        "success_count": success_count,
        "failure_count": len(failures),
        "success_rate": success_count / len(seeds) if seeds else 0.0,
        "seconds_total": elapsed,
        "mean_secs_per_map": mean_secs,
        "max_secs_per_map": max(timings) if timings else None,
        "min_secs_per_map": min(timings) if timings else None,
        "extrapolated_500_secs": (mean_secs * 500.0) if mean_secs is not None else None,
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0 if success_count == len(seeds) else 1


def _load_profile(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid profile JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"profile JSON must be an object: {path}")
    return data


def _extrapolated_secs(profile: dict[str, Any], count: int) -> float | None:
    mean = profile.get("mean_secs_per_map")
    if mean is None:
        return None
    return float(mean) * float(count)


def _bake_gate_ok(
    *,
    count: int,
    profile_path: Path | None,
    accept_extrapolated_secs: float | None,
) -> tuple[bool, str, dict[str, Any] | None]:
    if profile_path is None and accept_extrapolated_secs is None:
        return (
            False,
            "bake requires --profile or --accept-extrapolated-secs (R1/R2 gate)",
            None,
        )

    profile: dict[str, Any] | None = None
    extrapolated: float | None = None
    if profile_path is not None:
        try:
            profile = _load_profile(profile_path)
        except ValueError as exc:
            return False, str(exc), None
        extrapolated = _extrapolated_secs(profile, count)
        if extrapolated is None:
            return (
                False,
                f"profile {profile_path} missing mean_secs_per_map",
                profile,
            )

    if extrapolated is not None:
        cap = (
            float(accept_extrapolated_secs)
            if accept_extrapolated_secs is not None
            else DEFAULT_MAX_EXTRAPOLATED_SECS
        )
        if extrapolated > cap:
            return (
                False,
                (
                    f"extrapolated batch {extrapolated:.1f}s exceeds cap {cap:.1f}s "
                    "(pass --accept-extrapolated-secs to override)"
                ),
                profile,
            )

    if profile_path is None and accept_extrapolated_secs is not None:
        extrapolated = float(accept_extrapolated_secs)

    return True, "", profile


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_bake_cli(args: argparse.Namespace) -> int:
    from src.jax.map_pool.bake import MapPoolBakeError, bake_one_entry, save_pool_npz

    count = int(args.count)
    if count <= 0:
        print("bake --count must be positive", file=sys.stderr)
        return 1

    ok, message, profile = _bake_gate_ok(
        count=count,
        profile_path=args.profile,
        accept_extrapolated_secs=args.accept_extrapolated_secs,
    )
    if not ok:
        print(message, file=sys.stderr)
        return 1

    entries = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    seed_cursor = int(args.seed_start)
    attempts = 0
    max_attempts = count * 20
    while len(entries) < count:
        if attempts >= max_attempts:
            print(
                json.dumps(
                    {
                        "error": "exhausted bake attempts",
                        "entries": len(entries),
                        "requested": count,
                        "failures": failures[-20:],
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 1
        attempts += 1
        seed = seed_cursor
        seed_cursor += 1
        try:
            entries.append(bake_one_entry(seed))
        except MapPoolBakeError as exc:
            failures.append({"seed": seed, "error": str(exc)})
    elapsed = time.perf_counter() - started

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{args.label}.npz"
    manifest_path = out_dir / f"{args.label}.manifest.json"
    save_pool_npz(str(npz_path), entries)
    sha256 = _sha256_file(npz_path)

    manifest: dict[str, Any] = {
        "label": args.label,
        "pool_size": count,
        "npz_path": str(npz_path.relative_to(REPO_ROOT)),
        "sha256": sha256,
        "generator_version": GENERATOR_VERSION,
        "commit_sha": _git_head_sha(),
        "seed_start": int(args.seed_start),
        "seed_attempts": attempts,
        "rejected_seeds": len(failures),
        "bake_seconds": elapsed,
        "profile_path": str(args.profile) if args.profile is not None else None,
        "accept_extrapolated_secs": args.accept_extrapolated_secs,
    }
    if profile is not None:
        manifest["profile"] = {
            "mean_secs_per_map": profile.get("mean_secs_per_map"),
            "success_rate": profile.get("success_rate"),
            "extrapolated_batch_secs": _extrapolated_secs(profile, count),
        }

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    payload = {
        "ok": True,
        "npz": str(npz_path),
        "manifest": str(manifest_path),
        "pool_size": count,
        "sha256": sha256,
        "bake_seconds": elapsed,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def run_validate_cli(args: argparse.Namespace) -> int:
    from src.jax.map_pool.bake import (
        MapPoolBakeError,
        load_pool_npz,
        validate_stacked_pool,
    )

    pool_path = Path(args.pool)
    if not pool_path.is_file():
        print(f"pool not found: {pool_path}", file=sys.stderr)
        return 1
    try:
        arrays = load_pool_npz(str(pool_path))
        validate_stacked_pool(arrays)
    except (MapPoolBakeError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    pool_size = int(arrays["seed"].shape[0])
    payload = {
        "ok": True,
        "pool": str(pool_path),
        "pool_size": pool_size,
        "sha256": _sha256_file(pool_path),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def dispatch_map_pool(args: argparse.Namespace) -> int:
    match args.map_pool_command:
        case "profile":
            return run_profile_cli(args)
        case "bake":
            return run_bake_cli(args)
        case "validate":
            return run_validate_cli(args)
        case _:
            raise SystemExit(f"unknown map-pool command: {args.map_pool_command!r}")
