"""Emit machine-readable session context for coding agents (no JAX imports)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_preflight_gate_ids(repo_root: Path) -> dict[str, object]:
    gates_dir = repo_root / "conf" / "benchmark" / "gates"
    if not gates_dir.is_dir():
        return {"present": False, "path": "conf/benchmark/gates", "gate_ids": []}
    gate_ids = sorted(path.stem for path in gates_dir.glob("*.yaml"))
    return {
        "present": True,
        "path": "conf/benchmark/gates",
        "gate_ids": gate_ids,
        "list_command": "uv run ow benchmark gate --list",
    }


def _read_resolved_config_snapshot(
    repo_root: Path,
    *,
    include_snapshot: bool,
) -> dict[str, object]:
    smoke_cmd = [
        "uv",
        "run",
        "ow",
        "train",
        "print_resolved_config=true",
        "training=smoke",
        "training.total_updates=10",
        "curriculum=off",
    ]
    pointer: dict[str, object] = {
        "present": True,
        "print_command": "uv run ow train print_resolved_config=true",
        "smoke_command": " ".join(smoke_cmd),
        "make_smoke": "make agent-context RESOLVED=smoke",
    }
    if not include_snapshot:
        return pointer
    try:
        result = subprocess.run(
            smoke_cmd,
            check=False,
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {**pointer, "snapshot_present": False, "error": str(exc)}
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        return {
            **pointer,
            "snapshot_present": False,
            "error": stderr[:500] or f"exit {result.returncode}",
        }
    body = result.stdout
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    payload: dict[str, object] = {
        **pointer,
        "snapshot_present": True,
        "sha256": digest,
        "sha256_prefix": digest[:16],
        "snapshot": body[:8000],
    }
    if len(body) > 8000:
        payload["snapshot_truncated"] = True
    return payload


def _read_wandb_sweep_summary(repo_root: Path) -> dict[str, object]:
    cmd = [
        "uv",
        "run",
        "ow",
        "sweep",
        "list",
        "--backend",
        "wandb",
        "--limit",
        "5",
    ]
    pointer: dict[str, object] = {
        "present": False,
        "list_command": "uv run ow sweep list --backend wandb --limit 10",
        "status_command": "uv run ow sweep status --backend wandb --sweep-id <id>",
        "cancel_command": "uv run ow sweep cancel --backend wandb --sweep-id <id> --dry-run",
    }
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=45,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {**pointer, "error": str(exc)}
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        return {
            **pointer,
            "error": stderr[:500] or f"exit {result.returncode}",
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {**pointer, "error": f"invalid JSON from sweep list: {exc}"}
    sweeps = payload.get("sweeps")
    if not isinstance(sweeps, list):
        return {**pointer, "error": "sweep list missing sweeps array"}
    rows: list[dict[str, object]] = []
    for row in sweeps[:5]:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "id": row.get("id"),
                "name": row.get("name"),
                "state": row.get("state"),
            }
        )
    active = sum(
        1
        for row in rows
        if str(row.get("state", "")).lower() in {"running", "pending"}
    )
    return {
        **pointer,
        "present": True,
        "active_count": active,
        "recent": rows,
    }


def _gpu_contention_hint(repo_root: Path) -> dict[str, object]:
    patterns = (
        "ow train",
        "calibrate-seed-scheduler",
        "pytest",
    )
    active: list[str] = []
    for pattern in patterns:
        try:
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                check=False,
                capture_output=True,
                text=True,
                cwd=repo_root,
            )
        except OSError:
            continue
        if result.returncode == 0 and result.stdout.strip():
            active.append(pattern)
    return {
        "single_gpu_note": (
            "One GPU; check terminals and defer ow train / calibrate-seed-scheduler "
            "when contention is active."
        ),
        "contention": bool(active),
        "active_patterns": active,
    }


def _read_preflight_excerpt(repo_root: Path) -> dict[str, object]:
    path = repo_root / "docs" / "benchmarks" / "preflight-calibration.json"
    if not path.is_file():
        return {"path": str(path), "present": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    thresholds = payload.get("thresholds") or {}
    learning = thresholds.get("learning_signal") or {}
    tournament = thresholds.get("win_proof_tournament") or {}
    return {
        "path": "docs/benchmarks/preflight-calibration.json",
        "present": True,
        "learning_signal": {
            "window_updates": learning.get("window_updates"),
            "min_win_rate_delta": learning.get("min_win_rate_delta"),
            "max_approx_kl": learning.get("max_approx_kl"),
            "min_entropy": learning.get("min_entropy"),
        },
        "win_proof_tournament": {
            "noop_min_win_rate": tournament.get("noop_min_win_rate"),
            "random_min_win_rate": tournament.get("random_min_win_rate"),
        },
        "notes": thresholds.get("notes"),
    }


def _read_roadmap_excerpt(repo_root: Path, *, max_lines: int = 24) -> dict[str, object]:
    path = repo_root / "docs" / "ROADMAP.md"
    if not path.is_file():
        return {"path": str(path), "present": False}
    lines = path.read_text(encoding="utf-8").splitlines()
    section: str | None = None
    now: list[str] = []
    next_items: list[str] = []
    for line in lines:
        if line.startswith("## "):
            section = line[3:].strip().lower()
            continue
        if section == "now" and line.strip() and not line.startswith("|"):
            now.append(line.strip())
        if section == "next" and line.startswith("|") and "Item" not in line and "---" not in line:
            next_items.append(line.strip())
    return {
        "path": "docs/ROADMAP.md",
        "present": True,
        "now": now[:5],
        "next": next_items[:5],
    }


def _read_recent_runs(repo_root: Path, *, limit: int = 5) -> list[dict[str, object]]:
    index_path = repo_root / "outputs" / "indexes" / "runs.jsonl"
    if not index_path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:]


def _read_git_branch(repo_root: Path) -> dict[str, object]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
    except (OSError, subprocess.SubprocessError):
        return {"present": False}
    branch = result.stdout.strip()
    return {"present": True, "branch": branch or None}


def _read_latest_run_eval_summary(
    repo_root: Path,
    recent_runs: list[dict[str, object]],
) -> dict[str, object] | None:
    if not recent_runs:
        return None
    latest = recent_runs[-1]
    run_dir_value = latest.get("run_dir")
    if not isinstance(run_dir_value, str) or not run_dir_value:
        return None
    run_dir = Path(run_dir_value)
    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir
    if not run_dir.is_dir():
        return {
            "run_dir": str(run_dir),
            "present": False,
            "reason": "run_dir_missing",
        }
    from src.cli.run_status import queue_is_active, summarize_run_status

    summary = summarize_run_status(run_dir)
    jobs = summary.get("jobs")
    job_count = len(jobs) if isinstance(jobs, list) else 0
    active_jobs = 0
    if isinstance(jobs, list):
        active_jobs = sum(
            1
            for job in jobs
            if isinstance(job, dict)
            and str(job.get("status")) in {"queued", "running"}
        )
    return {
        "present": True,
        "run_dir": summary.get("run_dir"),
        "run_id": summary.get("run_id"),
        "campaign": summary.get("campaign"),
        "job_count": job_count,
        "active_jobs": active_jobs,
        "queue_active": queue_is_active(summary),
        "promoted_manifest": summary.get("promoted_manifest"),
        "last_log_marker": summary.get("last_log_marker"),
    }


def build_context(
    *,
    limit_runs: int = 5,
    resolved_snapshot: bool = False,
) -> dict[str, object]:
    repo_root = _repo_root()
    recent_runs = _read_recent_runs(repo_root, limit=limit_runs)
    preflight = _read_preflight_excerpt(repo_root)
    preflight["gates"] = _read_preflight_gate_ids(repo_root)
    return {
        "repo_root": str(repo_root),
        "git": _read_git_branch(repo_root),
        "preflight": preflight,
        "resolved_config": _read_resolved_config_snapshot(
            repo_root,
            include_snapshot=resolved_snapshot,
        ),
        "gpu_contention": _gpu_contention_hint(repo_root),
        "wandb_sweeps": _read_wandb_sweep_summary(repo_root),
        "roadmap": _read_roadmap_excerpt(repo_root),
        "recent_runs_index": recent_runs,
        "latest_run_eval": _read_latest_run_eval_summary(repo_root, recent_runs),
        "docs": {
            "agent_capabilities": "docs/AGENT_CAPABILITIES.md",
            "agents": "AGENTS.md",
            "onboarding": "docs/ONBOARDING.md",
            "conf_readme": "conf/README.md",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit agent session context as JSON.")
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="Output format (default: json).",
    )
    parser.add_argument(
        "--limit-runs",
        type=int,
        default=5,
        help="Max recent runs.jsonl entries to include.",
    )
    parser.add_argument(
        "--resolved",
        choices=("smoke",),
        default=None,
        help="Include truncated Hydra resolved-config snapshot (smoke profile).",
    )
    args = parser.parse_args(argv)
    payload = build_context(
        limit_runs=max(int(args.limit_runs), 0),
        resolved_snapshot=args.resolved == "smoke",
    )
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
