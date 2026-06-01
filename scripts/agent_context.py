"""Emit machine-readable session context for coding agents (no JAX imports)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def build_context(*, limit_runs: int = 5) -> dict[str, object]:
    repo_root = _repo_root()
    return {
        "repo_root": str(repo_root),
        "preflight": _read_preflight_excerpt(repo_root),
        "roadmap": _read_roadmap_excerpt(repo_root),
        "recent_runs_index": _read_recent_runs(repo_root, limit=limit_runs),
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
    args = parser.parse_args(argv)
    payload = build_context(limit_runs=max(int(args.limit_runs), 0))
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
