#!/usr/bin/env python3
"""Paired terminal-reward ablation: binary_win vs normalized_ship_differential."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = REPO_ROOT / "scripts/issues_jax_30update_benchmark.py"
BENCH_DIR = REPO_ROOT / "docs/benchmarks"
SUMMARY_PATH = BENCH_DIR / "terminal-reward-ablation.md"

SHARED_EXTRA = [
    "reward=terminal_only",
]


def _run_arm(
    *,
    label: str,
    out_path: Path,
    updates: int,
    reward_override: str | None,
    tier: str,
    snapshot_updates: list[int],
) -> None:
    cmd = [
        "uv",
        "run",
        "python",
        str(BENCHMARK),
        "--label",
        label,
        "--preset",
        "validation",
        "--tier",
        tier,
        "--updates",
        str(updates),
        "--snapshot-updates",
        *[str(u) for u in snapshot_updates],
        "--out",
        str(out_path),
    ]
    extra = list(SHARED_EXTRA)
    if reward_override is not None:
        extra.append(reward_override)
    if extra:
        cmd.extend(["--overrides", *extra])
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}"


def _write_summary(rows: list[dict], *, updates: int) -> None:
    lines = [
        "# Terminal reward ablation",
        "",
        f"Paired **{updates}u** runs on the workstation validation profile "
        "(`reward=terminal_only` baseline vs `reward=ship_differential` candidate).",
        "",
        "| Arm | Reward profile | overall_win_rate | average_reward | "
        "episode_reward_mean | survival_time | policy_loss | JSON |",
        "|-----|----------------|------------------|----------------|"
        "---------------------|---------------|-------------|------|",
    ]
    for row in rows:
        payload = row["payload"]
        lines.append(
            "| {arm} | {profile} | {win} | {pl} | {vl} | {kl} | {sps} | `{json}` |".format(
                arm=row["arm"],
                profile=row["reward_profile"],
                win=_fmt(payload.get("overall_win_rate")),
                pl=_fmt(payload.get("policy_loss")),
                vl=_fmt(payload.get("value_loss")),
                kl=_fmt(payload.get("approx_kl")),
                sps=_fmt(payload.get("env_steps_per_sec")),
                json=row["json_path"],
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `overall_win_rate` uses binary `terminal_is_first` telemetry in both arms.",
            "- Candidate terminal signal is graded in [-1, 1] via best-opponent normalization.",
            "",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument(
        "--tier",
        default="workstation",
        help="Benchmark tier label stored in JSON artifacts.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only rebuild markdown from existing JSON artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = [50, 100] if args.updates <= 100 else [250, 500]
    arms = [
        (
            "binary_win",
            f"terminal-reward-binary-{args.updates}u",
            BENCH_DIR / f"terminal-reward-binary-{args.updates}u.json",
            None,
        ),
        (
            "ship_differential",
            f"terminal-reward-ship-diff-{args.updates}u",
            BENCH_DIR / f"terminal-reward-ship-diff-{args.updates}u.json",
            "reward=ship_differential",
        ),
    ]
    if not args.skip_run:
        for _arm, label, out_path, reward_override in arms:
            _run_arm(
                label=label,
                out_path=out_path,
                updates=args.updates,
                reward_override=reward_override,
                tier=args.tier,
                snapshot_updates=snapshot,
            )
    rows = []
    for arm, _label, out_path, reward_override in arms:
        if not out_path.is_file():
            print(f"Missing artifact: {out_path}", file=sys.stderr)
            sys.exit(1)
        rows.append(
            {
                "arm": arm,
                "reward_profile": reward_override or "reward=terminal_only",
                "json_path": str(out_path.relative_to(REPO_ROOT)),
                "payload": _load_json(out_path),
            }
        )
    _write_summary(rows, updates=args.updates)


if __name__ == "__main__":
    main()
