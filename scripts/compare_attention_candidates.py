from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import compose_hydra_train_config

DEFAULT_CONFIGS: dict[str, list[str]] = {
    "attention_candidates_8": [
        "model=attention",
        "task.candidate_count=8",
        "run_name=attention_candidates_8",
        "artifacts.save_dir=artifacts/attention_candidates_8",
    ],
    "attention_candidates_16": [
        "model=attention",
        "task.candidate_count=16",
        "run_name=attention_candidates_16",
        "artifacts.save_dir=artifacts/attention_candidates_16",
    ],
    "attention_candidates_24": [
        "model=attention",
        "task.candidate_count=24",
        "run_name=attention_candidates_24",
        "artifacts.save_dir=artifacts/attention_candidates_24",
    ],
}
DEFAULT_COLUMNS = (
    "config",
    "seed",
    "candidate_count",
    "real_target_slots",
    "status",
    "update",
    "total_env_steps",
    "episode_reward_mean",
    "candidate_valid_avg",
    "candidate_enemy_share",
    "candidate_neutral_share",
    "candidate_friendly_share",
    "approx_kl",
    "clip_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare attention candidate-count runs. Compose Hydra group overrides to train "
            "8/16/24-candidate runs with the same seed, then summarize their JSONL logs."
        )
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_CONFIGS),
        choices=sorted(DEFAULT_CONFIGS),
        help="Named group-based configs to compare.",
    )
    parser.add_argument("--log-dir", default="artifacts/rl_template/logs", help="Directory containing run_name.jsonl logs.")
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="Print training commands for the compared configs before the summary table.",
    )
    return parser.parse_args()


def load_config(config_name: str) -> dict[str, Any]:
    cfg = compose_hydra_train_config(DEFAULT_CONFIGS[config_name])
    return {
        "seed": cfg.seed,
        "env": {"candidate_count": cfg.env.candidate_count},
        "run_name": cfg.run_name,
    }


def latest_jsonl_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last_line = ""
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                last_line = line
    return json.loads(last_line) if last_line else None


def row_for_config(config_name: str, log_dir: Path) -> dict[str, Any]:
    cfg = load_config(config_name)
    env_cfg = cfg.get("env", {}) if isinstance(cfg.get("env", {}), dict) else {}
    candidate_count = int(env_cfg.get("candidate_count", 8))
    run_name = str(cfg.get("run_name", config_name))
    row: dict[str, Any] = {
        "config": config_name,
        "seed": cfg.get("seed", ""),
        "candidate_count": candidate_count,
        "real_target_slots": max(0, candidate_count - 1),
    }
    record = latest_jsonl_record(log_dir / f"{run_name}.jsonl")
    if record is None:
        row["status"] = "missing_log"
        return row
    row["status"] = "ok"
    row.update(record)
    return row


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_table(rows: list[dict[str, Any]]) -> None:
    widths = {column: len(column) for column in DEFAULT_COLUMNS}
    for row in rows:
        for column in DEFAULT_COLUMNS:
            widths[column] = max(widths[column], len(format_cell(row.get(column, ""))))
    print(" | ".join(column.ljust(widths[column]) for column in DEFAULT_COLUMNS))
    print("-+-".join("-" * widths[column] for column in DEFAULT_COLUMNS))
    for row in rows:
        print(" | ".join(format_cell(row.get(column, "")).ljust(widths[column]) for column in DEFAULT_COLUMNS))


def main() -> None:
    args = parse_args()
    configs = list(args.configs)
    if args.print_commands:
        for config_name in configs:
            overrides = " ".join(DEFAULT_CONFIGS[config_name])
            print(f"uv run python -m src.train {overrides}")
        print()
    rows = [row_for_config(config_name, Path(args.log_dir)) for config_name in configs]
    print_table(rows)


if __name__ == "__main__":
    main()
