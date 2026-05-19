from __future__ import annotations

import argparse
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import TrainConfig, default_train_config_path


def _to_ordered_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _to_ordered_plain(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, list):
        return [_to_ordered_plain(item) for item in value]
    if isinstance(value, tuple):
        return [_to_ordered_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_ordered_plain(item) for key, item in value.items()}
    return value


def render_default_cfg() -> str:
    cfg = TrainConfig()
    data = _to_ordered_plain(cfg)
    return yaml.safe_dump(data, sort_keys=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate canonical default_cfg.yaml from TrainConfig dataclass defaults."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_train_config_path(),
        help="Path to write the generated YAML template.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate output matches the committed template without rewriting.",
    )
    args = parser.parse_args()

    rendered = render_default_cfg()
    output_path = args.output
    existing = output_path.read_text(encoding="utf-8") if output_path.exists() else None

    if args.check:
        if existing != rendered:
            print(
                f"{output_path} is out of date. Run: uv run python scripts/generate_default_cfg.py"
            )
            return 1
        print(f"{output_path} is up to date.")
        return 0

    output_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote canonical config template to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
