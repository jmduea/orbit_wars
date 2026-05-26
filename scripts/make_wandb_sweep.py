from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

COMMANDS: dict[str, list[str]] = {
    "ow_train": [
        "${env}",
        "uv",
        "run",
        "ow",
        "train",
        "${args_no_hyphens}",
    ],
}


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_none(item) for key, item in value.items() if item is not None
        }
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value


@hydra.main(version_base="1.3", config_path="../conf", config_name="sweep_gen")
def main(cfg: DictConfig) -> None:
    plain = OmegaConf.to_container(cfg, resolve=True)

    if not isinstance(plain, dict):
        raise TypeError("Expected composed sweep config to be a mapping.")

    out_dir = Path(str(plain.pop("out_dir", "artifacts/sweeps")))
    out_dir.mkdir(parents=True, exist_ok=True)

    command_name = str(plain.pop("command_name", "ow_train"))
    if command_name not in COMMANDS:
        raise ValueError(
            f"Unknown command_name={command_name!r}. Known commands: {sorted(COMMANDS)}"
        )

    name = str(plain.get("name") or "sweep")
    description = plain.pop("description", None)

    sweep = {
        "method": plain.get("method", "grid"),
        "metric": plain.get(
            "metric",
            {"name": "overall_win_rate", "goal": "maximize"},
        ),
        "command": COMMANDS[command_name],
        "parameters": plain.get("parameters", {}),
    }

    if plain.get("run_cap") is not None:
        sweep["run_cap"] = plain["run_cap"]

    if description is not None:
        sweep["description"] = description

    sweep = _drop_none(sweep)

    out_path = out_dir / f"{name}.yaml"
    out_path.write_text(
        yaml.safe_dump(sweep, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
