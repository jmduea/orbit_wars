from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

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


def _find_conf_dir() -> Path:
    candidates = [
        Path.cwd() / "conf",
        Path(__file__).resolve().parents[1] / "conf",
    ]
    for candidate in candidates:
        if (candidate / "sweep_gen.yaml").exists():
            return candidate.resolve()
    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not find Hydra config directory containing sweep_gen.yaml.\n"
        f"Checked:\n{checked}"
    )


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_none(item) for key, item in value.items() if item is not None
        }
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value


def compose_sweep_gen(overrides: list[str] | None = None) -> dict[str, Any]:
    """Compose ``sweep_gen`` without creating a Hydra run directory."""

    config_dir = _find_conf_dir()
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
            cfg = compose(config_name="sweep_gen", overrides=list(overrides or []))
            plain = OmegaConf.to_container(cfg, resolve=True)
    finally:
        GlobalHydra.instance().clear()

    if not isinstance(plain, dict):
        raise TypeError("Expected composed sweep config to be a mapping.")
    return cast(dict[str, Any], plain)


def write_wandb_sweep(cfg: dict[str, Any]) -> Path:
    plain = dict(cfg)

    out_dir = Path(str(plain.pop("out_dir", "outputs/_meta/sweeps")))
    out_dir.mkdir(parents=True, exist_ok=True)

    command_name = str(plain.pop("command_name", "ow_train"))
    if command_name not in COMMANDS:
        raise ValueError(
            f"Unknown command_name={command_name!r}. Known commands: {sorted(COMMANDS)}"
        )

    name = str(plain.get("name") or "sweep")
    description = plain.pop("description", None)

    metric = plain.get(
        "metric",
        {"name": "overall_win_rate", "goal": "maximize"},
    )
    goal = str(metric.get("goal", "maximize")).lower()
    metric_mode = "max" if "max" in goal else "min"
    parameters = dict(plain.get("parameters", {}))
    parameters.setdefault(
        "artifacts.promotion.metric_name", {"value": metric.get("name")}
    )
    parameters.setdefault(
        "artifacts.promotion.metric_mode", {"value": metric_mode}
    )

    sweep = {
        "method": plain.get("method", "grid"),
        "metric": metric,
        "command": COMMANDS[command_name],
        "parameters": parameters,
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
    return out_path


def _print_help() -> None:
    print(
        "Generate a W&B sweep YAML from conf/wandb_sweep recipes.\n\n"
        "Usage:\n"
        "  uv run ow make wandb_sweep=<recipe>\n"
        "  uv run python scripts/make_wandb_sweep.py wandb_sweep=<recipe>\n\n"
        "Recipes live under conf/wandb_sweep/. Output: outputs/_meta/sweeps/<name>.yaml"
    )


def main(argv: list[str] | None = None) -> None:
    overrides = list(sys.argv[1:] if argv is None else argv)
    if any(token in overrides for token in ("--help", "-h")):
        _print_help()
        return

    out_path = write_wandb_sweep(compose_sweep_gen(overrides))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
