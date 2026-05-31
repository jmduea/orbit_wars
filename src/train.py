from __future__ import annotations

import json
from pathlib import Path

from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf

from .config import register_runtime_resolvers, train_config_from_omegaconf
from .jax.train import run_jax_training

register_runtime_resolvers()


def _find_conf_dir() -> Path:
    """Find repo-local conf/ whether launched as python -m src.train or ow train."""

    candidates = [
        Path.cwd() / "conf",
        Path(__file__).resolve().parents[1] / "conf",
    ]

    for candidate in candidates:
        if (candidate / "config.yaml").exists():
            return candidate.resolve()

    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not find Hydra config directory containing config.yaml.\n"
        f"Checked:\n{checked}"
    )


def _run_training_from_cfg(cfg_raw: DictConfig) -> None:
    cfg = train_config_from_omegaconf(cfg_raw)

    if cfg.print_resolved_config:
        payload = OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    run_jax_training(cfg, cfg.resume_checkpoint)


def main() -> None:
    conf_dir = _find_conf_dir()

    @hydra_main(
        version_base="1.3",
        config_path=str(conf_dir),
        config_name="config",
    )
    def _entry(cfg_raw: DictConfig) -> None:
        _run_training_from_cfg(cfg_raw)

    _entry()


if __name__ == "__main__":
    main()
