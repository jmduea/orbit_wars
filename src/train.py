from __future__ import annotations

import json

from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf

from .config import train_config_from_omegaconf
from .jax_train import run_jax_training


@hydra_main(version_base="1.3", config_path="../conf", config_name="config")
def _hydra_entry(cfg_raw: DictConfig) -> None:
    cfg = train_config_from_omegaconf(cfg_raw)
    if cfg.print_resolved_config:
        payload = OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    run_jax_training(cfg, cfg.resume_checkpoint)


if __name__ == "__main__":
    _hydra_entry()
