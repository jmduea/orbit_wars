from __future__ import annotations

import json
import sys
from pathlib import Path

from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf

from .config import train_config_from_omegaconf
from .jax_train import run_jax_training


def _extract_legacy_cli_args(argv: list[str]) -> tuple[str | None, str | None]:
    legacy_config = None
    resume_checkpoint = None
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token == "--config" and idx + 1 < len(argv):
            legacy_config = argv[idx + 1]
            idx += 2
            continue
        if token.startswith("--config="):
            legacy_config = token.split("=", maxsplit=1)[1]
            idx += 1
            continue
        if token == "--resume-checkpoint" and idx + 1 < len(argv):
            resume_checkpoint = argv[idx + 1]
            idx += 2
            continue
        if token.startswith("--resume-checkpoint="):
            resume_checkpoint = token.split("=", maxsplit=1)[1]
            idx += 1
            continue
        idx += 1
    return legacy_config, resume_checkpoint


def _legacy_config_to_preset(path: str) -> str | None:
    name = Path(path).name
    return {
        "default_cfg.yaml": "jax",
        "attention_training.yaml": "attention",
        "jax_training.yaml": "jax",
        "shaped_reward_training.yaml": "shaped_reward",
        "attention_self_play_pool.yaml": "self_play",
        "attention_candidates_16.yaml": "candidate_sweep",
        "mixed_2p_4p_training.yaml": "jax_mixed_2p_4p",
    }.get(name)


@hydra_main(version_base=None, config_path="../conf", config_name="config")
def _hydra_entry(cfg_raw: DictConfig) -> None:
    cfg = train_config_from_omegaconf(cfg_raw)
    if cfg.print_resolved_config:
        payload = OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    _, resume_checkpoint = _extract_legacy_cli_args(sys.argv[1:])
    run_jax_training(cfg, resume_checkpoint)


if __name__ == "__main__":
    _hydra_entry()
