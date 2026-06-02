from __future__ import annotations

from src.config import TrainConfig


def checkpoint_replay_due(cfg: TrainConfig, update: int) -> bool:
    if not cfg.artifacts.replay.enabled:
        return False
    if update == cfg.training.total_updates:
        return True
    if cfg.artifacts.replay.final_checkpoint_only:
        return False
    every_n = max(int(cfg.artifacts.replay.every_n_checkpoints), 1)
    checkpoint_index = max(update // max(int(cfg.artifacts.checkpoint_every), 1), 1)
    return checkpoint_index % every_n == 0
