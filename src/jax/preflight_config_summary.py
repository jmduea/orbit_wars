"""Human-readable resolved training config for preflight gate runs."""

from __future__ import annotations

from src.config import compose_hydra_train_config
from src.config.schema import TrainConfig


def _last_group_override(overrides: list[str], group: str) -> str | None:
    prefix = f"{group}="
    value: str | None = None
    for item in overrides:
        if item.startswith(prefix) and item.count("=") == 1:
            value = item.split("=", 1)[1]
    return value


def _model_label(cfg: TrainConfig, overrides: list[str]) -> str:
    explicit = _last_group_override(overrides, "model")
    if explicit:
        return explicit
    decoder = cfg.model.pointer_decoder
    if decoder and decoder != "factorized_topk":
        return decoder
    return cfg.model.architecture


def _curriculum_label(cfg: TrainConfig, overrides: list[str]) -> str:
    explicit = _last_group_override(overrides, "curriculum")
    if explicit:
        return explicit
    return "on" if cfg.curriculum.enabled else "off"


def format_gate_train_config_summary(overrides: list[str]) -> tuple[str, ...]:
    """Resolve gate overrides and return multi-line stderr-friendly summary."""

    cfg = compose_hydra_train_config(overrides)
    training = cfg.training
    format_weights = training.format_weights or {}
    format_line = ", ".join(
        f"{player_count}p={weight:g}"
        for player_count, weight in sorted(format_weights.items())
    )
    training_group = _last_group_override(overrides, "training") or "(defaults)"
    task_group = _last_group_override(overrides, "task") or "(defaults)"

    return (
        "Resolved gate training config:",
        f"  model: {_model_label(cfg, overrides)}",
        f"  training group: {training_group}",
        f"  task group: {task_group}",
        f"  curriculum: {_curriculum_label(cfg, overrides)}",
        "  geometry:",
        f"    num_envs={training.num_envs}  format_weights={{{format_line}}}",
        (
            f"    rollout_steps={training.rollout_steps}  "
            f"total_updates={training.total_updates}  "
            f"candidate_count={cfg.task.candidate_count}  "
            f"max_moves_k={cfg.model.max_moves_k}"
        ),
        "  PPO:",
        (
            f"    lr={training.lr:g}  clip_coef={training.clip_coef:g}  "
            f"ent_coef={training.ent_coef:g}  vf_coef={training.vf_coef:g}"
        ),
        (
            f"    epochs={training.epochs}  max_grad_norm={training.max_grad_norm:g}  "
            f"update_chunk_rows={training.update_chunk_rows}"
        ),
        f"    reseed_every_updates={training.reseed_every_updates}",
    )
