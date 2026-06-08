"""Shared opponent ablation ladder override bundles (ce-optimize pre-loop)."""

from __future__ import annotations

LADDER_RUNG_OVERRIDES: dict[str, list[str]] = {
    "noop": ["train_bundle=opponent_recovery_floor"],
    "recovery": ["train_bundle=opponent_recovery"],
    "scripted_heavy": ["train_bundle=scripted_heavy"],
    "self_play": ["train_bundle=self_play_stage"],
    "production_mix": ["train_bundle=production_mix"],
}

LADDER_RUNG_ORDER: tuple[str, ...] = (
    "noop",
    "recovery",
    "scripted_heavy",
    "self_play",
    "production_mix",
)

PROFILE_BASE_OVERRIDES: list[str] = ["task=map_pool"]

THROUGHPUT_SHARED_OVERRIDES: list[str] = [
    "model=transformer_factorized_small",
    "model.max_moves_k=2",
    "task=shield_cheap",
    "task=map_pool",
    "training=2p4p_32_split",
    "training.rollout_steps=256",
    "task.candidate_count=3",
    "telemetry.wandb.enabled=false",
    "artifacts.artifact_pipeline.enabled=false",
    "artifacts.replay.enabled=false",
    "seed=42",
]
