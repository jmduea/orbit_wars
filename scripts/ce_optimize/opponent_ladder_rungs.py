"""Shared opponent ablation ladder override bundles (ce-optimize pre-loop)."""

from __future__ import annotations

LADDER_RUNG_OVERRIDES: dict[str, list[str]] = {
    "noop": [
        "opponents=base",
        "opponents.mode.opponent=noop",
        "curriculum=off",
        "opponents.self_play.enabled=false",
        "opponents.snapshot.pool_size=0",
        "opponents.snapshot.interval_updates=0",
    ],
    "scripted_heavy": [
        "curriculum=scripted_heavy",
        "opponents=base",
        "opponents.self_play.enabled=false",
        "opponents.snapshot.pool_size=0",
        "opponents.snapshot.interval_updates=0",
    ],
    "self_play": [
        "curriculum=self_play_only_stage",
        "opponents=base",
        "opponents.self_play.enabled=true",
        "opponents.snapshot.pool_size=2",
        "opponents.snapshot.interval_updates=10",
    ],
    "production_mix": [
        "opponents=default",
        "curriculum=default",
    ],
}

LADDER_RUNG_ORDER: tuple[str, ...] = (
    "noop",
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
