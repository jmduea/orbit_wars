from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore


@dataclass(slots=True)
class EnvConfig:
    """Environment and feature-shape configuration shared by all backends."""
    candidate_count: int = 8
    ship_bucket_count: int = 8
    max_fleets: int = 256
    player_count: int = 2
    max_ships: float = 400.0
    reward_capture_planet: float = 0.0
    reward_ship_delta: float = 0.0
    reward_production_delta: float = 0.0
    reward_terminal_scale: float = 1.0
    early_terminal_reward_shaping_enabled: bool = True
    early_terminal_reward_shaping_horizon: int = 500
    terminal_reward_mode: str = "binary_win"
    feature_history_steps: int = 1
    trajectory_shield_enabled: bool = True
    trajectory_shield_hit_mode: str = "selected_target"
    trajectory_shield_horizon: int = 500
    trajectory_shield_epsilon: float = 1e-6


@dataclass(slots=True)
class ModelConfig:
    """Policy architecture and observation-normalization configuration."""

    architecture: str = "gnn_pointer"
    value_head: str = "shared"
    hidden_size: int = 128
    attention_heads: int = 4
    max_moves_k: int = 3
    gnn_k_neighbors: int = 5
    gnn_message_passing_layers: int = 2
    normalize_observations: bool = True
    obs_norm_clip: float = 10.0


@dataclass(slots=True)
class PPOConfig:
    """PPO rollout, optimization, and loss hyperparameters."""

    rollout_steps: int = 32
    num_envs: int = 4
    total_updates: int = 200
    epochs: int = 4
    minibatch_size: int = 512
    update_chunk_rows_min: int = 8192
    update_chunk_rows_max: int | None = None
    rollout_microbatch_envs: int | None = None
    enable_gradient_checkpointing: bool = False
    gamma: float = 0.99
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5


@dataclass(slots=True)
class TrainingFormatConfig:
    format_schedule: list[dict[str, Any]] = field(default_factory=list)
    format_mix: list[dict[str, Any]] = field(default_factory=list)
    rollout_groups: list[dict[str, Any]] = field(default_factory=list)
    phases: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class CurriculumSnapshotConfig:
    pool_size: int = 0
    interval_updates: int = 0
    deterministic: bool = True
    selection: str = "uniform"
    fallback: str = "latest"


@dataclass(slots=True)
class CurriculumConfig:
    enabled: bool = False
    snapshot: CurriculumSnapshotConfig = field(default_factory=CurriculumSnapshotConfig)
    stages: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class MetricGroupsConfig:
    core_progress: bool = True
    losses: bool = True
    timing: bool = True
    curriculum: bool = True
    opponent_composition: bool = False
    action_decision: bool = False
    game_state: bool = False
    trajectory_shield_debug: bool = False
    historical_pool: bool = False
    events: bool = True


@dataclass(slots=True)
class TelemetryConfig:
    metric_groups: MetricGroupsConfig = field(default_factory=MetricGroupsConfig)


@dataclass(slots=True)
class WandBConfig:
    enabled: bool = False
    project: str | None = None
    entity: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    log_artifacts: bool = False
    log_model_every: int = 100
    watch_model: bool = False


@dataclass(slots=True)
class OpponentMixConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "latest": 1.0,
            "historical": 0.0,
            "nearest_sniper": 0.0,
            "turtle": 0.0,
            "opportunistic": 0.0,
            "random": 0.0,
            "noop": 0.0,
        }
    )
    temperature: float = 1.0
    curriculum: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ReplayConfig:
    enabled: bool = False
    every_n_checkpoints: int = 1
    opponent: str = "random"
    seed_policy: str = "update"
    max_steps: int = 500
    output_dir: str = "replays"


@dataclass(slots=True)
class CheckpointRetentionConfig:
    keep_last_n: int = 5
    keep_every_n_updates: int = 0
    keep_best_k_by_metric: int = 0
    best_metric_name: str = "episode_reward_mean"
    best_metric_mode: str = "max"
    min_update_for_pruning: int = 0
    dry_run_pruning: bool = False


@dataclass(slots=True)
class ArtifactPipelineConfig:
    enabled: bool = True
    checkpoint_queue_size: int = 1
    checkpoint_timeout_seconds: float = 300.0
    final_flush_timeout_seconds: float = 900.0
    interrupt_flush_timeout_seconds: float = 60.0
    exception_flush_timeout_seconds: float = 60.0
    latest_lag_warning_updates: int = 1
    coalesce_intermediate_checkpoints: bool = True
    replay_async: bool = True
    replay_backend: str = "docker"
    docker_validation_async: bool = False
    docker_image: str = "gcr.io/kaggle-images/python-simulations"
    docker_player_count: str = "both"
    docker_timeout_seconds: float = 1.0
    worker_autostart: bool = True
    worker_poll_seconds: float = 5.0
    worker_idle_exit_seconds: float = 300.0
    ledger_enabled: bool = True
    queue_dir: str = "artifact_jobs"
    fail_training_on_checkpoint_error: bool = True
    fail_training_on_optional_job_error: bool = False


@dataclass(slots=True)
class TrainConfig:
    seed: int = 42
    run_name: str = "orbit_wars_template_ppo"
    device: str = "auto"
    save_dir: str = "artifacts/rl_template"
    checkpoint_every: int = 10
    log_every: int = 1
    opponent: str = "random"
    self_play_update_interval: int = 10
    self_play_deterministic: bool = False
    self_play_enabled: bool = False
    self_play_pool_size: int = 0
    self_play_snapshot_interval: int = 0
    self_play_latest_probability: float = 0.5
    multi_opponent_mode: str = "mixed"
    alternate_player_sides: bool = True
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    training_format: TrainingFormatConfig = field(default_factory=TrainingFormatConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    opponent_mix: OpponentMixConfig = field(default_factory=OpponentMixConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    checkpoint_retention: CheckpointRetentionConfig = field(
        default_factory=CheckpointRetentionConfig
    )
    artifact_pipeline: ArtifactPipelineConfig = field(
        default_factory=ArtifactPipelineConfig
    )
    reseed_every_updates: int = 0
    reseed_on_plateau: bool = False
    plateau_metric: str = "episode_reward_mean"
    plateau_window: int = 10
    plateau_delta: float = 0.0
    heldout_eval_seed_set: list[int] = field(default_factory=list)
    print_resolved_config: bool = False
    resume_checkpoint: str | None = None


def register_config_schemas() -> None:
    """Register structured Hydra schemas for train/runtime configuration."""

    cs = ConfigStore.instance()
    cs.store(name="train_config", node=TrainConfig)
