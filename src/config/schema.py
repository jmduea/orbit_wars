from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore


@dataclass(slots=True)
class TaskConfig:
    """Environment and feature-shape configuration shared by all backends."""

    # TODO: FeatureEngineeringConfig or something more feature-adjacent.
    max_fleets: int = 256
    player_count: int = 2
    ship_speed: float = 6.0
    max_ships: float = 400.0
    ship_feature_scale: float = 1000.0
    feature_history_steps: int = 1
    # TODO: Move to ActionCodecConfig or something more action-adjacent.
    candidate_count: int = 8
    ship_bucket_count: int = 8
    ship_action_mode: str = "buckets"  # continuous_fraction for sigmoid fraction head
    trajectory_shield_mode: str = "cheap"  # off | cheap | tiered | exact
    rollout_factorized_sampling: str = "lattice"  # lattice: full cheap shield mask per K-step; selected_validate: unshielded sample + point check
    # off: no trajectory filtering beyond ordinary action legality
    # cheap: feature-derived source/target/bucket mask, no horizon scan
    # exact: current full per-edge/per-bucket trajectory shield
    # tiered: cheap sampling + exact selected-launch validation
    trajectory_shield_final_validate_selected: bool = False
    trajectory_shield_hit_mode: str = "selected_target"
    trajectory_shield_horizon: int = 500
    trajectory_shield_epsilon: float = 1e-6
    intercept_anchors: tuple[float, ...] = (1.0, 3.0, 6.0)
    edge_rank_mode: str = "snapshot"  # intercept_min for intercept-proximity top-K
    # train: JAX-native reset/step (no pure_callback); kaggle: reference planet/comet paths
    env_parity_mode: str = (
        "train"  # train | kaggle | legacy (legacy = pre-#188 comet-free hot path)
    )
    map_pool_path: str | None = None
    map_pool_sha256: str | None = None


@dataclass(slots=True)
class RewardConfig:
    """Reward shaping and terminal reward configuration."""

    reward_capture_planet: float = 0.0
    reward_ship_delta: float = 0.0
    reward_production_delta: float = 0.0
    reward_terminal_scale: float = 1.0
    early_terminal_reward_shaping_enabled: bool = True
    early_terminal_reward_shaping_horizon: int = 500
    terminal_reward_mode: str = "binary_win"


@dataclass(slots=True)
class PlanetFlowConfig:
    """Experimental Planet Flow action-layout configuration."""

    pressure_bucket_values: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(slots=True)
class ModelConfig:
    """Policy architecture and observation-normalization configuration."""

    architecture: str = "planet_graph_transformer"
    value_head: str = (
        "shared"  # format_routed for 2p/4p routing; distributional for C51 critic
    )
    value_bins: int = 51
    value_max: float = 1.0
    hidden_size: int = 128
    attention_heads: int = 4
    max_moves_k: int = 3
    planet_transformer_layers: int = 2
    spatial_attention_bias: bool = True
    pointer_decoder: str = "factorized_topk"
    decoder_carry: bool = False
    planet_flow: PlanetFlowConfig = field(default_factory=PlanetFlowConfig)
    normalize_observations: bool = True
    obs_norm_clip: float = 10.0


@dataclass(slots=True)
class TrainingConfig:
    """PPO rollout, optimization, and loss hyperparameters."""

    rollout_steps: int = 32
    num_envs: int = 4
    format_weights: dict[int, float] = field(default_factory=dict)
    total_updates: int = 200
    epochs: int = 1
    update_chunk_rows: int = 1024
    rollout_microbatch_envs: int | None = (
        None  # Keep since it's used in the rollout group config
    )
    rotate_format_rollouts: bool = False
    lean_rollout_metrics: bool = False  # TODO: telemetry?
    enable_gradient_checkpointing: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.15
    ent_coef: float = 0.006
    vf_coef: float = 1.0
    lr: float = 6e-5
    max_grad_norm: float = 1.0
    log_every: int = 1  # TODO: telemetry?
    reseed_every_updates: int = (
        50  # 0=off, -1=auto max(25, total_updates//10); default pinned by calibration
    )
    reseed_on_plateau: bool = False
    plateau_metric: str = "episode_reward_mean"
    plateau_window: int = 10
    plateau_delta: float = 0.0
    debug_replay_parity: bool = False


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
    debug: bool = False
    rollout_phase_timing: bool = False


@dataclass(slots=True)
class TelemetryConfig:
    metric_groups: MetricGroupsConfig = field(default_factory=MetricGroupsConfig)
    wandb: WandBConfig = field(default_factory=lambda: WandBConfig())


@dataclass(slots=True)
class WandBConfig:
    enabled: bool = False
    project: str | None = None
    entity: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    tags_from_config_groups: bool = True
    tag_config_groups: list[str] = field(
        default_factory=lambda: [
            "model",
            "training",
            "opponents",
            "curriculum",
            "reward",
        ]
    )
    rename_from_swept_params: bool = True
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


@dataclass(slots=True)
class OpponentSelfPlayConfig:
    enabled: bool = False
    update_interval: int = 10
    deterministic: bool = False


@dataclass(slots=True)
class OpponentModeConfig:
    opponent: str = "random"
    alternate_player_sides: bool = True


@dataclass(slots=True)
class BracketSelfPlayConfig:
    """Sample self-play opponents from main tournament bracket entries."""

    enabled: bool = False
    sample_count: int = 1


@dataclass(slots=True)
class OpponentsConfig:
    self_play: OpponentSelfPlayConfig = field(default_factory=OpponentSelfPlayConfig)
    mode: OpponentModeConfig = field(default_factory=OpponentModeConfig)
    mix: OpponentMixConfig = field(default_factory=OpponentMixConfig)
    snapshot: CurriculumSnapshotConfig = field(default_factory=CurriculumSnapshotConfig)
    bracket_self_play: BracketSelfPlayConfig = field(
        default_factory=BracketSelfPlayConfig
    )


@dataclass(slots=True)
class ReplayConfig:
    enabled: bool = False
    every_n_checkpoints: int = 1
    final_checkpoint_only: bool = False
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
class PromotionTournamentConfig:
    min_win_rate_vs_sniper: float = 0.55
    min_win_rate_vs_incumbent: float = 0.51
    min_first_place_rate_4p: float | None = None
    require_head_to_head: bool = True

    def __setattr__(self, name: str, value: object) -> None:
        if name == "min_win_rate_vs_baseline":
            name = "min_win_rate_vs_sniper"
        object.__setattr__(self, name, value)


@dataclass(slots=True)
class PromotionConfig:
    enabled: bool = True
    strategy: str = "metric"
    metric_name: str = "episode_reward_mean"
    metric_mode: str = "max"
    tournament: PromotionTournamentConfig = field(
        default_factory=PromotionTournamentConfig
    )


@dataclass(slots=True)
class BracketTrainingConfig:
    """Training-time qualifier ladder and 500M env-step budget tracking."""

    enabled: bool = False
    qualifier_max_env_steps: int = 500_000_000
    qualifier_eval_interval_updates: int = 50


@dataclass(slots=True)
class SsotPipelineConfig:
    """SSOT long-train profile: JAX qualifiers, rollout curriculum stages, W&B on."""

    enabled: bool = False
    qualifier_max_env_steps: int = 500_000_000
    qualifier_eval_interval_updates: int = 50
    qualifier_games_per_seed: int = 0


@dataclass(slots=True)
class UnifiedTournamentConfig:
    """Held-out unified ladder settings for Gate 5 and hybrid checkpoint_eval."""

    enabled: bool = False
    qualifier_mode: bool = False
    enforcement: bool = False
    games_per_pair: int = 4
    prerequisite_seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    incumbent_seeds: list[int] = field(default_factory=lambda: list(range(30)))
    four_p_baseline_fillers: list[str] = field(
        default_factory=lambda: ["noop", "random", "random"]
    )
    noop_min_combined: float = 0.7
    random_min_combined: float = 0.58
    incumbent_bootstrap_opponent: str = "nearest_sniper"
    max_steps: int = 500
    per_step_seconds: float = 1.0
    overage_budget_seconds: float = 60.0
    write_replays: bool = False


@dataclass(slots=True)
class TournamentConfig:
    enabled: bool = False
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    games_per_pair: int = 1
    max_steps: int = 500
    baselines: list[str] = field(default_factory=lambda: ["sniper"])
    formats: list[str] = field(
        default_factory=lambda: ["2p_vs_baseline", "2p_head_to_head"]
    )
    output_subdir: str = "tournament"
    write_replays: bool = False
    per_step_seconds: float = 1.0
    overage_budget_seconds: float = 60.0


@dataclass(slots=True)
class ArtifactPipelineConfig:
    enabled: bool = True
    checkpoint_queue_size: int = 1
    checkpoint_timeout_seconds: float = 300.0
    final_flush_timeout_seconds: float = 900.0
    interrupt_flush_timeout_seconds: float = 60.0
    exception_flush_timeout_seconds: float = 60.0
    coalesce_intermediate_checkpoints: bool = True
    replay_async: bool = True
    replay_backend: str = "docker"
    docker_validation_async: bool = False
    checkpoint_eval_async: bool = False
    docker_image: str = "gcr.io/kaggle-images/python-simulations"
    docker_player_count: str = "both"
    docker_timeout_seconds: float = 1.0
    worker_autostart: bool = True
    worker_poll_seconds: float = 5.0
    worker_idle_exit_seconds: float = 300.0
    ledger_enabled: bool = True
    queue_dir: str = "queue/optional_jobs"
    result_dir: str = "evaluations"
    fail_training_on_checkpoint_error: bool = True


@dataclass(slots=True)
class ArtifactsConfig:
    save_dir: str = "outputs"
    checkpoint_every: int = 10
    artifact_pipeline: ArtifactPipelineConfig = field(
        default_factory=ArtifactPipelineConfig
    )
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    checkpoint_retention: CheckpointRetentionConfig = field(
        default_factory=CheckpointRetentionConfig
    )
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    tournament: TournamentConfig = field(default_factory=TournamentConfig)
    unified_tournament: UnifiedTournamentConfig = field(
        default_factory=UnifiedTournamentConfig
    )
    bracket_training: BracketTrainingConfig = field(
        default_factory=BracketTrainingConfig
    )
    ssot_pipeline: SsotPipelineConfig = field(default_factory=SsotPipelineConfig)


@dataclass(slots=True)
class OutputConfig:
    root: str = "outputs"
    campaign: str = "scratch"
    run_id: str = "${orbit_run_id:${seed}}"
    retention_class: str = "compact"
    indexes_dir: str = "indexes"
    cache_dir: str = "cache"
    wandb_dir: str = "cache/wandb"
    wandb_artifact_dir: str = "cache/wandb-artifacts"
    wandb_data_dir: str = "cache/wandb-data"


@dataclass(slots=True)
class TrainConfig:
    seed: int = 42
    run_name: str = "ow"
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    opponents: OpponentsConfig = field(default_factory=OpponentsConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    heldout_eval_seed_set: list[int] = field(default_factory=list)
    training_seed_set: list[int] = field(default_factory=list)
    eval_seed_set: list[int] = field(default_factory=lambda: [43, 44, 45, 46])
    print_resolved_config: bool = False
    resume_checkpoint: str | None = None
    from_promoted: str | None = None


def register_config_schemas() -> None:
    """Register structured Hydra schemas for train/runtime configuration."""

    cs = ConfigStore.instance()
    cs.store(name="train_config", node=TrainConfig)
