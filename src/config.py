from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .conf_schema import (
    ArtifactPipelineConfig,
    EnvConfig,
    TrainConfig,
    register_config_schemas,
)
from .metric_registry import (
    CURRICULUM_PROMOTION_METRIC_NAMES,
    validate_scalar_update_metric_name,
)

_CURRICULUM_FAMILIES = {
    "latest",
    "historical",
    "random",
    "noop",
    "nearest_sniper",
    "turtle",
    "opportunistic",
}

__all__ = [
    "EnvConfig",
    "ArtifactPipelineConfig",
    "TrainConfig",
    "compose_hydra_train_config",
    "train_config_from_omegaconf",
]


def compose_hydra_train_config(overrides: list[str] | None = None) -> TrainConfig:
    """Compose the repository root Hydra config with optional overrides."""

    config_dir = Path(__file__).resolve().parents[1] / "conf"
    override_list = overrides or []
    register_config_schemas()
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        composed = compose(config_name="config", overrides=override_list)
    return train_config_from_omegaconf(composed, overrides=override_list)


def train_config_from_omegaconf(
    cfg_raw: Any, overrides: list[str] | None = None
) -> TrainConfig:
    """Convert a Hydra/OmegaConf object into a validated ``TrainConfig``."""

    _validate_responsibility_override_conflicts(overrides or _active_hydra_overrides())
    cfg_normalized = _normalize_responsibility_config(cfg_raw)
    _validate_no_legacy_format_conflicts(cfg_normalized)
    merged = OmegaConf.merge(OmegaConf.structured(TrainConfig), cfg_normalized)
    cfg: TrainConfig = OmegaConf.to_object(merged)
    cfg.heldout_eval_seed_set = _parse_seed_set(cfg.heldout_eval_seed_set)
    _validate_train_config(cfg)
    return cfg


_PPO_FIELDS = {
    "rollout_steps",
    "num_envs",
    "total_updates",
    "epochs",
    "minibatch_size",
    "update_chunk_rows_min",
    "update_chunk_rows_max",
    "rollout_microbatch_envs",
    "enable_gradient_checkpointing",
    "gamma",
    "clip_coef",
    "ent_coef",
    "vf_coef",
    "lr",
    "max_grad_norm",
}

_TOP_LEVEL_TRAINING_FIELDS = {
    "reseed_every_updates",
    "reseed_on_plateau",
    "plateau_metric",
    "plateau_window",
    "plateau_delta",
}

_RESPONSIBILITY_OVERRIDE_CONFLICTS = {
    "task": (
        "env.candidate_count",
        "env.ship_bucket_count",
        "env.max_fleets",
        "env.player_count",
        "env.max_ships",
        "env.feature_history_steps",
        "env.trajectory_shield_",
    ),
    "reward": ("env.reward_", "env.terminal_reward_mode", "env.early_terminal_reward_shaping_"),
    "training": ("ppo.", "reseed_", "plateau_"),
    "format": ("training_format.",),
    "opponents": (
        "opponent",
        "multi_opponent_mode",
        "alternate_player_sides",
        "self_play_",
        "opponent_mix.",
    ),
    "telemetry.wandb": ("wandb.",),
    "artifacts": ("artifact_pipeline.", "replay.", "checkpoint_retention.", "save_dir", "checkpoint_every"),
}


def _active_hydra_overrides() -> list[str]:
    try:
        from hydra.core.hydra_config import HydraConfig
    except ImportError:
        return []
    if not HydraConfig.initialized():
        return []
    overrides = getattr(HydraConfig.get(), "overrides", None)
    task_overrides = getattr(overrides, "task", None) if overrides is not None else None
    return [str(override) for override in task_overrides or []]


def _validate_responsibility_override_conflicts(overrides: list[str]) -> None:
    normalized = [_override_key(override) for override in overrides]
    for new_prefix, old_prefixes in _RESPONSIBILITY_OVERRIDE_CONFLICTS.items():
        has_new = any(
            key == new_prefix or key.startswith(f"{new_prefix}.") for key in normalized
        )
        if not has_new:
            continue
        old_hit = next(
            (
                key
                for key in normalized
                if not (key == new_prefix or key.startswith(f"{new_prefix}."))
                and any(key.startswith(old_prefix) for old_prefix in old_prefixes)
            ),
            None,
        )
        if old_hit is not None:
            raise ValueError(
                f"Conflicting config overrides: use either {new_prefix} or legacy {old_hit}, not both."
            )


def _override_key(override: str) -> str:
    text = override.lstrip("+")
    for separator in ("=", "~"):
        if separator in text:
            return text.split(separator, maxsplit=1)[0]
    return text


def _normalize_responsibility_config(cfg_raw: Any) -> Any:
    raw = OmegaConf.to_container(cfg_raw, resolve=False) if cfg_raw is not None else {}
    if not isinstance(raw, dict):
        return cfg_raw

    normalized = deepcopy(raw)
    task = normalized.pop("task", None)
    if isinstance(task, dict):
        _merge_runtime_section(normalized, "env", task)

    reward = normalized.pop("reward", None)
    if isinstance(reward, dict):
        _merge_runtime_section(normalized, "env", reward)

    training = normalized.pop("training", None)
    if isinstance(training, dict):
        ppo_values = {
            key: value for key, value in training.items() if key in _PPO_FIELDS
        }
        top_level_values = {
            key: value
            for key, value in training.items()
            if key in _TOP_LEVEL_TRAINING_FIELDS
        }
        if ppo_values:
            _merge_runtime_section(normalized, "ppo", ppo_values)
        for key, value in top_level_values.items():
            _set_runtime_value(normalized, key, value)

    training_format = normalized.pop("format", None)
    if isinstance(training_format, dict):
        _merge_runtime_section(normalized, "training_format", training_format)

    opponents = normalized.pop("opponents", None)
    if isinstance(opponents, dict):
        _normalize_opponents_config(normalized, opponents)

    telemetry = normalized.get("telemetry")
    if isinstance(telemetry, dict) and isinstance(telemetry.get("wandb"), dict):
        wandb = telemetry.pop("wandb")
        _merge_runtime_section(normalized, "wandb", wandb)

    artifacts = normalized.pop("artifacts", None)
    if isinstance(artifacts, dict):
        _normalize_artifacts_config(normalized, artifacts)

    _drop_none_values(normalized)
    return OmegaConf.create(normalized)


def _normalize_opponents_config(normalized: dict[str, Any], opponents: dict[str, Any]) -> None:
    mode = opponents.get("mode")
    if isinstance(mode, dict):
        for source_key, target_key in (
            ("opponent", "opponent"),
            ("multi_opponent_mode", "multi_opponent_mode"),
            ("alternate_player_sides", "alternate_player_sides"),
        ):
            if source_key in mode:
                _set_runtime_value(normalized, target_key, mode[source_key])

    self_play = opponents.get("self_play")
    if isinstance(self_play, dict):
        for source_key, target_key in (
            ("enabled", "self_play_enabled"),
            ("update_interval", "self_play_update_interval"),
            ("deterministic", "self_play_deterministic"),
        ):
            if source_key in self_play:
                _set_runtime_value(normalized, target_key, self_play[source_key])

    mix = opponents.get("mix")
    if isinstance(mix, dict):
        opponent_mix: dict[str, Any] = {}
        if isinstance(mix.get("weights"), dict):
            opponent_mix["weights"] = mix["weights"]
        if "temperature" in mix:
            opponent_mix["temperature"] = mix["temperature"]
        if "curriculum" in mix:
            opponent_mix["curriculum"] = mix["curriculum"]
        if opponent_mix:
            _merge_runtime_section(normalized, "opponent_mix", opponent_mix)

    snapshot = opponents.get("snapshot")
    if isinstance(snapshot, dict):
        if "pool_size" in snapshot:
            _set_runtime_value(normalized, "self_play_pool_size", snapshot["pool_size"])
        if "interval_updates" in snapshot:
            _set_runtime_value(
                normalized,
                "self_play_snapshot_interval", snapshot["interval_updates"]
            )
        curriculum_snapshot = {
            key: value
            for key, value in snapshot.items()
            if key in {"pool_size", "interval_updates", "deterministic", "selection", "fallback"}
        }
        if curriculum_snapshot:
            curriculum = normalized.setdefault("curriculum", {})
            if isinstance(curriculum, dict):
                _merge_runtime_section(curriculum, "snapshot", curriculum_snapshot)


def _normalize_artifacts_config(normalized: dict[str, Any], artifacts: dict[str, Any]) -> None:
    for key in ("save_dir", "checkpoint_every"):
        if key in artifacts:
            _set_runtime_value(normalized, key, artifacts[key])
    for key in ("artifact_pipeline", "replay", "checkpoint_retention"):
        value = artifacts.get(key)
        if isinstance(value, dict):
            _merge_runtime_section(normalized, key, value)


def _merge_runtime_section(
    normalized: dict[str, Any], section_name: str, source: dict[str, Any]
) -> None:
    existing = normalized.get(section_name)
    if isinstance(existing, dict):
        merged = deepcopy(source)
        _deep_update(merged, existing)
        normalized[section_name] = merged
    elif existing is None:
        normalized[section_name] = deepcopy(source)


def _set_runtime_value(normalized: dict[str, Any], key: str, value: Any) -> None:
    if key not in normalized or normalized[key] is None:
        normalized[key] = deepcopy(value)


def _deep_update(base: dict[str, Any], overlay: dict[str, Any]) -> None:
    for key, value in overlay.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _drop_none_values(value: Any) -> None:
    if not isinstance(value, dict):
        return
    for key in list(value):
        item = value[key]
        if item is None:
            del value[key]
        elif isinstance(item, dict):
            _drop_none_values(item)


def _validate_train_config(cfg: TrainConfig) -> None:
    _validate_registered_update_metric_name(
        cfg.checkpoint_retention.best_metric_name,
        field_name="checkpoint_retention.best_metric_name",
    )
    _validate_registered_update_metric_name(
        cfg.plateau_metric,
        field_name="plateau_metric",
    )

    env = cfg.env
    if int(env.feature_history_steps) <= 0:
        raise ValueError("env.feature_history_steps must be a positive integer.")
    if env.trajectory_shield_hit_mode not in {"selected_target", "non_friendly"}:
        raise ValueError(
            "env.trajectory_shield_hit_mode must be 'selected_target' or 'non_friendly'."
        )
    if int(env.trajectory_shield_horizon) <= 0:
        raise ValueError("env.trajectory_shield_horizon must be a positive integer.")
    if float(env.trajectory_shield_epsilon) < 0.0:
        raise ValueError("env.trajectory_shield_epsilon must be non-negative.")

    value_head = cfg.model.value_head.strip().lower()
    if value_head not in {"shared", "format_routed"}:
        raise ValueError("model.value_head must be 'shared' or 'format_routed'.")

    ppo = cfg.ppo
    if int(ppo.update_chunk_rows_min) <= 0:
        raise ValueError("ppo.update_chunk_rows_min must be a positive integer.")
    if ppo.update_chunk_rows_max is not None and int(ppo.update_chunk_rows_max) <= 0:
        raise ValueError("ppo.update_chunk_rows_max must be a positive integer when set.")
    if (
        ppo.update_chunk_rows_max is not None
        and int(ppo.update_chunk_rows_max) < int(ppo.update_chunk_rows_min)
    ):
        raise ValueError(
            "ppo.update_chunk_rows_max must be >= ppo.update_chunk_rows_min when both are set."
        )
    if ppo.rollout_microbatch_envs is not None and int(ppo.rollout_microbatch_envs) <= 0:
        raise ValueError("ppo.rollout_microbatch_envs must be a positive integer when set.")

    artifact_pipeline = cfg.artifact_pipeline
    if int(artifact_pipeline.checkpoint_queue_size) <= 0:
        raise ValueError("artifact_pipeline.checkpoint_queue_size must be positive.")
    for field_name in (
        "checkpoint_timeout_seconds",
        "final_flush_timeout_seconds",
        "interrupt_flush_timeout_seconds",
        "exception_flush_timeout_seconds",
        "docker_timeout_seconds",
        "worker_poll_seconds",
        "worker_idle_exit_seconds",
    ):
        if float(getattr(artifact_pipeline, field_name)) <= 0.0:
            raise ValueError(f"artifact_pipeline.{field_name} must be positive.")
    if artifact_pipeline.replay_backend not in {"docker", "local"}:
        raise ValueError("artifact_pipeline.replay_backend must be 'docker' or 'local'.")
    if artifact_pipeline.docker_player_count not in {"2", "4", "both"}:
        raise ValueError("artifact_pipeline.docker_player_count must be '2', '4', or 'both'.")
    if int(artifact_pipeline.latest_lag_warning_updates) < 0:
        raise ValueError("artifact_pipeline.latest_lag_warning_updates must be non-negative.")
    if not str(artifact_pipeline.queue_dir).strip():
        raise ValueError("artifact_pipeline.queue_dir must be a non-empty relative path.")
    if Path(artifact_pipeline.queue_dir).is_absolute():
        raise ValueError("artifact_pipeline.queue_dir must be relative to the run directory.")

    _validate_curriculum_config(cfg)

    if cfg.curriculum.enabled:
        return

    if not cfg.self_play_enabled:
        if cfg.self_play_pool_size != 0:
            raise ValueError(
                "self_play_pool_size must be 0 when self_play_enabled is false."
            )
        if cfg.self_play_snapshot_interval != 0:
            raise ValueError(
                "self_play_snapshot_interval must be 0 when self_play_enabled is false."
            )
        if cfg.opponent_mix.curriculum:
            raise ValueError(
                "opponent_mix.curriculum must be empty when self_play_enabled is false."
            )
        historical_weight = float(cfg.opponent_mix.weights.get("historical", 0.0))
        if historical_weight > 0.0:
            raise ValueError(
                "opponent_mix.weights.historical must be 0 when self_play_enabled is false."
            )
    else:
        if cfg.self_play_pool_size <= 0:
            raise ValueError(
                "self_play_pool_size must be > 0 when self_play_enabled is true."
            )
        if cfg.self_play_snapshot_interval <= 0:
            raise ValueError(
                "self_play_snapshot_interval must be > 0 when self_play_enabled is true."
            )


def _validate_curriculum_config(cfg: TrainConfig) -> None:
    curriculum = cfg.curriculum
    if not curriculum.enabled:
        return
    if cfg.training_format.phases:
        raise ValueError(
            "training_format.phases is deprecated when curriculum.enabled is true; "
            "migrate progressive difficulty to curriculum.stages."
        )
    if cfg.opponent_mix.curriculum:
        raise ValueError(
            "opponent_mix.curriculum is deprecated when curriculum.enabled is true; "
            "migrate weighted opponent schedules to curriculum.stages."
        )
    if cfg.self_play_pool_size not in {0, int(curriculum.snapshot.pool_size)}:
        raise ValueError(
            "self_play_pool_size is deprecated when curriculum.enabled is true; "
            "use curriculum.snapshot.pool_size."
        )
    if cfg.self_play_snapshot_interval not in {
        0,
        int(curriculum.snapshot.interval_updates),
    }:
        raise ValueError(
            "self_play_snapshot_interval is deprecated when curriculum.enabled is true; "
            "use curriculum.snapshot.interval_updates."
        )
    if float(cfg.self_play_latest_probability) != 0.5:
        raise ValueError(
            "self_play_latest_probability is deprecated when curriculum.enabled is true; "
            "use curriculum.stages[*].opponent_families.latest."
        )
    if not curriculum.stages:
        raise ValueError(
            "curriculum.stages must be non-empty when curriculum.enabled is true."
        )
    snapshot = curriculum.snapshot
    if snapshot.selection not in {"uniform", "recent_biased"}:
        raise ValueError(
            "curriculum.snapshot.selection must be 'uniform' or 'recent_biased'."
        )
    if snapshot.fallback != "latest":
        raise ValueError(
            "curriculum.snapshot.fallback currently supports only 'latest'."
        )
    seen_ids: set[str] = set()
    for index, stage in enumerate(curriculum.stages):
        if not isinstance(stage, dict):
            raise ValueError("curriculum.stages entries must be mappings.")
        stage_id = str(stage.get("id", "")).strip()
        if not stage_id:
            raise ValueError(f"curriculum.stages[{index}].id must be non-empty.")
        if stage_id in seen_ids:
            raise ValueError(f"curriculum.stages id {stage_id!r} is duplicated.")
        seen_ids.add(stage_id)
        if int(stage.get("min_updates", 0)) < 0:
            raise ValueError(
                f"curriculum.stages[{index}].min_updates must be non-negative."
            )
        if int(stage.get("cooldown_updates", 0)) < 0:
            raise ValueError(
                f"curriculum.stages[{index}].cooldown_updates must be non-negative."
            )
        weights = dict(stage.get("opponent_families", {}))
        if not weights:
            raise ValueError(
                f"curriculum.stages[{index}].opponent_families must be non-empty."
            )
        unknown = sorted(set(weights) - _CURRICULUM_FAMILIES)
        if unknown:
            raise ValueError(
                f"curriculum.stages[{index}].opponent_families contains unknown families: "
                f"{', '.join(unknown)}."
            )
        total = 0.0
        for family, raw_weight in weights.items():
            weight = float(raw_weight)
            if weight < 0.0 or weight == float("inf") or weight != weight:
                raise ValueError(
                    f"curriculum.stages[{index}].opponent_families.{family} must be finite and non-negative."
                )
            total += weight
        if total <= 0.0:
            raise ValueError(
                f"curriculum.stages[{index}].opponent_families must sum to > 0."
            )
        if float(weights.get("historical", 0.0)) > 0.0:
            if int(snapshot.pool_size) <= 0 or int(snapshot.interval_updates) <= 0:
                raise ValueError(
                    "curriculum historical opponents require snapshot.pool_size > 0 "
                    "and snapshot.interval_updates > 0."
                )
        promote_if = stage.get("promote_if")
        if promote_if:
            if not isinstance(promote_if, dict):
                raise ValueError(
                    f"curriculum.stages[{index}].promote_if must be a mapping."
                )
            metric = str(promote_if.get("metric", "")).strip()
            if metric not in CURRICULUM_PROMOTION_METRIC_NAMES:
                raise ValueError(
                    f"curriculum.stages[{index}].promote_if.metric must be one of "
                    f"{', '.join(sorted(CURRICULUM_PROMOTION_METRIC_NAMES))}."
                )
            if str(promote_if.get("op", ">=")).strip() not in {">=", ">", "<=", "<"}:
                raise ValueError(
                    f"curriculum.stages[{index}].promote_if.op is invalid."
                )
            if int(promote_if.get("window_updates", 1)) <= 0:
                raise ValueError(
                    f"curriculum.stages[{index}].promote_if.window_updates must be positive."
                )


def _validate_no_legacy_format_conflicts(cfg_raw: Any) -> None:
    """Reject ambiguous configs that define rollout/grouping fields in both old/new locations."""

    raw = OmegaConf.to_container(cfg_raw, resolve=False) if cfg_raw is not None else {}
    if not isinstance(raw, dict):
        return
    training_format = raw.get("training_format")
    ppo = raw.get("ppo")
    if not isinstance(training_format, dict) or not isinstance(ppo, dict):
        return

    if "rollout_groups" in training_format and "rollout_groups" in ppo:
        raise ValueError(
            "Conflicting rollout group definitions: use only training_format.rollout_groups; "
            "ppo.rollout_groups is deprecated and no longer supported."
        )
    if "phases" in training_format and "phases" in ppo:
        raise ValueError(
            "Conflicting phase definitions: use only training_format.phases; "
            "ppo.phases is deprecated and no longer supported."
        )
    if "num_envs_2p" in ppo or "num_envs_4p" in ppo:
        raise ValueError(
            "ppo.num_envs_2p/ppo.num_envs_4p are deprecated; configure per-format env counts "
            "via training_format.rollout_groups[*].num_envs only."
        )


def _validate_registered_update_metric_name(name: str, *, field_name: str) -> None:
    metric_name = str(name or "").strip()
    if not metric_name:
        raise ValueError(
            f"{field_name} must be a non-empty registered telemetry metric."
        )
    try:
        validate_scalar_update_metric_name(metric_name)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be a registered canonical scalar telemetry metric, got {metric_name!r}."
        ) from exc


def _parse_seed_set(raw: object) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if ".." in text:
            start_s, end_s = text.split("..", maxsplit=1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            return list(range(start, end + step, step))
        if "-" in text and text.count("-") == 1 and text.replace("-", "").isdigit():
            start_s, end_s = text.split("-", maxsplit=1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            return list(range(start, end + step, step))
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    if isinstance(raw, list | tuple | set):
        return [int(v) for v in raw]
    return []
