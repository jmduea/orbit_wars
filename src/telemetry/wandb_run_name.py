"""W&B run naming helpers for Hydra multirun and sweep jobs."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from src.config import TrainConfig
from src.config.rollout_allocation import rollout_player_counts, run_name_env_count

SWEPT_KEY_EXCLUDE_PREFIXES: tuple[str, ...] = (
    "seed",
    "run_name",
    "print_resolved_config",
    "resume_checkpoint",
    "heldout_eval_seed_set",
    "output.",
    "hydra.",
    "telemetry.wandb.",
)

SWEPT_KEY_EXCLUDE_KEYS: frozenset[str] = frozenset(
    {
        "seed",
        "run_name",
        "print_resolved_config",
        "resume_checkpoint",
    }
)

_SUFFIX_MAX_LENGTH = 64
_WANDB_RUN_NAME_MAX_LENGTH = 128

_KEY_SHORT_NAMES: dict[str, str] = {
    "training.lr": "lr",
    "training.learning_rate": "lr",
    "training.gamma": "gamma",
    "training.gae_lambda": "gae",
    "training.ent_coef": "ent",
    "training.rollout_steps": "rs",
    "training.num_envs": "env",
    "training.rollout_microbatch_envs": "rmb",
    "training.update_chunk_rows": "ucr",
    "training.total_updates": "u",
    "training": "tr",
}

def _hydra_job_num() -> int | None:
    try:
        from hydra.core.hydra_config import HydraConfig

        if not HydraConfig.initialized():
            return None
        job = getattr(HydraConfig.get(), "job", None)
        job_num = getattr(job, "num", None) if job is not None else None
        return None if job_num is None else int(job_num)
    except Exception:
        return None


def _run_name_component(value: str) -> str:
    component = value.strip().lower().replace(" ", "")
    return (
        "".join(char if char.isalnum() or char in "_." else "" for char in component)
        or "unknown"
    )


def _rollout_player_counts(cfg: TrainConfig) -> list[int]:
    return rollout_player_counts(cfg)


def _format_run_name_component(cfg: TrainConfig) -> str:
    player_counts = _rollout_player_counts(cfg)
    if len(player_counts) > 1:
        return "mix" + "p".join(str(count) for count in player_counts) + "p"
    return f"{player_counts[0]}p"


def _opponent_run_name_component(cfg: TrainConfig) -> str:
    if bool(cfg.opponents.self_play.enabled):
        return "selfplay"
    weights = cfg.opponents.mix.weights
    active_weights = {
        str(name): float(weight)
        for name, weight in weights.items()
        if float(weight) > 0.0
    }
    if active_weights:
        opponent = max(active_weights, key=lambda name: active_weights[name])
    else:
        opponent = str(cfg.opponents.dispatch)
    return _run_name_component(opponent)


def _run_name_env_count(cfg: TrainConfig) -> int:
    return run_name_env_count(cfg)


_CHOICE_VALUE_PREFIXES: tuple[str, ...] = (
    "mix_",
    "format_",
    "model_",
    "opponents_",
    "curriculum_",
    "reward_",
    "telemetry_",
)


def parse_override_entries(overrides: Sequence[str]) -> dict[str, str]:
    """Parse Hydra override strings into a flat key map."""

    parsed: dict[str, str] = {}
    for raw in overrides:
        if not isinstance(raw, str) or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def is_excluded_swept_key(key: str) -> bool:
    """Return whether a config key should never appear in sweep suffixes."""

    if key in SWEPT_KEY_EXCLUDE_KEYS:
        return True
    return any(
        key == prefix.rstrip(".") or key.startswith(prefix)
        for prefix in SWEPT_KEY_EXCLUDE_PREFIXES
    )


def sweep_varying_parameter_keys(parameters: Mapping[str, Any]) -> frozenset[str]:
    """Return sweep parameter keys that vary across multirun jobs."""

    varying: set[str] = set()
    for key, raw_spec in parameters.items():
        if not isinstance(key, str) or not isinstance(raw_spec, Mapping):
            continue
        spec = dict(raw_spec)
        values = spec.get("values")
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            if len(values) > 1:
                varying.add(key)
            continue
        if spec.get("distribution") is not None:
            varying.add(key)
    return frozenset(varying)


def fixed_baseline_overrides(parameters: Mapping[str, Any]) -> dict[str, str]:
    """Extract fixed ``value`` entries from a W&B sweep parameter block."""

    baseline: dict[str, str] = {}
    for key, raw_spec in parameters.items():
        if not isinstance(key, str) or not isinstance(raw_spec, Mapping):
            continue
        if "value" not in raw_spec:
            continue
        baseline[key] = _stringify_override_value(raw_spec["value"])
    return baseline


def override_diff_keys(
    job: Mapping[str, str],
    baseline: Mapping[str, str],
) -> frozenset[str]:
    """Return keys whose override values differ from the sweep baseline."""

    keys = set(job) | set(baseline)
    return frozenset(
        key
        for key in keys
        if not is_excluded_swept_key(key) and job.get(key) != baseline.get(key)
    )


def detect_swept_keys(
    *,
    job_overrides: Mapping[str, str],
    sweep_parameters: Mapping[str, Any],
) -> frozenset[str]:
    """Compute swept keys as override diff intersecting varying sweep params."""

    varying = sweep_varying_parameter_keys(sweep_parameters)
    diff = override_diff_keys(job_overrides, fixed_baseline_overrides(sweep_parameters))
    return frozenset(key for key in diff if key in varying)


def is_hydra_multirun() -> bool:
    """Return whether the active Hydra job is part of a multirun sweep."""

    try:
        from hydra.core.hydra_config import HydraConfig
        from hydra.types import RunMode
    except Exception:
        return False
    if not HydraConfig.initialized():
        return False
    return HydraConfig.get().mode == RunMode.MULTIRUN


def is_wandb_sweep_job() -> bool:
    """Return whether the process is running under a W&B sweep agent."""

    return bool(os.environ.get("WANDB_SWEEP_ID") or os.environ.get("WANDB_SWEEP_YAML"))


def should_apply_sweep_run_rename(cfg: TrainConfig) -> bool:
    """Return whether post-init W&B rename should run for this job."""

    if not cfg.telemetry.wandb.rename_from_swept_params:
        return False
    return is_hydra_multirun() or is_wandb_sweep_job()


def _hydra_job_overrides() -> list[str]:
    try:
        from hydra.core.hydra_config import HydraConfig
    except Exception:
        return []
    if not HydraConfig.initialized():
        return []
    return list(HydraConfig.get().overrides.task)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_sweep_parameters_from_path(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"sweep YAML must be a mapping: {path}")
    parameters = payload.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError(f"sweep YAML must contain parameters: {path}")
    return dict(parameters)


def _compose_wandb_sweep_parameters(recipe: str) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    config_dir = _repo_root() / "conf"
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
            sweep = OmegaConf.to_container(
                compose(
                    config_name="sweep_gen",
                    overrides=[f"wandb_sweep={recipe}"],
                ),
                resolve=True,
            )
    finally:
        GlobalHydra.instance().clear()
    if not isinstance(sweep, dict):
        raise ValueError(f"composed wandb_sweep={recipe!r} must be a mapping")
    parameters = sweep.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError(f"wandb_sweep={recipe!r} must expose parameters")
    return dict(parameters)


def load_active_sweep_parameters() -> dict[str, Any] | None:
    """Load sweep parameters from env or the active Hydra ``wandb_sweep`` choice."""

    env_path = os.environ.get("WANDB_SWEEP_YAML")
    if env_path:
        path = Path(env_path)
        if not path.is_absolute():
            path = _repo_root() / path
        if path.is_file():
            return _load_sweep_parameters_from_path(path)

    try:
        from hydra.core.hydra_config import HydraConfig
    except Exception:
        return None
    if not HydraConfig.initialized():
        return None
    recipe = HydraConfig.get().runtime.choices.get("wandb_sweep")
    if not recipe:
        return None
    return _compose_wandb_sweep_parameters(str(recipe))


def _compact_float(value: float) -> str:
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    use_scientific = 0 < abs_value < 1e-2 or abs_value >= 1e3
    text = f"{value:.2e}" if use_scientific else f"{value:.6g}"
    if "e" in text or "E" in text:
        mantissa, exponent = text.lower().split("e", 1)
        mantissa = mantissa.rstrip("0").rstrip(".") or "0"
        exponent = exponent.lstrip("+")
        if exponent.startswith("-"):
            exponent = f"-{exponent[1:].lstrip('0') or '0'}"
        else:
            exponent = exponent.lstrip("0") or "0"
        return f"{mantissa}e{exponent}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _strip_choice_prefix(value: str) -> str:
    lowered = value.strip().lower()
    for prefix in _CHOICE_VALUE_PREFIXES:
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix) :]
            break
    compact = _run_name_component(lowered)
    return compact.replace("_", "")


def _stringify_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(item) for item in value) + "]"
    return str(value)


def _short_param_key(key: str) -> str:
    if key in _KEY_SHORT_NAMES:
        return _KEY_SHORT_NAMES[key]
    if "." in key:
        return key.rsplit(".", 1)[-1]
    return key


def _compact_param_token(key: str, value: Any) -> str:
    short_key = _short_param_key(key)
    if isinstance(value, bool):
        compact_value = "1" if value else "0"
    elif isinstance(value, float):
        compact_value = _compact_float(value)
    elif isinstance(value, int):
        compact_value = str(value)
    else:
        compact_value = _strip_choice_prefix(str(value))
    return f"{short_key}{compact_value}"


def build_sweep_run_suffix(
    swept_params: Mapping[str, Any],
    *,
    max_length: int = _SUFFIX_MAX_LENGTH,
) -> str:
    """Build a compact deterministic suffix for swept scalar parameters.

    Args:
        swept_params: Swept key/value pairs (typically from override strings).
        max_length: Maximum suffix length before truncation.

    Returns:
        Compact suffix such as ``env128_rs250`` (empty when no params).
    """

    if not swept_params:
        return ""
    tokens = [
        _compact_param_token(key, value)
        for key, value in sorted(swept_params.items(), key=lambda item: item[0])
    ]
    suffix = "_".join(tokens)
    if len(suffix) <= max_length:
        return suffix
    truncated = suffix[:max_length]
    if truncated.endswith("_"):
        truncated = truncated.rstrip("_")
    return truncated


def _swept_matches(swept_keys: frozenset[str], prefixes: tuple[str, ...]) -> bool:
    for key in swept_keys:
        bare = key.split(".", 1)[0]
        for prefix in prefixes:
            stem = prefix.rstrip(".")
            if key == stem or key.startswith(prefix) or bare == stem:
                return True
    return False


def compose_run_name_prefix(cfg: TrainConfig, swept_keys: frozenset[str]) -> str:
    """Build the non-swept prefix portion of a sweep run display name."""

    from datetime import datetime, timezone

    parts: list[str] = []
    if not _swept_matches(swept_keys, ("model.",)):
        parts.append(_run_name_component(str(cfg.model.architecture)))
    if not _swept_matches(swept_keys, ("training.", "training.format_weights", "task.player_count")):
        parts.append(_format_run_name_component(cfg))
    if not _swept_matches(swept_keys, ("opponents.",)):
        parts.append(_opponent_run_name_component(cfg))
    if not _swept_matches(swept_keys, ("training.total_updates",)):
        parts.append(f"u{int(cfg.training.total_updates)}")
    if not _swept_matches(
        swept_keys,
        ("training.num_envs", "training.format_weights"),
    ):
        parts.append(f"env{_run_name_env_count(cfg)}")
    if not _swept_matches(swept_keys, ("seed",)):
        parts.append(f"s{int(cfg.seed)}")
    job_num = _hydra_job_num()
    if job_num is not None:
        parts.append(f"job{job_num:04d}")
    parts.append(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    return "-".join(parts)


def _coerce_swept_values(
    cfg: TrainConfig,
    swept_keys: frozenset[str],
    job_overrides: Mapping[str, str],
) -> dict[str, Any]:
    from dataclasses import asdict

    flat = _flatten_config(asdict(cfg))
    values: dict[str, Any] = {}
    for key in sorted(swept_keys):
        if key in job_overrides:
            values[key] = job_overrides[key]
            continue
        if key in flat:
            values[key] = flat[key]
    return values


def _flatten_config(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten_config(value, prefix=full_key))
        else:
            flat[full_key] = value
    return flat


def compose_sweep_display_name(
    cfg: TrainConfig,
    *,
    swept_keys: frozenset[str],
    job_overrides: Mapping[str, str] | None = None,
) -> str:
    """Compose ``{prefix}-{suffix}`` for a sweep/multirun job."""

    overrides = dict(job_overrides or parse_override_entries(_hydra_job_overrides()))
    prefix = compose_run_name_prefix(cfg, swept_keys)
    suffix = build_sweep_run_suffix(_coerce_swept_values(cfg, swept_keys, overrides))
    if not suffix:
        return prefix[:_WANDB_RUN_NAME_MAX_LENGTH]
    combined = f"{prefix}-{suffix}"
    if len(combined) <= _WANDB_RUN_NAME_MAX_LENGTH:
        return combined
    return combined[:_WANDB_RUN_NAME_MAX_LENGTH].rstrip("-")


def resolve_sweep_display_name(
    cfg: TrainConfig,
    *,
    job_overrides: Mapping[str, str] | None = None,
    sweep_parameters: Mapping[str, Any] | None = None,
) -> str | None:
    """Return a sweep display name or ``None`` when rename should not apply."""

    if not should_apply_sweep_run_rename(cfg):
        return None
    parameters = sweep_parameters if sweep_parameters is not None else load_active_sweep_parameters()
    if not parameters:
        return None
    overrides = dict(job_overrides or parse_override_entries(_hydra_job_overrides()))
    swept_keys = detect_swept_keys(
        job_overrides=overrides,
        sweep_parameters=parameters,
    )
    if not swept_keys:
        return None
    return compose_sweep_display_name(
        cfg,
        swept_keys=swept_keys,
        job_overrides=overrides,
    )


def apply_post_init_run_rename(run: Any, cfg: TrainConfig) -> None:
    """Rename an initialized W&B run to emphasize swept parameters."""

    display_name = resolve_sweep_display_name(cfg)
    if not display_name:
        return
    run.name = display_name
