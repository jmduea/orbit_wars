"""Load ``PreflightGateSpec`` train/evaluation config from ``conf/benchmark/gates/`` YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.jax.preflight_calibration import (
    PREFLIGHT_TRAIN_BASE,
    WINDOW_UPDATES,
    default_calibration_json_path,
    load_thresholds,
)
from src.jax.preflight_profiles import (
    default_profiles_path,
    ppo_overrides_for_model,
)
from src.jax.preflight import GATE_ORDER, PreflightGateSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
GATES_DIR = REPO_ROOT / "conf" / "benchmark" / "gates"

PLANET_FLOW_TRAIN_BASE: tuple[str, ...] = (
    "curriculum=off",
    "telemetry.metric_groups.action_decision=true",
    *PREFLIGHT_TRAIN_BASE,
    "artifacts=planet_flow_proof",
    "artifacts.artifact_pipeline.enabled=true",
)


def gate_yaml_paths() -> list[Path]:
    if not GATES_DIR.is_dir():
        return []
    return sorted(GATES_DIR.glob("*.yaml"))


def load_gate_yaml(gate_id: str) -> dict[str, Any]:
    path = GATES_DIR / f"{gate_id}.yaml"
    if not path.is_file():
        known = [item.stem for item in gate_yaml_paths()]
        raise FileNotFoundError(
            f"Unknown gate recipe {gate_id!r}. Known YAML gates: {', '.join(known) or '(none)'}"
        )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Gate recipe must be a mapping: {path}")
    recipe_gate_id = str(payload.get("gate_id") or path.stem)
    if recipe_gate_id != gate_id:
        raise ValueError(
            f"Gate recipe gate_id mismatch in {path}: expected {gate_id!r}, got {recipe_gate_id!r}"
        )
    payload["path"] = str(path.relative_to(REPO_ROOT))
    return payload


def _threshold_section(
    thresholds: dict[str, object],
    key: str,
) -> dict[str, object]:
    section = thresholds.get(key, {})
    return section if isinstance(section, dict) else {}


def _float_threshold(section: dict[str, object], name: str, default: float) -> float:
    raw = section.get(name, default)
    return float(raw) if raw is not None else default


def _optional_float(section: dict[str, object], name: str) -> float | None:
    if name not in section:
        return None
    raw = section[name]
    if raw is None:
        return None
    return float(raw)


def _train_base(name: str | None) -> tuple[str, ...]:
    if name == "planet_flow":
        return PLANET_FLOW_TRAIN_BASE
    return PREFLIGHT_TRAIN_BASE


def _resolve_variant_key(model: str) -> str:
    if model == "planet_flow_target_heatmap":
        return "planet_flow_target_heatmap"
    return "default"


def _resolved_model(
    *,
    cli_model: str,
    variant: dict[str, Any],
) -> str:
    fixed = variant.get("model")
    if isinstance(fixed, str) and fixed.strip():
        return fixed.strip()
    if bool(variant.get("model_from_cli", False)):
        return cli_model
    return cli_model


def _evaluation_overrides(
    evaluation: dict[str, Any],
    *,
    thresholds: dict[str, object],
    thresholds_key: str,
) -> dict[str, Any]:
    section = _threshold_section(thresholds, thresholds_key)
    resolved = dict(evaluation)
    if resolved.get("min_win_rate_delta") == "from_thresholds":
        resolved["min_win_rate_delta"] = _optional_float(section, "min_win_rate_delta")
        if resolved["min_win_rate_delta"] is None:
            resolved["min_win_rate_delta"] = 0.08
    if resolved.get("max_post_mask_unreachable_demand_rate") == "from_thresholds":
        resolved["max_post_mask_unreachable_demand_rate"] = _optional_float(
            section, "max_post_mask_unreachable_demand_rate"
        )
    if "window_updates" not in resolved:
        resolved["window_updates"] = int(section.get("window_updates", WINDOW_UPDATES))
    if "max_approx_kl" not in resolved:
        resolved["max_approx_kl"] = _float_threshold(section, "max_approx_kl", 0.15)
    if "min_entropy" not in resolved:
        resolved["min_entropy"] = _float_threshold(section, "min_entropy", 1.0e-4)
    needs_key = resolved.get("needs_calibration_key")
    if isinstance(needs_key, str):
        calibrated = isinstance(thresholds.get(needs_key), dict)
        if not calibrated:
            resolved["needs_calibration_reason"] = (
                "needs-calibration: Planet Flow pressure-action thresholds are not "
                "present in preflight calibration"
            )
    return resolved


def build_gate_spec(
    gate_id: str,
    *,
    model: str,
    thresholds_path: Path | None = None,
    profiles_path: Path | None = None,
    repo_root: Path | None = None,
) -> PreflightGateSpec:
    recipe = load_gate_yaml(gate_id)
    root = repo_root or REPO_ROOT
    thresholds = load_thresholds(thresholds_path or default_calibration_json_path(root))
    variant_key = _resolve_variant_key(model)
    train_section = recipe.get("train")
    if not isinstance(train_section, dict):
        raise ValueError(f"Gate {gate_id!r} missing train section in YAML")
    variant = train_section.get(variant_key)
    if not isinstance(variant, dict):
        raise ValueError(
            f"Gate {gate_id!r} missing train.{variant_key} variant in YAML"
        )

    thresholds_key = str(
        variant.get("thresholds_key") or recipe.get("thresholds_key") or "learning_signal"
    )
    evaluation_raw = variant.get("evaluation")
    evaluation = evaluation_raw if isinstance(evaluation_raw, dict) else {}
    eval_resolved = _evaluation_overrides(
        evaluation,
        thresholds=thresholds,
        thresholds_key=thresholds_key,
    )

    resolved_model = _resolved_model(cli_model=model, variant=variant)
    raw_overrides = variant.get("train_overrides")
    if not isinstance(raw_overrides, list) or not raw_overrides:
        raise ValueError(f"Gate {gate_id!r} train.{variant_key}.train_overrides must be a list")
    train_overrides: list[str] = [f"model={resolved_model}", *_train_base(variant.get("train_base"))]
    train_overrides.extend(str(item) for item in raw_overrides)

    if bool(variant.get("apply_ppo_profile", False)):
        profile_path = profiles_path or default_profiles_path(root)
        train_overrides.extend(
            ppo_overrides_for_model(
                resolved_model,
                profiles_path=profile_path,
                repo_root=root,
            )
        )

    min_win_rate_delta = eval_resolved.get("min_win_rate_delta")
    if min_win_rate_delta is not None:
        min_win_rate_delta = float(min_win_rate_delta)

    max_unreachable = eval_resolved.get("max_post_mask_unreachable_demand_rate")
    if max_unreachable is not None:
        max_unreachable = float(max_unreachable)

    needs_calibration_reason = eval_resolved.get("needs_calibration_reason")
    if needs_calibration_reason is not None:
        needs_calibration_reason = str(needs_calibration_reason)

    return PreflightGateSpec(
        gate_id=gate_id,
        train_overrides=tuple(train_overrides),
        min_win_rate_delta=min_win_rate_delta,
        window_updates=int(eval_resolved.get("window_updates", WINDOW_UPDATES)),
        require_curriculum_promotion=bool(
            eval_resolved.get("require_curriculum_promotion", False)
        ),
        max_approx_kl=float(eval_resolved.get("max_approx_kl", 0.15)),
        min_entropy=float(eval_resolved.get("min_entropy", 1.0e-4)),
        max_post_mask_unreachable_demand_rate=max_unreachable,
        needs_calibration_reason=needs_calibration_reason,
        require_planet_flow_control_metrics=bool(
            eval_resolved.get("require_planet_flow_control_metrics", False)
        ),
    )


def gate_specs(
    model: str,
    *,
    thresholds_path: Path | None = None,
    profiles_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, PreflightGateSpec]:
    return {
        gate_id: build_gate_spec(
            gate_id,
            model=model,
            thresholds_path=thresholds_path,
            profiles_path=profiles_path,
            repo_root=repo_root,
        )
        for gate_id in GATE_ORDER
    }
