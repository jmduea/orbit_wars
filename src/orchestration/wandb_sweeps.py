from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .population import ShortlistRow, rank_shortlist


def load_sweep_config(path: Path) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def resolve_standalone_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Pick deterministic Hydra values from W&B sweep parameter specs.

    Standalone Kaggle workers do not call ``wandb.agent``; they need a fixed
    candidate without sampling. Fixed ``value`` entries win; ``values`` lists
    use the first entry; log-uniform specs use ``min`` for reproducibility.
    """

    resolved: dict[str, Any] = {}
    for key, raw_spec in parameters.items():
        if not isinstance(key, str) or not isinstance(raw_spec, Mapping):
            continue
        spec = dict(raw_spec)
        if "value" in spec:
            resolved[key] = spec["value"]
            continue
        values = spec.get("values")
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            if values:
                resolved[key] = values[0]
            continue
        if "min" in spec:
            resolved[key] = spec["min"]
    return resolved


def load_standalone_config(path: Path) -> dict[str, Any]:
    """Load fixed Hydra overrides from a packaged W&B sweep YAML."""

    sweep = load_sweep_config(path)
    parameters = sweep.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError(f"sweep YAML must contain parameters: {path}")
    return resolve_standalone_parameters(parameters)


def resolve_wandb_group_from_sweep(
    sweep: Mapping[str, Any],
    *,
    sweep_yaml_path: Path | None = None,
) -> str:
    """Resolve W&B group: sweep fixed group → output.campaign → YAML stem."""

    parameters = sweep.get("parameters")
    if isinstance(parameters, Mapping):
        for key in ("telemetry.wandb.group", "output.campaign"):
            spec = parameters.get(key)
            if isinstance(spec, Mapping) and spec.get("value") is not None:
                return str(spec["value"])
    if sweep_yaml_path is not None:
        return sweep_yaml_path.stem
    return "default"


def add_population_metadata(
    sweep: Mapping[str, Any],
    *,
    group: str,
    tags: Sequence[str],
) -> dict[str, Any]:
    """Return a W&B sweep config with required population metadata."""

    result = json.loads(json.dumps(dict(sweep)))
    parameters = result.setdefault("parameters", {})
    parameters.setdefault("telemetry.wandb.group", {"value": group})
    tag_spec = parameters.setdefault("telemetry.wandb.tags", {"value": []})
    existing = list(tag_spec.get("value", []))
    tag_spec["value"] = sorted({*existing, *tags})
    return result


def create_sweep(
    sweep: Mapping[str, Any],
    *,
    project: str,
    entity: str | None = None,
) -> str:
    """Create a W&B sweep and return its ID."""

    import wandb  # type: ignore

    return str(wandb.sweep(dict(sweep), project=project, entity=entity))


def rows_from_wandb_runs(runs: Sequence[Any]) -> list[ShortlistRow]:
    rows: list[ShortlistRow] = []
    for run in runs:
        summary = getattr(run, "summary", {}) or {}
        config = getattr(run, "config", {}) or {}
        artifacts = _checkpoint_artifacts(run)
        checkpoint = artifacts[0] if artifacts else {}
        rows.append(
            ShortlistRow(
                run_id=str(getattr(run, "id", getattr(run, "name", "unknown"))),
                name=str(getattr(run, "name", "unknown")),
                state=str(getattr(run, "state", "unknown")),
                checkpoint_artifact=_optional_str(checkpoint.get("name")),
                checkpoint_artifact_version=_optional_str(checkpoint.get("version")),
                checkpoint_artifact_aliases=tuple(
                    str(alias) for alias in checkpoint.get("aliases", ())
                ),
                metrics={
                    key: float(summary[key])
                    for key in (
                        "episode_reward_mean",
                        "samples_per_sec",
                        "ppo_samples_per_sec",
                    )
                    if key in summary and summary[key] is not None
                },
                config=dict(config),
            )
        )
    return rows


def shortlist_from_api(
    *,
    project: str,
    entity: str | None,
    sweep_id: str,
    limit: int = 10,
) -> list[ShortlistRow]:
    import wandb  # type: ignore

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"sweep": sweep_id})
    return rank_shortlist(rows_from_wandb_runs(list(runs)), limit=limit)


def _checkpoint_artifacts(run: Any) -> list[dict[str, object]]:
    try:
        logged = run.logged_artifacts()
    except Exception:
        return []
    rows: list[dict[str, object]] = []
    for artifact in logged:
        artifact_type = str(getattr(artifact, "type", ""))
        if artifact_type == "checkpoint":
            name = str(getattr(artifact, "name", "checkpoint"))
            rows.append(
                {
                    "name": name,
                    "version": _artifact_version(artifact, name),
                    "aliases": _artifact_aliases(artifact),
                    "update": _artifact_update(artifact),
                }
            )
    return sorted(rows, key=_checkpoint_sort_key, reverse=True)


def _artifact_aliases(artifact: Any) -> tuple[str, ...]:
    aliases = getattr(artifact, "aliases", ()) or ()
    result: list[str] = []
    for alias in aliases:
        result.append(str(getattr(alias, "alias", alias)))
    return tuple(sorted(result))


def _artifact_version(artifact: Any, name: str) -> str | None:
    version = getattr(artifact, "version", None)
    if version:
        return str(version)
    if ":" in name:
        return name.rsplit(":", 1)[1]
    return None


def _artifact_update(artifact: Any) -> int:
    metadata = getattr(artifact, "metadata", {}) or {}
    try:
        return int(metadata.get("update", -1))
    except (TypeError, ValueError, AttributeError):
        return -1


def _checkpoint_sort_key(row: Mapping[str, object]) -> tuple[int, int, int]:
    aliases = {str(alias) for alias in row.get("aliases", ())}
    version = str(row.get("version") or "")
    version_number = -1
    if version.startswith("v"):
        try:
            version_number = int(version[1:])
        except ValueError:
            pass
    alias_rank = 0
    if "best" in aliases or "promoted" in aliases:
        alias_rank = 2
    elif "latest" in aliases:
        alias_rank = 1
    return (
        alias_rank,
        int(row.get("update", -1)),
        version_number,
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
