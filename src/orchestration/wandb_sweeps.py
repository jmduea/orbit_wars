from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .population import ShortlistRow, rank_shortlist


def load_sweep_config(path: Path) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


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
        rows.append(
            ShortlistRow(
                run_id=str(getattr(run, "id", getattr(run, "name", "unknown"))),
                name=str(getattr(run, "name", "unknown")),
                state=str(getattr(run, "state", "unknown")),
                checkpoint_artifact=artifacts[0] if artifacts else None,
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


def _checkpoint_artifacts(run: Any) -> list[str]:
    try:
        logged = run.logged_artifacts()
    except Exception:
        return []
    names: list[str] = []
    for artifact in logged:
        artifact_type = str(getattr(artifact, "type", ""))
        if artifact_type == "checkpoint":
            names.append(str(getattr(artifact, "name", "checkpoint")))
    return names
