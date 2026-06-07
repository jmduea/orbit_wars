from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from src.artifacts.checkpoint_compat import (
    action_layout_version_for_pointer_decoder,
    pointer_decoder_for_model,
)
from src.config import TrainConfig
from src.config.rollout_allocation import rollout_player_counts, run_name_env_count


@dataclass(slots=True)
class RunContext:
    run_id: str
    campaign_slug: str
    run_dir: Path
    manifest_path: Path
    campaign_dir: Path
    campaign_manifest_path: Path
    logs_dir: Path
    log_path: Path
    debug_log_path: Path
    checkpoints_dir: Path
    queue_dir: Path
    evaluations_dir: Path
    wandb_dir: Path
    wandb_artifact_dir: Path
    wandb_data_dir: Path
    indexes_dir: Path
    retention_class: str
    model_compatibility_family: str


def _hydra_runtime_output_dir() -> Path | None:
    try:
        from hydra.core.hydra_config import HydraConfig
    except Exception:
        return None
    if not HydraConfig.initialized():
        return None
    runtime = HydraConfig.get().runtime
    output_dir = getattr(runtime, "output_dir", None)
    if not output_dir:
        return None
    return Path(str(output_dir))


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


def compose_run_name(cfg: TrainConfig) -> str:
    """Compose a display run name from the fields most likely to differ."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parts = [
        _run_name_component(str(cfg.model.architecture)),
        _format_run_name_component(cfg),
        _opponent_run_name_component(cfg),
        f"u{int(cfg.training.total_updates)}",
        f"env{run_name_env_count(cfg)}",
        f"s{int(cfg.seed)}",
    ]
    job_num = _hydra_job_num()
    if job_num is not None:
        parts.append(f"job{job_num:04d}")
    parts.append(timestamp)
    return "-".join(parts)


def _format_run_name_component(cfg: TrainConfig) -> str:
    player_counts = rollout_player_counts(cfg)
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
        opponent = max(active_weights, key=active_weights.get)
    else:
        opponent = str(cfg.opponents.mode.opponent)
    return _run_name_component(opponent)


def _run_name_component(value: str) -> str:
    component = value.strip().lower().replace(" ", "")
    return (
        "".join(char if char.isalnum() or char in "_." else "" for char in component)
        or "unknown"
    )


def resolve_run_paths(cfg: TrainConfig) -> tuple[TrainConfig, RunContext]:
    """Resolve the canonical run context rooted at Hydra's output directory."""

    output_dir = _hydra_runtime_output_dir()
    output_root = Path(cfg.output.root)
    campaign_slug = str(cfg.output.campaign)
    run_id = _effective_run_id(cfg)
    run_name = compose_run_name(cfg)
    if output_dir is not None:
        run_dir = output_dir
        run_id = run_dir.name
        campaign_dir = (
            run_dir.parents[1] if run_dir.parent.name == "runs" else run_dir.parent
        )
    else:
        campaign_dir = output_root / "campaigns" / campaign_slug
        run_dir = campaign_dir / "runs" / run_id

    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    queue_dir = run_dir / cfg.artifacts.artifact_pipeline.queue_dir
    evaluations_dir = run_dir / cfg.artifacts.artifact_pipeline.result_dir
    indexes_dir = output_root / cfg.output.indexes_dir
    cache_dir = output_root / cfg.output.cache_dir
    context = RunContext(
        run_id=run_id,
        campaign_slug=campaign_slug,
        run_dir=run_dir,
        manifest_path=run_dir / "manifest.json",
        campaign_dir=campaign_dir,
        campaign_manifest_path=campaign_dir / "campaign_manifest.json",
        logs_dir=logs_dir,
        log_path=logs_dir / f"{run_name}_jax.jsonl",
        debug_log_path=logs_dir / f"{run_name}_debug.jsonl",
        checkpoints_dir=checkpoints_dir,
        queue_dir=queue_dir,
        evaluations_dir=evaluations_dir,
        wandb_dir=run_dir / cfg.output.wandb_dir,
        wandb_artifact_dir=_cache_path(
            output_root, Path(cfg.output.cache_dir), cfg.output.wandb_artifact_dir
        ),
        wandb_data_dir=_cache_path(
            output_root, Path(cfg.output.cache_dir), cfg.output.wandb_data_dir
        ),
        indexes_dir=indexes_dir,
        retention_class=str(cfg.output.retention_class),
        model_compatibility_family=str(cfg.model.architecture),
    )
    artifacts = replace(cfg.artifacts, save_dir=str(checkpoints_dir))
    output = replace(cfg.output, run_id=run_id)
    return replace(cfg, run_name=run_name, artifacts=artifacts, output=output), context


def _merge_campaign_manifest_on_run_start(
    cfg: TrainConfig,
    context: RunContext,
    created_at: str,
) -> dict[str, object]:
    """Seed or merge campaign promotion fields on run start."""

    import warnings

    promotion = cfg.artifacts.promotion
    existing: dict[str, object] = {}
    if context.campaign_manifest_path.exists():
        raw = json.loads(context.campaign_manifest_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            existing = raw

    metric_name = str(promotion.metric_name or "").strip()
    metric_mode = str(promotion.metric_mode or "max").strip().lower()
    frozen_name = str(existing.get("promotion_metric_name", "")).strip()
    frozen_mode = str(existing.get("promotion_metric_mode", "max")).strip().lower()

    if frozen_name:
        if metric_name and metric_name != frozen_name:
            warnings.warn(
                f"Campaign {context.campaign_slug!r} promotion metric "
                f"{frozen_name!r} differs from run config {metric_name!r}.",
                stacklevel=2,
            )
        metric_name = frozen_name
        metric_mode = frozen_mode or metric_mode
    elif promotion.enabled and metric_name:
        existing["promotion_metric_name"] = metric_name
        existing["promotion_metric_mode"] = metric_mode

    return {
        "campaign": context.campaign_slug,
        "campaign_dir": str(context.campaign_dir),
        "updated_at": created_at,
        "default_retention_class": context.retention_class,
        "promotion_metric_name": metric_name or existing.get("promotion_metric_name"),
        "promotion_metric_mode": metric_mode
        or existing.get("promotion_metric_mode", "max"),
        "current_best_value": existing.get("current_best_value"),
        "current_best_run_id": existing.get("current_best_run_id"),
    }


def write_run_manifests(
    cfg: TrainConfig, context: RunContext, metadata: Mapping[str, object]
) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    pointer_decoder = pointer_decoder_for_model(cfg.model)
    run_manifest = {
        "run_id": context.run_id,
        "campaign": context.campaign_slug,
        "run_name": cfg.run_name,
        "job_type": metadata.get("job_type", "train"),
        "model_compatibility_family": context.model_compatibility_family,
        "pointer_decoder": pointer_decoder,
        "action_layout_version": action_layout_version_for_pointer_decoder(
            pointer_decoder
        ),
        "seed": int(cfg.seed),
        "retention_class": context.retention_class,
        "hydra_output_dir": str(context.run_dir),
        "resolved_config_path": str(context.run_dir / ".hydra" / "config.yaml"),
        "hydra_overrides_path": str(context.run_dir / ".hydra" / "overrides.yaml"),
        "paths": {
            "logs_dir": str(context.logs_dir),
            "log_path": str(context.log_path),
            "debug_log_path": str(context.debug_log_path),
            "checkpoints_dir": str(context.checkpoints_dir),
            "queue_dir": str(context.queue_dir),
            "evaluations_dir": str(context.evaluations_dir),
            "wandb_dir": str(context.wandb_dir),
            "wandb_artifact_dir": str(context.wandb_artifact_dir),
            "wandb_data_dir": str(context.wandb_data_dir),
        },
        "wandb": {
            "project": cfg.telemetry.wandb.project,
            "entity": cfg.telemetry.wandb.entity,
            "group": cfg.telemetry.wandb.group,
            "tags": list(cfg.telemetry.wandb.tags),
        },
        "git": _git_identity(),
        "created_at": created_at,
        "produced_artifacts": [],
        **dict(metadata),
    }
    campaign_manifest = _merge_campaign_manifest_on_run_start(cfg, context, created_at)
    atomic_write_json(context.manifest_path, run_manifest)
    atomic_write_json(context.campaign_manifest_path, campaign_manifest)
    append_jsonl_atomic(
        context.indexes_dir / "runs.jsonl",
        {
            "run_id": context.run_id,
            "campaign": context.campaign_slug,
            "run_dir": str(context.run_dir),
            "created_at": created_at,
        },
    )


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)
        _fsync_dir(path.parent)
    finally:
        tmp_path.unlink(missing_ok=True)


def append_produced_artifact(
    manifest_path: Path,
    entry: Mapping[str, object],
) -> None:
    """Append one artifact record to the run manifest when the path is new."""

    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("produced_artifacts")
    if not isinstance(artifacts, list):
        artifacts = []
    entry_path = str(entry.get("path", "")).strip()
    if entry_path and any(
        isinstance(item, dict) and str(item.get("path", "")).strip() == entry_path
        for item in artifacts
    ):
        return
    artifacts.append(dict(entry))
    manifest["produced_artifacts"] = artifacts
    atomic_write_json(manifest_path, manifest)


def append_jsonl_atomic(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(dict(record), sort_keys=True) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _effective_run_id(cfg: TrainConfig) -> str:
    run_id = str(cfg.output.run_id).strip()
    if run_id and "${" not in run_id:
        return run_id
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-s{int(cfg.seed)}-{uuid.uuid4().hex[:8]}"


def _cache_path(output_root: Path, cache_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.parts and path.parts[0] == cache_dir.name:
        return output_root / path
    return output_root / cache_dir / path


def _git_identity() -> dict[str, object]:
    repo = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo, text=True
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _fsync_dir(path: Path) -> None:
    if os.name != "posix":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
