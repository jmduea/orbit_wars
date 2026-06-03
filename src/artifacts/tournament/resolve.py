"""Resolve tournament agents from checkpoints, shortlist JSON, and promotion."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.artifacts.checkpoint_compat import (
    checkpoint_feature_metadata,
    feature_metadata,
    load_checkpoint_payload,
    validate_checkpoint_config_compatibility,
)
from src.artifacts.promotion import resolve_from_promoted
from src.artifacts.run_paths import RunContext, _cache_path
from src.artifacts.tournament.runner import build_baseline_agent, build_checkpoint_agent
from src.config import TrainConfig

from .types import AgentEntry

DEFAULT_BOOTSTRAP_INCUMBENT = "nearest_sniper"

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ShortlistResolveResult:
    """Outcome of resolving shortlist rows to local checkpoint agents."""

    agents: tuple[AgentEntry, ...]
    skipped: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


def load_train_config_from_checkpoint(checkpoint_path: Path) -> TrainConfig:
    checkpoint = load_checkpoint_payload(checkpoint_path)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint:
        raise ValueError(f"checkpoint does not contain config: {checkpoint_path}")
    validate_checkpoint_config_compatibility(
        checkpoint, checkpoint_path=checkpoint_path
    )
    cfg = checkpoint["config"]
    if not isinstance(cfg, TrainConfig):
        raise TypeError(
            f"checkpoint config must be TrainConfig, got {type(cfg)!r}: {checkpoint_path}"
        )
    return cfg


def agent_from_baseline(
    name: str,
    *,
    agent_id: str = "incumbent",
) -> AgentEntry:
    """Build a scripted baseline agent for unified tournament Stage 2."""

    normalized = name.strip().lower()
    return AgentEntry(
        agent_id=agent_id,
        checkpoint_path=Path(f"scripted:{normalized}"),
        cfg=TrainConfig(),
        act_fn=build_baseline_agent(name),
    )


def agent_from_checkpoint(
    checkpoint_path: Path,
    *,
    agent_id: str | None = None,
) -> AgentEntry:
    agent_id = agent_id or checkpoint_path.stem
    resolved = checkpoint_path.resolve()
    cfg = load_train_config_from_checkpoint(resolved)
    submission_agent = build_checkpoint_agent(cfg, resolved)
    return AgentEntry(
        agent_id=agent_id,
        checkpoint_path=resolved,
        cfg=cfg,
        act_fn=submission_agent,
    )


def run_context_for_agent(
    agent: AgentEntry,
    *,
    campaign: str | None = None,
    output_root: str | None = None,
) -> RunContext:
    """Build run context from the checkpoint's on-disk run directory."""

    cfg = agent.cfg
    campaign_slug = str(campaign or cfg.output.campaign)
    root = Path(output_root or cfg.output.root)
    checkpoint_path = agent.checkpoint_path.resolve()

    if checkpoint_path.parent.name == "checkpoints":
        run_dir = checkpoint_path.parent.parent
        run_id = run_dir.name
        campaign_dir = (
            run_dir.parent.parent
            if run_dir.parent.name == "runs"
            else root / "campaigns" / campaign_slug
        )
    else:
        run_id = str(cfg.output.run_id)
        campaign_dir = root / "campaigns" / campaign_slug
        run_dir = campaign_dir / "runs" / run_id

    return RunContext(
        run_id=run_id,
        campaign_slug=campaign_slug,
        run_dir=run_dir,
        manifest_path=run_dir / "manifest.json",
        campaign_dir=campaign_dir,
        campaign_manifest_path=campaign_dir / "campaign_manifest.json",
        logs_dir=run_dir / "logs",
        log_path=run_dir / "logs" / f"{run_id}_jax.jsonl",
        debug_log_path=run_dir / "logs" / f"{run_id}_debug.jsonl",
        checkpoints_dir=run_dir / "checkpoints",
        queue_dir=run_dir / cfg.artifacts.artifact_pipeline.queue_dir,
        evaluations_dir=run_dir / cfg.artifacts.artifact_pipeline.result_dir,
        wandb_dir=run_dir / cfg.output.wandb_dir,
        wandb_artifact_dir=_cache_path(
            root, Path(cfg.output.cache_dir), cfg.output.wandb_artifact_dir
        ),
        wandb_data_dir=_cache_path(
            root, Path(cfg.output.cache_dir), cfg.output.wandb_data_dir
        ),
        indexes_dir=root / cfg.output.indexes_dir,
        retention_class=str(cfg.output.retention_class),
        model_compatibility_family=str(cfg.model.architecture),
    )


def feature_metadata_for_agent(agent: AgentEntry) -> dict[str, int | float | str | tuple]:
    checkpoint = load_checkpoint_payload(agent.checkpoint_path)
    stored = checkpoint_feature_metadata(checkpoint)
    if stored is not None:
        return dict(stored)
    return feature_metadata(agent.cfg.task, model_cfg=agent.cfg.model)


def validate_agents_feature_compatible(agents: Sequence[AgentEntry]) -> None:
    """Raise when tournament agents have incompatible feature metadata."""

    if len(agents) <= 1:
        return
    reference = feature_metadata_for_agent(agents[0])
    reference_keys = (
        "schema_version",
        "planet_feature_dim",
        "edge_feature_dim",
        "global_feature_dim",
        "edge_k",
        "encoder_backbone",
    )
    for agent in agents[1:]:
        metadata = feature_metadata_for_agent(agent)
        for key in reference_keys:
            if metadata.get(key) != reference.get(key):
                raise ValueError(
                    f"Agent {agent.agent_id!r} feature metadata {key}="
                    f"{metadata.get(key)!r} differs from reference "
                    f"{reference.get(key)!r} ({agents[0].agent_id!r})."
                )


def load_shortlist_rows(shortlist_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(shortlist_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"shortlist must be a JSON list: {shortlist_path}")
    rows: list[dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            rows.append(item)
    return rows


def download_wandb_checkpoint_artifact(
    artifact_name: str,
    cache_dir: Path,
    *,
    version: str | None = None,
    aliases: Sequence[str] = (),
) -> Path | None:
    """Download a W&B checkpoint artifact and return the first ``.pkl`` path."""

    try:
        import wandb  # type: ignore
    except ImportError:
        return None

    api = wandb.Api()
    if aliases:
        ref = f"{artifact_name}:{aliases[0]}"
    elif version:
        ref = f"{artifact_name}:{version}"
    else:
        ref = f"{artifact_name}:latest"
    try:
        artifact = api.artifact(ref)
        download_root = cache_dir / artifact_name.replace("/", "_")
        download_root.mkdir(parents=True, exist_ok=True)
        local_dir = Path(artifact.download(root=str(download_root)))
    except Exception as exc:
        logger.warning("W&B artifact download failed for %s: %s", ref, exc)
        return None

    candidates = sorted(local_dir.glob("**/*.pkl"))
    return candidates[0] if candidates else None


def resolve_shortlist_agents(
    shortlist_path: Path,
    *,
    limit: int,
    checkpoint_paths: dict[str, Path] | None = None,
    wandb_cache_dir: Path | None = None,
) -> ShortlistResolveResult:
    """Resolve shortlist rows to local checkpoint agents."""

    rows = load_shortlist_rows(shortlist_path)[: max(limit, 1)]
    agents: list[AgentEntry] = []
    skipped: list[str] = []
    errors: list[str] = []

    for row in rows:
        run_id = str(row.get("run_id", "")).strip() or str(row.get("name", "")).strip()
        if not run_id:
            skipped.append("row_missing_run_id")
            continue

        path: Path | None = None
        if checkpoint_paths and run_id in checkpoint_paths:
            path = checkpoint_paths[run_id]
        local_path = row.get("checkpoint_path")
        if path is None and isinstance(local_path, str) and local_path.strip():
            path = Path(local_path)
        artifact_name = row.get("checkpoint_artifact")
        if (
            (path is None or not path.exists())
            and isinstance(artifact_name, str)
            and artifact_name.strip()
            and wandb_cache_dir is not None
        ):
            aliases = row.get("checkpoint_artifact_aliases") or ()
            if isinstance(aliases, list):
                alias_tuple = tuple(str(alias) for alias in aliases)
            else:
                alias_tuple = ()
            version = row.get("checkpoint_artifact_version")
            downloaded = download_wandb_checkpoint_artifact(
                artifact_name.strip(),
                wandb_cache_dir,
                version=str(version) if version else None,
                aliases=alias_tuple,
            )
            if downloaded is not None:
                path = downloaded

        if path is None or not path.exists():
            skipped.append(run_id)
            errors.append(
                f"{run_id}: no local checkpoint_path and W&B artifact unavailable "
                f"(artifact={artifact_name!r})"
            )
            continue
        try:
            agents.append(agent_from_checkpoint(path, agent_id=run_id))
        except (ValueError, TypeError) as exc:
            skipped.append(run_id)
            errors.append(f"{run_id}: {exc}")

    return ShortlistResolveResult(
        agents=tuple(agents),
        skipped=tuple(skipped),
        errors=tuple(errors),
    )


def resolve_promoted_agent(
    campaign: str,
    output_root: str,
) -> AgentEntry | None:
    try:
        resolved = resolve_from_promoted(campaign, output_root)
    except (FileNotFoundError, ValueError):
        return None
    checkpoint_path = Path(resolved["checkpoint_path"])
    if not checkpoint_path.exists():
        return None
    return agent_from_checkpoint(checkpoint_path, agent_id="incumbent")
