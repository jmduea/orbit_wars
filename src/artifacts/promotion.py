from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from src.artifacts.checkpoint_compat import feature_metadata
from src.artifacts.checkpoint_retention import _collect_metric_by_update
from src.artifacts.run_paths import RunContext, _git_identity, atomic_write_json, append_jsonl_atomic
from src.config import TrainConfig

MetricMode = Literal["max", "min"]

PROMOTED_MANIFEST_NAME = "manifest.json"


@dataclass(slots=True)
class PromotionAttempt:
    """Outcome of a single promotion compare-and-swap attempt."""

    promoted: bool
    reason: str
    metric_name: str
    metric_value: float | None = None
    metric_mode: str = "max"
    promoted_manifest_path: Path | None = None


def promoted_dir(campaign_dir: Path) -> Path:
    return campaign_dir / "promoted" / "current_best"


def promoted_manifest_path(campaign_dir: Path) -> Path:
    return promoted_dir(campaign_dir) / PROMOTED_MANIFEST_NAME


def resolve_from_promoted(campaign_slug: str, output_root: str) -> dict[str, str]:
    """Load promoted checkpoint pointer for ``ow train from_promoted=...``."""

    campaign_dir = Path(output_root) / "campaigns" / campaign_slug
    manifest_path = promoted_manifest_path(campaign_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No promoted manifest for campaign {campaign_slug!r} at {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_path = str(payload.get("checkpoint_path", "")).strip()
    if not checkpoint_path:
        raise ValueError(
            f"Promoted manifest at {manifest_path} is missing checkpoint_path."
        )
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Promoted checkpoint path does not exist: {checkpoint_path}"
        )
    return {
        "campaign": campaign_slug,
        "checkpoint_path": checkpoint_path,
        "manifest_path": str(manifest_path),
    }


def _metric_improves(
    candidate: float,
    incumbent: float | None,
    *,
    mode: MetricMode,
) -> bool:
    if incumbent is None:
        return True
    if mode == "max":
        return candidate > incumbent
    return candidate < incumbent


def _run_metric_at_update(log_path: Path, metric_name: str, update: int) -> float | None:
    metrics = _collect_metric_by_update(log_path, metric_name)
    value = metrics.get(update)
    return None if value is None else float(value)


def promote_if_better(
    cfg: TrainConfig,
    context: RunContext,
    *,
    checkpoint_path: Path,
    update: int,
    log_path: Path,
    run_best_value: float | None,
) -> tuple[PromotionAttempt, float | None]:
    """Compare-and-swap promote when the run-local best improves campaign best."""

    promotion = cfg.artifacts.promotion
    if not promotion.enabled:
        return (
            PromotionAttempt(promoted=False, reason="disabled", metric_name=""),
            run_best_value,
        )

    metric_name = str(promotion.metric_name or "").strip()
    metric_mode = str(promotion.metric_mode or "max").strip().lower()
    if metric_mode not in {"max", "min"}:
        metric_mode = "max"

    metric_value = _run_metric_at_update(log_path, metric_name, update)
    if metric_value is None:
        return (
            PromotionAttempt(
                promoted=False,
                reason="metric_missing",
                metric_name=metric_name,
                metric_mode=metric_mode,
            ),
            run_best_value,
        )

    if not _metric_improves(
        metric_value, run_best_value, mode=metric_mode  # type: ignore[arg-type]
    ):
        return (
            PromotionAttempt(
                promoted=False,
                reason="run_best_unchanged",
                metric_name=metric_name,
                metric_value=metric_value,
                metric_mode=metric_mode,
            ),
            run_best_value,
        )

    new_run_best = metric_value
    campaign_manifest_path = context.campaign_manifest_path
    campaign_payload: dict[str, object] = {}
    if campaign_manifest_path.exists():
        raw = json.loads(campaign_manifest_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            campaign_payload = raw

    frozen_name = str(campaign_payload.get("promotion_metric_name", "")).strip()
    if frozen_name and frozen_name != metric_name:
        return (
            PromotionAttempt(
                promoted=False,
                reason="metric_mismatch",
                metric_name=metric_name,
                metric_value=metric_value,
                metric_mode=metric_mode,
            ),
            new_run_best,
        )

    incumbent_value = campaign_payload.get("current_best_value")
    incumbent_float: float | None
    if isinstance(incumbent_value, int | float):
        incumbent_float = float(incumbent_value)
    else:
        incumbent_float = None

    if not _metric_improves(
        metric_value, incumbent_float, mode=metric_mode  # type: ignore[arg-type]
    ):
        return (
            PromotionAttempt(
                promoted=False,
                reason="campaign_best_unchanged",
                metric_name=metric_name,
                metric_value=metric_value,
                metric_mode=metric_mode,
            ),
            new_run_best,
        )

    now = datetime.now(timezone.utc).isoformat()
    overrides_path = context.run_dir / ".hydra" / "overrides.yaml"
    promoted_payload: dict[str, object] = {
        "campaign": context.campaign_slug,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metric_mode": metric_mode,
        "source_run_id": context.run_id,
        "source_update": int(update),
        "hydra_overrides_path": str(overrides_path),
        "git": _git_identity(),
        "feature_metadata": feature_metadata(cfg.task, model_cfg=cfg.model),
        "updated_at": now,
    }

    manifest_out = promoted_manifest_path(context.campaign_dir)
    atomic_write_json(manifest_out, promoted_payload)

    campaign_payload.update(
        {
            "campaign": context.campaign_slug,
            "campaign_dir": str(context.campaign_dir),
            "promotion_metric_name": metric_name,
            "promotion_metric_mode": metric_mode,
            "current_best_value": metric_value,
            "current_best_run_id": context.run_id,
            "updated_at": now,
        }
    )
    atomic_write_json(campaign_manifest_path, campaign_payload)

    indexes_dir = context.indexes_dir
    append_jsonl_atomic(
        indexes_dir / "promoted.jsonl",
        {
            "campaign": context.campaign_slug,
            "run_id": context.run_id,
            "update": int(update),
            "metric_name": metric_name,
            "metric_value": metric_value,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "promoted_manifest_path": str(manifest_out),
            "updated_at": now,
        },
    )

    return (
        PromotionAttempt(
            promoted=True,
            reason="promoted",
            metric_name=metric_name,
            metric_value=metric_value,
            metric_mode=metric_mode,
            promoted_manifest_path=manifest_out,
        ),
        new_run_best,
    )
