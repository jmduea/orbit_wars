"""Operator promotion helpers (show, history, demote)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from src.artifacts.promotion_manifest import (
    append_promotion_index,
    merge_campaign_manifest,
    promoted_manifest_path,
    write_promoted_manifest,
)

DemoteAction = Literal["cleared", "restored_previous", "noop"]


@dataclass(slots=True)
class DemoteResult:
    """Outcome of an operator demote attempt."""

    action: DemoteAction
    reason: str
    campaign: str
    campaign_dir: Path
    dry_run: bool
    previous_manifest_path: Path | None = None
    restored_manifest_path: Path | None = None


def campaign_dir(output_root: Path, campaign: str) -> Path:
    slug = str(campaign).strip()
    if not slug:
        raise ValueError("campaign slug is required")
    return output_root / "campaigns" / slug


def _indexes_dir(output_root: Path) -> Path:
    return output_root / "indexes"


def read_promotion_index(
    output_root: Path,
    *,
    campaign: str | None = None,
) -> list[dict[str, object]]:
    """Load promotion index records, optionally filtered by campaign slug."""

    path = _indexes_dir(output_root) / "promoted.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    slug = str(campaign or "").strip() or None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            continue
        if slug is not None and str(raw.get("campaign", "")).strip() != slug:
            continue
        rows.append(raw)
    return rows


def _promotion_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Promotion events only (exclude operator demote audit rows)."""

    out: list[dict[str, object]] = []
    for record in records:
        event = str(record.get("event", "promoted")).strip().lower()
        if event == "demoted":
            continue
        if str(record.get("checkpoint_path", "")).strip():
            out.append(record)
    return out


def load_current_promoted_manifest(campaign_dir: Path) -> dict[str, object] | None:
    """Return the current promoted manifest payload, if present."""

    path = promoted_manifest_path(campaign_dir)
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Promoted manifest at {path} is not a JSON object.")
    return raw


def demote_campaign(
    output_root: Path,
    campaign: str,
    *,
    to_previous: bool = False,
    dry_run: bool = False,
    reason: str = "operator_demote",
) -> DemoteResult:
    """Clear or roll back campaign promotion for operator workflows."""

    root = output_root.resolve()
    camp_dir = campaign_dir(root, campaign)
    manifest_path = promoted_manifest_path(camp_dir)
    campaign_manifest_path = camp_dir / "campaign_manifest.json"
    indexes = _indexes_dir(root)

    history = read_promotion_index(root, campaign=campaign)
    promotions = _promotion_records(history)

    if not manifest_path.is_file() and not promotions:
        return DemoteResult(
            action="noop",
            reason="no_promotion",
            campaign=campaign,
            campaign_dir=camp_dir,
            dry_run=dry_run,
        )

    now = datetime.now(timezone.utc).isoformat()
    previous_payload = load_current_promoted_manifest(camp_dir) if manifest_path.is_file() else None

    if to_previous:
        if len(promotions) < 2:
            return DemoteResult(
                action="noop",
                reason="no_previous_promotion",
                campaign=campaign,
                campaign_dir=camp_dir,
                dry_run=dry_run,
                previous_manifest_path=manifest_path if manifest_path.is_file() else None,
            )
        target = promotions[-2]
        checkpoint_path = Path(str(target.get("checkpoint_path", "")).strip())
        if not checkpoint_path.is_file():
            return DemoteResult(
                action="noop",
                reason="previous_checkpoint_missing",
                campaign=campaign,
                campaign_dir=camp_dir,
                dry_run=dry_run,
                previous_manifest_path=manifest_path if manifest_path.is_file() else None,
            )
        restored_payload: dict[str, object] = {
            "campaign": campaign,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "metric_name": target.get("metric_name"),
            "metric_value": target.get("metric_value"),
            "metric_mode": target.get("metric_mode", "max"),
            "source_run_id": target.get("run_id"),
            "source_update": target.get("update"),
            "updated_at": now,
            "restored_from_index": True,
        }
        for key in (
            "hydra_overrides_path",
            "git",
            "feature_metadata",
            "promotion_strategy",
            "tournament_id",
            "tournament_win_rate_vs_sniper",
            "tournament_win_rate_vs_incumbent",
        ):
            if key in target:
                restored_payload[key] = target[key]

        if dry_run:
            return DemoteResult(
                action="restored_previous",
                reason="dry_run",
                campaign=campaign,
                campaign_dir=camp_dir,
                dry_run=True,
                previous_manifest_path=manifest_path if manifest_path.is_file() else None,
            )

        manifest_out = write_promoted_manifest(camp_dir, restored_payload)
        metric_value = target.get("metric_value")
        merge_campaign_manifest(
            campaign_manifest_path,
            {
                "campaign": campaign,
                "campaign_dir": str(camp_dir),
                "current_best_value": metric_value,
                "current_best_run_id": target.get("run_id"),
                "updated_at": now,
            },
        )
        append_promotion_index(
            indexes,
            {
                "event": "demoted",
                "campaign": campaign,
                "action": "restored_previous",
                "reason": reason,
                "previous_checkpoint_path": (
                    str(previous_payload.get("checkpoint_path"))
                    if isinstance(previous_payload, dict)
                    else None
                ),
                "restored_checkpoint_path": str(checkpoint_path.resolve()),
                "promoted_manifest_path": str(manifest_out),
                "updated_at": now,
            },
        )
        return DemoteResult(
            action="restored_previous",
            reason=reason,
            campaign=campaign,
            campaign_dir=camp_dir,
            dry_run=False,
            previous_manifest_path=manifest_path if manifest_path.is_file() else None,
            restored_manifest_path=manifest_out,
        )

    had_manifest = manifest_path.is_file()
    if dry_run:
        return DemoteResult(
            action="cleared",
            reason="dry_run",
            campaign=campaign,
            campaign_dir=camp_dir,
            dry_run=True,
            previous_manifest_path=manifest_path if had_manifest else None,
        )

    if had_manifest:
        manifest_path.unlink()

    merge_campaign_manifest(
        campaign_manifest_path,
        {
            "campaign": campaign,
            "campaign_dir": str(camp_dir),
            "current_best_value": None,
            "current_best_run_id": None,
            "updated_at": now,
        },
    )
    append_promotion_index(
        indexes,
        {
            "event": "demoted",
            "campaign": campaign,
            "action": "cleared",
            "reason": reason,
            "previous_checkpoint_path": (
                str(previous_payload.get("checkpoint_path"))
                if isinstance(previous_payload, dict)
                else None
            ),
            "updated_at": now,
        },
    )
    return DemoteResult(
        action="cleared",
        reason=reason,
        campaign=campaign,
        campaign_dir=camp_dir,
        dry_run=False,
        previous_manifest_path=manifest_path if had_manifest else None,
    )
