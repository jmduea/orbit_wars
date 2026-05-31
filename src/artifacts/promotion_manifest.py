from __future__ import annotations

from pathlib import Path
from typing import Mapping

from src.artifacts.run_paths import append_jsonl_atomic, atomic_write_json

PROMOTED_MANIFEST_NAME = "manifest.json"


def promoted_dir(campaign_dir: Path) -> Path:
    return campaign_dir / "promoted" / "current_best"


def promoted_manifest_path(campaign_dir: Path) -> Path:
    return promoted_dir(campaign_dir) / PROMOTED_MANIFEST_NAME


def write_promoted_manifest(
    campaign_dir: Path, payload: Mapping[str, object]
) -> Path:
    manifest_out = promoted_manifest_path(campaign_dir)
    atomic_write_json(manifest_out, dict(payload))
    return manifest_out


def merge_campaign_manifest(
    campaign_manifest_path: Path,
    updates: Mapping[str, object],
) -> None:
    campaign_payload: dict[str, object] = {}
    if campaign_manifest_path.exists():
        import json

        raw = json.loads(campaign_manifest_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            campaign_payload = raw
    campaign_payload.update(dict(updates))
    atomic_write_json(campaign_manifest_path, campaign_payload)


def append_promotion_index(
    indexes_dir: Path, record: Mapping[str, object]
) -> None:
    append_jsonl_atomic(indexes_dir / "promoted.jsonl", dict(record))
