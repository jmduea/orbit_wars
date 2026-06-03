"""Tests for incumbent lineage qualifier skip."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from src.artifacts.tournament.bracket.lineage import (
    parent_checkpoint_path,
    qualifier_skip_for_checkpoint,
)


def _write_checkpoint(path: Path, *, parent: str | None = None) -> None:
    payload: dict[str, object] = {"update": 1, "params": {}}
    if parent is not None:
        payload["parent_checkpoint_path"] = parent
    path.write_bytes(pickle.dumps(payload))


def _write_promoted_manifest(campaign_dir: Path, checkpoint: Path) -> None:
    promoted_dir = campaign_dir / "promoted" / "current_best"
    promoted_dir.mkdir(parents=True)
    manifest = {"checkpoint_path": str(checkpoint.resolve())}
    (promoted_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_parent_matches_incumbent_skips(tmp_path: Path) -> None:
    incumbent = tmp_path / "incumbent.pkl"
    child = tmp_path / "child.pkl"
    _write_checkpoint(incumbent)
    _write_checkpoint(child, parent=str(incumbent.resolve()))
    campaign_dir = tmp_path / "campaigns" / "demo"
    _write_promoted_manifest(campaign_dir, incumbent)
    assert (
        qualifier_skip_for_checkpoint(
            child,
            campaign="demo",
            output_root=tmp_path,
        )
        is True
    )


def test_parent_mismatch_does_not_skip(tmp_path: Path) -> None:
    incumbent = tmp_path / "incumbent.pkl"
    other = tmp_path / "other.pkl"
    child = tmp_path / "child.pkl"
    _write_checkpoint(incumbent)
    _write_checkpoint(other)
    _write_checkpoint(child, parent=str(other.resolve()))
    campaign_dir = tmp_path / "campaigns" / "demo"
    _write_promoted_manifest(campaign_dir, incumbent)
    assert (
        qualifier_skip_for_checkpoint(
            child,
            campaign="demo",
            output_root=tmp_path,
        )
        is False
    )


def test_missing_parent_does_not_skip(tmp_path: Path) -> None:
    ckpt = tmp_path / "solo.pkl"
    _write_checkpoint(ckpt)
    assert parent_checkpoint_path(ckpt) is None
    assert (
        qualifier_skip_for_checkpoint(
            ckpt,
            campaign="demo",
            output_root=tmp_path,
        )
        is False
    )
