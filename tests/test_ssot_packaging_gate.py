"""SSOT packaging validation gate before long train."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.schema import (
    ArtifactsConfig,
    OutputConfig,
    SsotPipelineConfig,
    TrainConfig,
)
from src.artifacts.packaging_validation import (
    assert_ssot_packaging_gate,
    write_packaging_validation_record,
)


def _ssot_cfg(tmp_path: Path, *, require: bool = True) -> TrainConfig:
    cfg = TrainConfig()
    cfg.output = OutputConfig(root=str(tmp_path))
    cfg.artifacts = ArtifactsConfig(
        ssot_pipeline=SsotPipelineConfig(
            enabled=True,
            require_packaging_validation=require,
        )
    )
    return cfg


def test_packaging_gate_raises_without_marker(tmp_path: Path) -> None:
    cfg = _ssot_cfg(tmp_path)
    with pytest.raises(RuntimeError, match="packaging validation"):
        assert_ssot_packaging_gate(cfg)


def test_packaging_gate_allows_after_marker_written(tmp_path: Path) -> None:
    cfg = _ssot_cfg(tmp_path)
    ckpt = tmp_path / "jax_ckpt.pkl"
    ckpt.write_bytes(b"stub")
    marker = tmp_path / "ssot" / "packaging_validation.json"
    write_packaging_validation_record(
        marker,
        checkpoint_path=ckpt,
        packaging_seed=0,
        packaging_player_count="4",
        package_path=tmp_path / "submission.tar.gz",
    )
    cfg.artifacts.ssot_pipeline.packaging_validation_path = str(marker)
    assert_ssot_packaging_gate(cfg)


def test_packaging_gate_skipped_when_disabled(tmp_path: Path) -> None:
    cfg = _ssot_cfg(tmp_path, require=False)
    assert_ssot_packaging_gate(cfg)
