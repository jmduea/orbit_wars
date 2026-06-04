"""SSOT packaging validation marker (step 4 → step 5 gate)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.config import TrainConfig


@dataclass(frozen=True, slots=True)
class PackagingValidationRecord:
    ok: bool
    checkpoint_path: str
    packaging_seed: int
    packaging_player_count: str
    package_path: str
    validated_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "checkpoint_path": self.checkpoint_path,
            "packaging_seed": self.packaging_seed,
            "packaging_player_count": self.packaging_player_count,
            "package_path": self.package_path,
            "validated_at": self.validated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> PackagingValidationRecord:
        return cls(
            ok=bool(payload.get("ok")),
            checkpoint_path=str(payload.get("checkpoint_path", "")),
            packaging_seed=int(payload.get("packaging_seed", 0)),
            packaging_player_count=str(payload.get("packaging_player_count", "4")),
            package_path=str(payload.get("package_path", "")),
            validated_at=str(payload.get("validated_at", "")),
        )


def default_packaging_validation_path(cfg: TrainConfig) -> Path:
    ssot = cfg.artifacts.ssot_pipeline
    if ssot.packaging_validation_path:
        return Path(ssot.packaging_validation_path)
    return Path(cfg.output.root) / "ssot" / "packaging_validation.json"


def write_packaging_validation_record(
    path: Path,
    *,
    checkpoint_path: Path,
    packaging_seed: int,
    packaging_player_count: str,
    package_path: Path,
) -> PackagingValidationRecord:
    record = PackagingValidationRecord(
        ok=True,
        checkpoint_path=str(checkpoint_path.resolve()),
        packaging_seed=int(packaging_seed),
        packaging_player_count=str(packaging_player_count),
        package_path=str(package_path.resolve()),
        validated_at=datetime.now(UTC).isoformat(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict(), indent=2) + "\n", encoding="utf-8")
    return record


def load_packaging_validation_record(path: Path) -> PackagingValidationRecord | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"packaging validation record must be a JSON object: {path}")
    return PackagingValidationRecord.from_dict(payload)


def assert_ssot_packaging_gate(cfg: TrainConfig) -> None:
    """Raise when SSOT long train starts without a packaging validation marker."""

    ssot = getattr(cfg.artifacts, "ssot_pipeline", None)
    if not ssot or not ssot.enabled or not ssot.require_packaging_validation:
        return
    marker_path = default_packaging_validation_path(cfg)
    record = load_packaging_validation_record(marker_path)
    if record is None or not record.ok:
        raise RuntimeError(
            "SSOT long train requires packaging validation before start. "
            f"Run: uv run ow eval package --checkpoint <winner.pkl> --output-dir <dir> "
            f"--validate-docker --packaging-seed 0 --packaging-player-count 4 "
            f"(writes marker to {marker_path}). "
            "Or set artifacts.ssot_pipeline.require_packaging_validation=false for smokes."
        )
