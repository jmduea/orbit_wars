from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import TrainConfig


class TelemetryLogger:
    """Shared optional telemetry logger with best-effort WandB integration."""

    def __init__(self, cfg: TrainConfig, run_metadata: dict[str, Any] | None = None):
        self._cfg = cfg
        self._run = None
        self._wandb = None
        self._enabled = bool(cfg.wandb.enabled)
        self._log_model_every = max(int(cfg.wandb.log_model_every), 1)
        self._init(run_metadata or {})

    def _init(self, run_metadata: dict[str, Any]) -> None:
        if not self._enabled:
            return
        try:
            import wandb  # type: ignore
        except ImportError:
            return
        self._wandb = wandb
        tags = list(self._cfg.wandb.tags)
        self._run = wandb.init(
            project=self._cfg.wandb.project,
            entity=self._cfg.wandb.entity,
            group=self._cfg.wandb.group,
            tags=tags,
            name=self._cfg.run_name,
            config={},
            reinit=True,
            job_type=str(run_metadata.get("job_type", "train")),
        )
        if self._run is not None:
            resolved_cfg = self._flatten(asdict(self._cfg))
            existing_keys = set(self._run.config.keys())
            missing_keys = {
                key: value for key, value in resolved_cfg.items() if key not in existing_keys
            }
            if missing_keys:
                self._run.config.update(missing_keys, allow_val_change=True)
        if run_metadata:
            self.log(run_metadata, step=0)

    @property
    def active(self) -> bool:
        return self._run is not None and self._wandb is not None

    def watch_model(self, model: Any) -> None:
        if not self.active or not self._cfg.wandb.watch_model:
            return
        self._wandb.watch(model, log="all", log_freq=self._log_model_every)

    def _flatten(self, payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        flat: dict[str, Any] = {}
        for key, value in payload.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                flat.update(self._flatten(value, prefix=full_key))
            else:
                flat[full_key] = value
        return flat

    def log(self, record: dict[str, Any], *, step: int | None = None) -> None:
        if not self.active:
            return
        self._wandb.log(self._flatten(record), step=step)

    def log_artifact(self, path: str | Path, *, name: str, artifact_type: str) -> None:
        if not self.active or not self._cfg.wandb.log_artifacts:
            return
        artifact_path = Path(path)
        if not artifact_path.exists():
            return
        artifact = self._wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(str(artifact_path))
        self._run.log_artifact(artifact)

    def log_checkpoint(self, path: str | Path, *, update: int) -> None:
        self.log_artifact(path, name=f"checkpoint-u{update}", artifact_type="checkpoint")

    def log_replay(self, path: str | Path, *, update: int) -> None:
        self.log_artifact(path, name=f"replay-u{update}", artifact_type="replay")

    def finish(self) -> None:
        if self._run is None:
            return
        self._run.finish()
        self._run = None


def build_telemetry(cfg: TrainConfig, run_metadata: dict[str, Any] | None = None) -> TelemetryLogger:
    return TelemetryLogger(cfg, run_metadata=run_metadata)
