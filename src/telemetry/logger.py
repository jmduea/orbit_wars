from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.config import TrainConfig
from src.telemetry.wandb_run_name import apply_post_init_run_rename
from src.telemetry.wandb_tags import derive_config_group_tags, merge_wandb_tags


class TelemetryLogger:
    """Shared optional telemetry logger with best-effort WandB integration."""

    def __init__(self, cfg: TrainConfig, run_metadata: dict[str, Any] | None = None):
        self._cfg = cfg
        self._run = None
        self._wandb = None
        self._enabled = bool(cfg.telemetry.wandb.enabled)
        self._log_model_every = max(int(cfg.telemetry.wandb.log_model_every), 1)
        self._last_wandb_step: int | None = None
        self._run_metadata = dict(run_metadata or {})
        self._init()

    def _init(self) -> None:
        if not self._enabled:
            return
        run_metadata = self._run_metadata
        wandb_dir = run_metadata.get("wandb_dir")
        wandb_artifact_dir = run_metadata.get("wandb_artifact_dir")
        wandb_data_dir = run_metadata.get("wandb_data_dir")
        if wandb_dir:
            Path(str(wandb_dir)).mkdir(parents=True, exist_ok=True)
            os.environ["WANDB_DIR"] = str(wandb_dir)
        if wandb_artifact_dir:
            Path(str(wandb_artifact_dir)).mkdir(parents=True, exist_ok=True)
            os.environ["WANDB_ARTIFACT_DIR"] = str(wandb_artifact_dir)
        if wandb_data_dir:
            Path(str(wandb_data_dir)).mkdir(parents=True, exist_ok=True)
            os.environ["WANDB_DATA_DIR"] = str(wandb_data_dir)
        try:
            import wandb  # type: ignore
        except ImportError:
            return
        self._wandb = wandb
        wandb_cfg = self._cfg.telemetry.wandb
        derived_tags: list[str] = []
        if wandb_cfg.tags_from_config_groups:
            derived_tags = derive_config_group_tags(
                allowlist=wandb_cfg.tag_config_groups,
            )
        tags = merge_wandb_tags(manual=wandb_cfg.tags, derived=derived_tags)
        self._run = wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            group=wandb_cfg.group,
            tags=tags,
            name=self._cfg.run_name,
            config={},
            reinit=True,
            job_type=str(run_metadata.get("job_type", "train")),
            dir=str(wandb_dir) if wandb_dir else None,
            id=os.environ.get("WANDB_RUN_ID") or None,
            resume=os.environ.get("WANDB_RESUME") or None,
        )
        define_metric = getattr(wandb, "define_metric", None)
        if callable(define_metric):
            define_metric("win_rate_delta_10", summary="mean")
            define_metric("win_rate_recovery_delta_10", summary="max")
            define_metric("win_rate_window_mean_10", summary="max")
            define_metric("win_rate_best_window_mean_10", summary="max")
            define_metric("entropy_delta_10", summary="min")
            define_metric("entropy_retention_ratio_10", summary="min")
        apply_post_init_run_rename(self._run, self._cfg)
        if self._run is not None:
            resolved_cfg = self._flatten(asdict(self._cfg))
            existing_keys = set(self._run.config.keys())
            missing_keys = {
                key: value
                for key, value in resolved_cfg.items()
                if key not in existing_keys
            }
            if missing_keys:
                self._run.config.update(missing_keys, allow_val_change=True)
        if run_metadata:
            self.log(run_metadata, step=0)

    @property
    def active(self) -> bool:
        return self._run is not None and self._wandb is not None

    def watch_model(self, model: Any) -> None:
        if not self.active or not self._cfg.telemetry.wandb.watch_model:
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
        wandb_step = self._next_wandb_step(step)
        self._wandb.log(self._flatten(record), step=wandb_step)

    def _next_wandb_step(self, requested_step: int | None) -> int | None:
        if requested_step is None:
            return None
        requested_step = int(requested_step)
        if self._last_wandb_step is None or requested_step >= self._last_wandb_step:
            self._last_wandb_step = requested_step
            return requested_step
        self._last_wandb_step += 1
        return self._last_wandb_step

    def log_artifact(
        self,
        path: str | Path,
        *,
        name: str,
        artifact_type: str,
        metadata: dict[str, object] | None = None,
        aliases: list[str] | None = None,
    ) -> None:
        if not self.active or not self._cfg.telemetry.wandb.log_artifacts:
            return
        artifact_path = Path(path)
        if not artifact_path.exists():
            return
        artifact = self._wandb.Artifact(
            name=name,
            type=artifact_type,
            metadata=metadata,
        )
        artifact.add_file(str(artifact_path))
        self._run.log_artifact(artifact, aliases=aliases)

    def log_checkpoint(self, path: str | Path, *, update: int) -> None:
        self.log_artifact(
            path,
            name=f"checkpoint-u{update}",
            artifact_type="checkpoint",
            metadata=self._checkpoint_metadata(update=update),
            aliases=["latest", f"update-{int(update)}"],
        )

    def log_promoted_checkpoint(
        self,
        path: str | Path,
        *,
        update: int,
        metric_name: str,
        metric_value: float,
    ) -> None:
        """Upload a promoted checkpoint with best/promoted aliases."""

        self.log_artifact(
            path,
            name=f"checkpoint-promoted-u{update}",
            artifact_type="checkpoint",
            metadata=self._checkpoint_metadata(
                update=update,
                metric_name=metric_name,
                metric_value=float(metric_value),
            ),
            aliases=["best", "promoted", f"update-{int(update)}"],
        )

    def _checkpoint_metadata(
        self,
        *,
        update: int,
        metric_name: str | None = None,
        metric_value: float | None = None,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "update": int(update),
            "run_name": self._cfg.run_name,
            "campaign": self._run_metadata.get("campaign"),
            "run_id": self._run_metadata.get("run_id"),
            "wandb_run_id": getattr(self._run, "id", None),
        }
        if metric_name is not None:
            metadata["metric_name"] = metric_name
        if metric_value is not None:
            metadata["metric_value"] = metric_value
        return {key: value for key, value in metadata.items() if value is not None}

    def log_replay(self, path: str | Path, *, update: int) -> None:
        self.log_artifact(path, name=f"replay-u{update}", artifact_type="replay")

    def finish(self) -> None:
        if self._run is None:
            return
        self._run.finish()
        self._run = None
