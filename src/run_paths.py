from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .config import TrainConfig


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


def compose_run_name(cfg: TrainConfig) -> str:
    """Compose a collision-resistant run name from experiment + seed + time."""

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = cfg.run_name
    hydra_job = ""
    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            cfg_hydra = HydraConfig.get()
            choices = getattr(cfg_hydra.runtime, "choices", {}) or {}
            experiment = str(choices.get("experiment") or cfg.run_name)
            job = getattr(cfg_hydra, "job", None)
            job_num = getattr(job, "num", None) if job is not None else None
            if job_num is not None:
                hydra_job = f"-job{int(job_num):04d}"
    except Exception:
        pass
    return f"{experiment}-s{int(cfg.seed)}-{ts}{hydra_job}"


def resolve_run_paths(cfg: TrainConfig) -> tuple[TrainConfig, Path, Path, Path]:
    """Resolve unified run paths rooted in Hydra runtime output_dir when present."""

    output_dir = _hydra_runtime_output_dir()
    run_name = cfg.run_name
    if output_dir is not None:
        run_name = compose_run_name(cfg)
        save_dir = output_dir / "checkpoints"
        logs_dir = output_dir / "logs"
    else:
        save_dir = Path(cfg.save_dir)
        logs_dir = save_dir / "logs"
    run_dir = save_dir / run_name
    log_path = logs_dir / f"{run_name}.jsonl"
    return replace(cfg, run_name=run_name, save_dir=str(save_dir)), run_dir, log_path, save_dir
