from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class KernelPackage:
    package_dir: Path
    metadata_path: Path
    worker_path: Path


def render_kernel_package(
    *,
    package_dir: Path,
    kernel_id: str,
    title: str,
    worker_source: Path,
    env: Mapping[str, str],
    repo_root: Path | None = None,
) -> KernelPackage:
    """Render a Kaggle script-kernel package for dry-run or launch."""

    package_dir.mkdir(parents=True, exist_ok=True)
    if repo_root is not None:
        _copy_repo_payload(repo_root=repo_root, package_dir=package_dir)
    worker_path = package_dir / "kaggle_worker_entry.py"
    shutil.copyfile(worker_source, worker_path)
    metadata = {
        "id": kernel_id,
        "title": title,
        "code_file": worker_path.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    metadata_path = package_dir / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    env_path = package_dir / "worker-env.json"
    env_path.write_text(json.dumps(dict(env), indent=2) + "\n", encoding="utf-8")
    return KernelPackage(
        package_dir=package_dir,
        metadata_path=metadata_path,
        worker_path=worker_path,
    )


def _copy_repo_payload(*, repo_root: Path, package_dir: Path) -> None:
    for name in ("src", "conf"):
        target = package_dir / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(
            repo_root / name,
            target,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    scripts_target = package_dir / "scripts"
    if scripts_target.exists():
        shutil.rmtree(scripts_target)
    scripts_target.mkdir(parents=True, exist_ok=True)
    for script in ("kaggle_worker_entry.py",):
        shutil.copyfile(repo_root / "scripts" / script, scripts_target / script)
    for name in ("pyproject.toml", "uv.lock"):
        source = repo_root / name
        if source.exists():
            shutil.copyfile(source, package_dir / name)
