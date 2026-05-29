from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import shutil
import tarfile
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from src.orchestration.accelerators import is_tpu_accelerator

KERNEL_PACKAGE_SOURCE_MODE = "embedded-payload-v6"


@dataclass(frozen=True, slots=True)
class KernelPackage:
    package_dir: Path
    metadata_path: Path
    worker_path: Path
    summary_path: Path


def render_kernel_package(
    *,
    package_dir: Path,
    kernel_id: str,
    title: str,
    worker_source: Path,
    env: Mapping[str, str],
    repo_root: Path | None = None,
    accelerator: str | None = None,
) -> KernelPackage:
    """Render a Kaggle script-kernel package for dry-run or launch.

    Kaggle script kernels execute only the declared ``code_file`` as
    ``/kaggle/src/script.py``. Other files in the local push directory are not
    guaranteed to be materialized under ``/kaggle/working``. Therefore the root
    ``kaggle_worker_entry.py`` must be self-contained and embed the repo payload.
    """

    package_dir.mkdir(parents=True, exist_ok=True)
    _remove_managed_payload(package_dir)
    if repo_root is not None:
        _copy_repo_payload(
            repo_root=repo_root,
            package_dir=package_dir,
            accelerator=accelerator,
        )
    env_path = package_dir / "worker-env.json"
    env_path.write_text(json.dumps(dict(env), indent=2) + "\n", encoding="utf-8")
    worker_path = package_dir / "kaggle_worker_entry.py"
    payload_sha256 = None
    if repo_root is not None:
        payload = _payload_archive(package_dir)
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        worker_path.write_text(
            _bootstrap_source(payload, manifest=_payload_manifest(package_dir)),
            encoding="utf-8",
        )
    else:
        shutil.copyfile(worker_source, worker_path)
    metadata = {
        "id": kernel_id,
        "title": title,
        "code_file": worker_path.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": not is_tpu_accelerator(accelerator or ""),
        "enable_tpu": is_tpu_accelerator(accelerator or ""),
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    metadata_path = package_dir / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    summary_path = package_dir / "package-summary.json"
    summary_path.write_text(
        json.dumps(
            _package_summary(
                package_dir=package_dir,
                kernel_id=kernel_id,
                title=title,
                metadata=metadata,
                env=env,
                payload_sha256=payload_sha256,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return KernelPackage(
        package_dir=package_dir,
        metadata_path=metadata_path,
        worker_path=worker_path,
        summary_path=summary_path,
    )


def _copy_repo_payload(
    *, repo_root: Path, package_dir: Path, accelerator: str | None
) -> None:
    for name in ("src", "conf"):
        target = package_dir / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(
            repo_root / name,
            target,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )
    scripts_target = package_dir / "scripts"
    if scripts_target.exists():
        shutil.rmtree(scripts_target)
    scripts_target.mkdir(parents=True, exist_ok=True)
    for script in (
        "kaggle_worker_entry.py",
        "benchmark_jax_rl.py",
        "kaggle_runtime_env.py",
    ):
        source = repo_root / "scripts" / script
        if source.exists():
            shutil.copyfile(source, scripts_target / script)
    pyproject = repo_root / "pyproject.toml"
    copied_lock = True
    if pyproject.exists():
        copied_lock = _copy_pyproject_for_kaggle_gpu(
            source=pyproject,
            target=package_dir / "pyproject.toml",
            accelerator=accelerator,
        )
    if copied_lock:
        lock = repo_root / "uv.lock"
        if lock.exists():
            shutil.copyfile(lock, package_dir / "uv.lock")
    readme = repo_root / "README.md"
    if readme.exists():
        shutil.copyfile(readme, package_dir / "README.md")


def _copy_pyproject_for_kaggle_gpu(
    *, source: Path, target: Path, accelerator: str | None
) -> bool:
    text = source.read_text(encoding="utf-8")
    if not _is_nvidia_accelerator(accelerator):
        target.write_text(text, encoding="utf-8")
        return True

    rewritten = _rewrite_jax_cuda_extras_to_plain_jax(text)
    target.write_text(rewritten, encoding="utf-8")
    return rewritten == text


def _rewrite_jax_cuda_extras_to_plain_jax(text: str) -> str:
    return re.sub(r"jax\[(?:cuda|cuda12|cuda13)\]", "jax", text)


def _is_nvidia_accelerator(accelerator: str | None) -> bool:
    return bool(accelerator and accelerator.strip().lower().startswith("nvidia"))


def _remove_managed_payload(package_dir: Path) -> None:
    managed_dirs = ("src", "conf", "scripts")
    managed_files = (
        "kaggle_worker_entry.py",
        "kernel-metadata.json",
        "worker-env.json",
        "package-summary.json",
        "pyproject.toml",
        "uv.lock",
        "README.md",
    )
    for name in managed_dirs:
        target = package_dir / name
        if target.exists():
            shutil.rmtree(target)
    for name in managed_files:
        target = package_dir / name
        if target.exists():
            target.unlink()


def _package_summary(
    *,
    package_dir: Path,
    kernel_id: str,
    title: str,
    metadata: Mapping[str, object],
    env: Mapping[str, str],
    payload_sha256: str | None,
) -> dict[str, object]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kernel_id": kernel_id,
        "title": title,
        "code_file": metadata.get("code_file"),
        "package_source_mode": KERNEL_PACKAGE_SOURCE_MODE,
        "payload_sha256": payload_sha256,
        "top_level_entries": sorted(
            {
                *(
                    path.name
                    for path in package_dir.iterdir()
                    if not path.name.startswith(".")
                ),
                "package-summary.json",
            }
        ),
        "generated_env": {
            key: _redact_if_secret(key, value) for key, value in sorted(env.items())
        },
    }


def _redact_if_secret(key: str, value: object) -> object:
    secret_markers = ("KEY", "TOKEN", "SECRET", "PASSWORD", "API_KEY")
    upper_key = key.upper()
    if any(marker in upper_key for marker in secret_markers):
        return "<redacted>"
    return value


def _payload_manifest(package_dir: Path) -> dict[str, object]:
    included_roots = (
        "src",
        "conf",
        "scripts",
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "worker-env.json",
    )
    files: list[str] = []
    for name in included_roots:
        path = package_dir / name
        if not path.exists():
            continue
        if path.is_file():
            files.append(name)
            continue
        for child in sorted(path.rglob("*")):
            if child.is_file():
                files.append(str(child.relative_to(package_dir)))
    return {
        "mode": KERNEL_PACKAGE_SOURCE_MODE,
        "file_count": len(files),
        "has_worker": "scripts/kaggle_worker_entry.py" in files,
        "has_kaggle_jax": "src/orchestration/kaggle_jax.py" in files,
        "sample": files[:25],
    }


def _payload_archive(package_dir: Path) -> bytes:
    included = (
        "src",
        "conf",
        "scripts",
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "worker-env.json",
    )
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in included:
            path = package_dir / name
            if path.exists():
                archive.add(path, arcname=name, recursive=True)
    return buffer.getvalue()


def _bootstrap_source(payload: bytes, *, manifest: Mapping[str, object]) -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    manifest_json = json.dumps(dict(manifest), sort_keys=True)
    chunks = "\n".join(f'    "{chunk}"' for chunk in textwrap.wrap(encoded, 76))
    return f'''#!/usr/bin/env python3
"""Self-contained Kaggle bootstrap for Orbit Wars population workers."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import runpy
import sys
import tarfile
from pathlib import Path

ORBIT_WARS_PACKAGE_BOOTSTRAP = {KERNEL_PACKAGE_SOURCE_MODE!r}
_PAYLOAD_SHA256 = {digest!r}
_PAYLOAD_MANIFEST = json.loads({manifest_json!r})
_PAYLOAD_B64 = (
{chunks}
)


def _extract_payload(root: Path, payload: bytes) -> None:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        try:
            archive.extractall(root, filter="data")
        except TypeError:
            archive.extractall(root)


def main() -> None:
    root = Path.cwd()
    payload = base64.b64decode(_PAYLOAD_B64)
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    print("ORBIT_WARS_PACKAGE_BOOTSTRAP=" + ORBIT_WARS_PACKAGE_BOOTSTRAP, flush=True)
    print("ORBIT_WARS_PACKAGE_PAYLOAD_SHA256=" + actual_sha256, flush=True)
    print("ORBIT_WARS_PACKAGE_PAYLOAD_MANIFEST=" + json.dumps(_PAYLOAD_MANIFEST, sort_keys=True), flush=True)
    if actual_sha256 != _PAYLOAD_SHA256:
        raise SystemExit(
            "embedded payload sha256 mismatch: " + actual_sha256 + " != " + _PAYLOAD_SHA256
        )
    _extract_payload(root, payload)
    worker = root / "scripts" / "kaggle_worker_entry.py"
    if not worker.exists():
        top_level = sorted(path.name for path in root.iterdir())
        raise SystemExit(
            "embedded payload extraction did not create scripts/kaggle_worker_entry.py; "
            + "cwd=" + str(root)
            + " top_level=" + ",".join(top_level)
        )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    runpy.run_path(str(worker), run_name="__main__")


if __name__ == "__main__":
    main()
'''
