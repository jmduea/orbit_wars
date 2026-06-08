from __future__ import annotations

import base64
import hashlib
import json
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.orchestration.accelerators import is_tpu_accelerator
from src.orchestration.remote_package import (
    DEFAULT_KAGGLE_SCRIPTS,
    RemotePackageOptions,
    copy_repo_payload,
    package_summary,
    payload_archive,
    payload_manifest,
    remove_managed_payload,
    write_worker_env,
)

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
    remove_managed_payload(package_dir)
    if repo_root is not None:
        copy_repo_payload(
            repo_root=repo_root,
            package_dir=package_dir,
            options=RemotePackageOptions(
                scripts=DEFAULT_KAGGLE_SCRIPTS,
                accelerator=accelerator,
                strip_jax_for_nvidia_gpu=True,
                include_map_pool=True,
            ),
        )
    write_worker_env(package_dir=package_dir, env=env)
    worker_path = package_dir / "kaggle_worker_entry.py"
    payload_sha256 = None
    if repo_root is not None:
        payload = payload_archive(package_dir)
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
            package_summary(
                package_dir=package_dir,
                package_source_mode=KERNEL_PACKAGE_SOURCE_MODE,
                payload_sha256=payload_sha256,
                env=env,
                extra={
                    "kernel_id": kernel_id,
                    "title": title,
                    "code_file": metadata.get("code_file"),
                },
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


def _payload_manifest(package_dir: Path) -> dict[str, object]:
    return payload_manifest(package_dir, mode=KERNEL_PACKAGE_SOURCE_MODE)


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
