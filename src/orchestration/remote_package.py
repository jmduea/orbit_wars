from __future__ import annotations

import io
import json
import shutil
import tarfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

REMOTE_PACKAGE_SOURCE_MODE = "remote-tarball-v1"
DEFAULT_MAP_POOL_RELATIVE = Path("data/jax_map_pool/default_v1.npz")

DEFAULT_KAGGLE_SCRIPTS: tuple[str, ...] = (
    "kaggle_worker_entry.py",
    "benchmark_jax_rl.py",
    "kaggle_runtime_env.py",
)
DEFAULT_COLAB_SCRIPTS: tuple[str, ...] = (
    "colab_worker_entry.py",
    "kaggle_worker_entry.py",
    "benchmark_jax_rl.py",
    "kaggle_runtime_env.py",
)

_COPY_IGNORE = shutil.ignore_patterns(
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    "outputs",
    ".git",
    ".venv",
)


@dataclass(frozen=True, slots=True)
class RemotePackageOptions:
    """Options for copying a repo tree into a remote worker package directory."""

    scripts: tuple[str, ...] = DEFAULT_COLAB_SCRIPTS
    accelerator: str | None = None
    strip_jax_for_nvidia_gpu: bool = False
    include_map_pool: bool = True
    hydra_overrides: tuple[str, ...] = ()


@dataclass(slots=True)
class RemoteTarballResult:
    """Summary of a rendered remote tarball package."""

    tarball_path: Path
    package_dir: Path
    worker_env_path: Path
    payload_sha256: str
    manifest: dict[str, object]
    warnings: list[str] = field(default_factory=list)


def copy_repo_payload(
    *,
    repo_root: Path,
    package_dir: Path,
    options: RemotePackageOptions | None = None,
) -> list[str]:
    """Copy repo sources into ``package_dir`` for remote worker bootstrap.

    Returns:
        Warning strings (for example missing map pool when ``task=map_pool``).
    """

    opts = options or RemotePackageOptions()
    package_dir.mkdir(parents=True, exist_ok=True)
    payload_warnings: list[str] = []

    for name in ("src", "conf"):
        target = package_dir / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(repo_root / name, target, ignore=_COPY_IGNORE)

    scripts_target = package_dir / "scripts"
    if scripts_target.exists():
        shutil.rmtree(scripts_target)
    scripts_target.mkdir(parents=True, exist_ok=True)
    for script in opts.scripts:
        source = repo_root / "scripts" / script
        if source.exists():
            shutil.copyfile(source, scripts_target / script)

    pyproject = repo_root / "pyproject.toml"
    copied_lock = True
    if pyproject.exists():
        copied_lock = _copy_pyproject(
            source=pyproject,
            target=package_dir / "pyproject.toml",
            strip_jax_for_nvidia_gpu=opts.strip_jax_for_nvidia_gpu
            and _is_nvidia_accelerator(opts.accelerator),
        )
    if copied_lock:
        lock = repo_root / "uv.lock"
        if lock.exists():
            shutil.copyfile(lock, package_dir / "uv.lock")

    readme = repo_root / "README.md"
    if readme.exists():
        shutil.copyfile(readme, package_dir / "README.md")

    if opts.include_map_pool:
        map_pool_source = repo_root / DEFAULT_MAP_POOL_RELATIVE
        if map_pool_source.is_file():
            map_pool_target = package_dir / DEFAULT_MAP_POOL_RELATIVE
            map_pool_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(map_pool_source, map_pool_target)
        else:
            warning = _map_pool_warning(opts.hydra_overrides, map_pool_source)
            if warning:
                payload_warnings.append(warning)
                warnings.warn(warning, stacklevel=2)

    return payload_warnings


def remove_managed_payload(package_dir: Path, *, extra_paths: tuple[str, ...] = ()) -> None:
    """Remove prior managed package contents before re-rendering."""

    managed_dirs = ("src", "conf", "scripts", "data")
    managed_files = (
        "kernel-metadata.json",
        "worker-env.json",
        "package-summary.json",
        "pyproject.toml",
        "uv.lock",
        "README.md",
        *extra_paths,
    )
    for name in managed_dirs:
        target = package_dir / name
        if target.exists():
            shutil.rmtree(target)
    for name in managed_files:
        target = package_dir / name
        if target.exists():
            target.unlink()


def write_worker_env(*, package_dir: Path, env: Mapping[str, str | list[str]]) -> Path:
    """Write ``worker-env.json`` under ``package_dir``."""

    env_path = package_dir / "worker-env.json"
    env_path.write_text(json.dumps(dict(env), indent=2) + "\n", encoding="utf-8")
    return env_path


def worker_env_with_hydra_overrides(
    overrides: list[str],
    *,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str | list[str]]:
    """Build worker env payload with a JSON-serializable Hydra override list."""

    payload: dict[str, str | list[str]] = {"HYDRA_OVERRIDES": list(overrides)}
    if extra:
        payload.update(extra)
    return payload


def hydra_overrides_from_worker_env(env: Mapping[str, object]) -> list[str]:
    """Parse ``HYDRA_OVERRIDES`` from a loaded worker-env payload."""

    raw = env.get("HYDRA_OVERRIDES")
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [item.strip() for item in text.split(",") if item.strip()]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
        raise ValueError("HYDRA_OVERRIDES must be a JSON list of override strings.")
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item).strip()]
    raise ValueError("HYDRA_OVERRIDES must be a JSON list of override strings.")


def payload_manifest(
    package_dir: Path,
    *,
    mode: str = REMOTE_PACKAGE_SOURCE_MODE,
) -> dict[str, object]:
    """Summarize files included in a remote package directory."""

    included_roots = (
        "src",
        "conf",
        "scripts",
        "data",
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
        "mode": mode,
        "file_count": len(files),
        "has_worker": "scripts/kaggle_worker_entry.py" in files
        or "scripts/colab_worker_entry.py" in files,
        "has_kaggle_jax": "src/orchestration/kaggle_jax.py" in files,
        "has_map_pool": str(DEFAULT_MAP_POOL_RELATIVE) in files,
        "sample": files[:25],
    }


def payload_archive(
    package_dir: Path,
    *,
    extra_roots: tuple[str, ...] = (),
) -> bytes:
    """Build a gzip tarball of managed package roots."""

    included = (
        "src",
        "conf",
        "scripts",
        "data",
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "worker-env.json",
        *extra_roots,
    )
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in included:
            path = package_dir / name
            if path.exists():
                archive.add(path, arcname=name, recursive=True)
    return buffer.getvalue()


def tarball_member_names(payload: bytes) -> list[str]:
    """Return member paths from a gzip tarball payload."""

    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        return sorted(archive.getnames())


def package_summary(
    *,
    package_dir: Path,
    package_source_mode: str,
    payload_sha256: str | None,
    env: Mapping[str, str | list[str]],
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a redacted package summary for operator inspection."""

    summary: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_source_mode": package_source_mode,
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
    if extra:
        summary.update(extra)
    return summary


def render_remote_tarball(
    *,
    repo_root: Path,
    package_dir: Path,
    tarball_path: Path,
    env: Mapping[str, str | list[str]],
    options: RemotePackageOptions | None = None,
) -> RemoteTarballResult:
    """Render a Colab-style tarball plus ``worker-env.json`` under ``package_dir``."""

    import hashlib

    opts = options or RemotePackageOptions()
    remove_managed_payload(package_dir)
    payload_warnings = copy_repo_payload(
        repo_root=repo_root,
        package_dir=package_dir,
        options=opts,
    )
    worker_env_path = write_worker_env(package_dir=package_dir, env=env)
    payload = payload_archive(package_dir)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    tarball_path.parent.mkdir(parents=True, exist_ok=True)
    tarball_path.write_bytes(payload)
    manifest = payload_manifest(package_dir)
    return RemoteTarballResult(
        tarball_path=tarball_path,
        package_dir=package_dir,
        worker_env_path=worker_env_path,
        payload_sha256=payload_sha256,
        manifest=manifest,
        warnings=payload_warnings,
    )


def _copy_pyproject(
    *,
    source: Path,
    target: Path,
    strip_jax_for_nvidia_gpu: bool,
) -> bool:
    text = source.read_text(encoding="utf-8")
    if not strip_jax_for_nvidia_gpu:
        target.write_text(text, encoding="utf-8")
        return True

    rewritten = _strip_jax_runtime_dependencies(text)
    target.write_text(rewritten, encoding="utf-8")
    return rewritten == text


def _strip_jax_runtime_dependencies(text: str) -> str:
    """Remove JAX runtime deps from packaged GPU pyproject so uv sync does not reinstall JAX."""

    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('"jax') or stripped.startswith("'jax"):
            continue
        lines.append(line)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _is_nvidia_accelerator(accelerator: str | None) -> bool:
    return bool(accelerator and accelerator.strip().lower().startswith("nvidia"))


def _map_pool_warning(hydra_overrides: tuple[str, ...], map_pool_source: Path) -> str | None:
    if not _requests_map_pool(hydra_overrides):
        return None
    return (
        f"task=map_pool requested but map pool file is missing: {map_pool_source}. "
        "Remote training will fail without the baked pool artifact."
    )


def _requests_map_pool(hydra_overrides: tuple[str, ...]) -> bool:
    for item in hydra_overrides:
        if item.strip() == "task=map_pool":
            return True
        if item.startswith("task.map_pool_path="):
            return True
    return False


def _redact_if_secret(key: str, value: object) -> object:
    secret_markers = ("KEY", "TOKEN", "SECRET", "PASSWORD", "API_KEY")
    upper_key = key.upper()
    if any(marker in upper_key for marker in secret_markers):
        return "<redacted>"
    return value
