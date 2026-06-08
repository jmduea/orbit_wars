"""Colab training runner orchestration (package, launch, status, sync)."""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.orchestration.colab_cli import ColabCli, ColabCliError, parse_session_slug_from_text
from src.orchestration.remote_package import (
    REMOTE_PACKAGE_SOURCE_MODE,
    RemotePackageOptions,
    package_summary,
    render_remote_tarball,
    worker_env_with_hydra_overrides,
)
from src.orchestration.wandb_sweeps import (
    hydra_overrides_from_wandb_config,
    shortlist_from_api,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = REPO_ROOT / "outputs/colab_runner/kernel"
DEFAULT_LEDGER = REPO_ROOT / "outputs/colab_runner/launches.jsonl"
DEFAULT_SESSIONS = REPO_ROOT / "outputs/colab_runner/sessions.json"
DEFAULT_SYNC_DIR = REPO_ROOT / "outputs/colab_runner/synced"
DEFAULT_SHORTLIST = REPO_ROOT / "outputs/colab_runner/shortlist.json"
REMOTE_TARBALL_NAME = "orbit_wars.tgz"
REMOTE_TARBALL_PATH = f"/content/{REMOTE_TARBALL_NAME}"
REMOTE_WORKDIR = "/content/orbit_wars"
VALID_GPUS = frozenset({"T4", "L4", "A100", "H100", "V5E1", "V6E1"})
DEFAULT_GPU = "T4"
DEFAULT_LAUNCH_TIMEOUT = 86400


@dataclass(slots=True)
class ColabRequest:
    """Colab worker package and launch options."""

    work_dir: Path = field(default_factory=lambda: DEFAULT_WORK_DIR)
    gpu: str = DEFAULT_GPU
    timeout: int = DEFAULT_LAUNCH_TIMEOUT
    hydra_overrides: list[str] = field(default_factory=list)
    dry_run: bool = False
    ledger: Path = field(default_factory=lambda: DEFAULT_LEDGER)
    sessions_path: Path = field(default_factory=lambda: DEFAULT_SESSIONS)
    sync_dir: Path = field(default_factory=lambda: DEFAULT_SYNC_DIR)
    session: str | None = None
    project: str = "orbit_wars"
    entity: str | None = None
    sweep_id: str | None = None
    limit: int = 10
    shortlist_path: Path | None = None
    from_shortlist: Path | None = None
    rank: int = 0
    output: Path | None = None
    force: bool = False
    trust_base_jax: str = "0"
    wandb_api_key: str | None = None

    @classmethod
    def from_namespace(cls, args: Any) -> "ColabRequest":
        data = {f.name: getattr(args, f.name, None) for f in cls.__dataclass_fields__.values()}
        if data.get("hydra_overrides") is None:
            data["hydra_overrides"] = []
        if data.get("work_dir") is None:
            data["work_dir"] = DEFAULT_WORK_DIR
        if data.get("ledger") is None:
            data["ledger"] = DEFAULT_LEDGER
        if data.get("sessions_path") is None:
            data["sessions_path"] = DEFAULT_SESSIONS
        if data.get("sync_dir") is None:
            data["sync_dir"] = DEFAULT_SYNC_DIR
        if data.get("gpu") is None:
            data["gpu"] = DEFAULT_GPU
        if data.get("timeout") is None:
            data["timeout"] = DEFAULT_LAUNCH_TIMEOUT
        if data.get("trust_base_jax") is None:
            data["trust_base_jax"] = "0"
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _validate_hydra_overrides(overrides: list[str]) -> None:
    if not overrides:
        return
    from src.config import validate_hydra_overrides

    validate_hydra_overrides(overrides)


def _campaign_from_overrides(overrides: list[str]) -> str:
    for item in overrides:
        if item.startswith("output.campaign="):
            return item.split("=", 1)[1].strip() or "colab_long"
    return "colab_long"


def _git_short_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "nogit"
    if completed.returncode != 0:
        return "nogit"
    return (completed.stdout or "").strip() or "nogit"


def default_session_slug(overrides: list[str]) -> str:
    campaign = _campaign_from_overrides(overrides)
    safe_campaign = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in campaign)
    return f"ow-{safe_campaign}-{_git_short_sha()}"


def _shortlist_payload(row: Any) -> dict[str, Any]:
    overrides = hydra_overrides_from_wandb_config(dict(row.config))
    return {
        "run_id": row.run_id,
        "name": row.name,
        "state": row.state,
        "checkpoint_artifact": row.checkpoint_artifact,
        "checkpoint_artifact_version": row.checkpoint_artifact_version,
        "checkpoint_artifact_aliases": list(row.checkpoint_artifact_aliases),
        "metrics": dict(row.metrics),
        "score": row.score,
        "hydra_overrides": list(overrides),
        "config": dict(row.config),
    }


def _replace_request(request: ColabRequest, **updates: object) -> ColabRequest:
    data = {field: getattr(request, field) for field in ColabRequest.__dataclass_fields__}
    data.update(updates)
    return ColabRequest(**data)


def _merge_shortlist_overrides(
    request: ColabRequest,
) -> list[str]:
    overrides = list(request.hydra_overrides)
    shortlist_file = request.from_shortlist
    if shortlist_file is None:
        return overrides
    rows = _load_shortlist_rows(shortlist_file)
    rank = int(request.rank)
    if rank < 0 or rank >= len(rows):
        raise SystemExit(f"shortlist rank {rank} out of range (rows={len(rows)}).")
    row = rows[rank]
    merged = list(row.get("hydra_overrides") or [])
    if not merged and isinstance(row.get("config"), dict):
        merged = list(hydra_overrides_from_wandb_config(row["config"]))
    seen = set(merged)
    for item in overrides:
        key = item.split("=", 1)[0]
        merged = [existing for existing in merged if not existing.startswith(f"{key}=")]
        merged.append(item)
        seen.add(item)
    return merged


def _load_shortlist_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"shortlist must be a JSON list: {path}")
    return [row for row in payload if isinstance(row, dict)]


def _worker_env(
    overrides: list[str],
    *,
    trust_base_jax: str,
    wandb_api_key: str | None,
) -> dict[str, str | list[str]]:
    extra: dict[str, str] = {
        "ORBIT_WARS_COLAB_WORKER_MODE": "standalone",
        "ORBIT_WARS_COLAB_TRUST_BASE_JAX": trust_base_jax,
    }
    if wandb_api_key:
        extra["WANDB_API_KEY"] = wandb_api_key
    elif os.environ.get("WANDB_API_KEY"):
        extra["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    return worker_env_with_hydra_overrides(overrides, extra=extra)


def prepare_package(
    request: ColabRequest,
    *,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Render tarball package and summary JSON under ``work_dir``."""

    merged = list(overrides if overrides is not None else request.hydra_overrides)
    _validate_hydra_overrides(merged)
    package_dir = request.work_dir
    tarball_path = package_dir / REMOTE_TARBALL_NAME
    env = _worker_env(
        merged,
        trust_base_jax=request.trust_base_jax,
        wandb_api_key=request.wandb_api_key,
    )
    result = render_remote_tarball(
        repo_root=REPO_ROOT,
        package_dir=package_dir,
        tarball_path=tarball_path,
        env=env,
        options=RemotePackageOptions(
            hydra_overrides=tuple(merged),
            scripts=(
                "colab_worker_entry.py",
                "kaggle_worker_entry.py",
                "benchmark_jax_rl.py",
                "kaggle_runtime_env.py",
            ),
        ),
    )
    summary = package_summary(
        package_dir=package_dir,
        package_source_mode=REMOTE_PACKAGE_SOURCE_MODE,
        payload_sha256=result.payload_sha256,
        env=env,
        extra={
            "gpu": request.gpu,
            "session_slug": default_session_slug(merged),
            "hydra_overrides": merged,
            "remote_tarball_path": REMOTE_TARBALL_PATH,
            "remote_workdir": REMOTE_WORKDIR,
            "warnings": result.warnings,
            "manifest": result.manifest,
        },
    )
    summary_path = package_dir / "package-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {
        "package_dir": str(package_dir),
        "tarball_path": str(result.tarball_path),
        "worker_env_path": str(result.worker_env_path),
        "summary_path": str(summary_path),
        "payload_sha256": result.payload_sha256,
        "session_slug": summary["session_slug"],
        "gpu": request.gpu,
        "hydra_overrides": merged,
        "warnings": result.warnings,
    }


def preflight(request: ColabRequest, *, cli: ColabCli | None = None) -> dict[str, Any]:
    """Run Colab CLI and workspace preflight checks."""

    checks: list[dict[str, str]] = []
    colab = cli or ColabCli()
    try:
        executable = colab.resolve_executable()
        _record_check(checks, "colab_cli", "ok", executable)
    except ColabCliError as exc:
        _record_check(checks, "colab_cli", "error", str(exc))
        return {"ok": False, "checks": checks}

    try:
        version = colab.version()
        _record_check(checks, "colab_version", "ok", version)
    except ColabCliError as exc:
        _record_check(checks, "colab_version", "warning", str(exc))

    auth = colab.auth_check(timeout=30)
    if auth.returncode == 0:
        _record_check(checks, "colab_auth", "ok", "colab sessions succeeded.")
    else:
        detail = (auth.stderr or auth.stdout or "").strip()
        _record_check(checks, "colab_auth", "error", detail or "colab sessions failed.")

    gpu = request.gpu.strip().upper()
    if gpu in VALID_GPUS:
        _record_check(checks, "gpu_type", "ok", gpu)
    else:
        _record_check(
            checks,
            "gpu_type",
            "warning",
            f"GPU {gpu!r} not in known set {sorted(VALID_GPUS)}; passing through to colab new.",
        )

    try:
        request.work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=request.work_dir, prefix=".preflight-", delete=True):
            pass
        _record_check(checks, "package_dir_writable", "ok", str(request.work_dir))
    except OSError as exc:
        _record_check(checks, "package_dir_writable", "error", str(exc))

    merged = _merge_shortlist_overrides(request)
    if merged:
        try:
            _validate_hydra_overrides(merged)
            _record_check(checks, "hydra_overrides", "ok", f"{len(merged)} override(s)")
        except Exception as exc:
            _record_check(checks, "hydra_overrides", "error", str(exc))

    return {
        "ok": not any(check["status"] == "error" for check in checks),
        "checks": checks,
    }


def launch(request: ColabRequest, *, cli: ColabCli | None = None) -> dict[str, Any]:
    """Provision a Colab session, upload payload, and exec the worker bootstrap."""

    colab = cli or ColabCli()
    merged = _merge_shortlist_overrides(request)
    request = _replace_request(request, hydra_overrides=merged)
    preflight_payload = preflight(request, cli=colab)
    if not preflight_payload["ok"]:
        print(json.dumps(preflight_payload, indent=2, sort_keys=True))
        raise SystemExit(1)

    package = prepare_package(request, overrides=merged)
    session_slug = str(package["session_slug"])

    if request.dry_run:
        payload = {
            "dry_run": True,
            "session_slug": session_slug,
            "gpu": request.gpu,
            "timeout": request.timeout,
            "tarball_path": package["tarball_path"],
            "hydra_overrides": merged,
            "remote_tarball_path": REMOTE_TARBALL_PATH,
            "remote_workdir": REMOTE_WORKDIR,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload

    new_result = colab.new(session_slug, gpu=request.gpu)
    if new_result.returncode != 0:
        detail = (new_result.stderr or new_result.stdout or "").strip()
        raise SystemExit(f"colab new failed: {detail}")

    resolved_slug = parse_session_slug_from_text(
        (new_result.stdout or "") + "\n" + (new_result.stderr or "")
    ) or session_slug

    upload_result = colab.upload(
        resolved_slug,
        Path(package["tarball_path"]),
        REMOTE_TARBALL_PATH,
    )
    if upload_result.returncode != 0:
        detail = (upload_result.stderr or upload_result.stdout or "").strip()
        raise SystemExit(f"colab upload failed: {detail}")

    bootstrap = _bootstrap_script()
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    ) as handle:
        handle.write(bootstrap)
        bootstrap_path = Path(handle.name)

    try:
        exec_result = colab.exec(
            resolved_slug,
            command="",
            timeout=request.timeout,
            local_file=bootstrap_path,
        )
    finally:
        bootstrap_path.unlink(missing_ok=True)

    record = {
        "event": "launch",
        "session": resolved_slug,
        "gpu": request.gpu,
        "timeout": request.timeout,
        "package_dir": package["package_dir"],
        "tarball_path": package["tarball_path"],
        "hydra_overrides": merged,
        "returncode": exec_result.returncode,
        "stdout_tail": _tail(exec_result.stdout),
        "stderr_tail": _tail(exec_result.stderr),
    }
    _append_ledger(request.ledger, record)
    _upsert_session(
        request.sessions_path,
        resolved_slug,
        {
            "session": resolved_slug,
            "campaign": _campaign_from_overrides(merged),
            "gpu": request.gpu,
            "remote_workdir": REMOTE_WORKDIR,
            "package_dir": package["package_dir"],
            "hydra_overrides": merged,
            "last_event": "launch",
            "last_returncode": exec_result.returncode,
        },
    )
    payload = {
        "session": resolved_slug,
        "returncode": exec_result.returncode,
        "remote_workdir": REMOTE_WORKDIR,
        "sync_hint": f"ow train colab sync --session {resolved_slug}",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if exec_result.returncode != 0:
        raise SystemExit(exec_result.returncode)
    return payload


def status(request: ColabRequest, *, cli: ColabCli | None = None) -> dict[str, Any]:
    """Return session status plus the latest ledger event."""

    if not request.session:
        raise SystemExit("--session is required for colab status.")
    colab = cli or ColabCli()
    info = colab.status(request.session)
    last_ledger = _last_ledger_event(request.ledger, session=request.session)
    payload = {
        "session": request.session,
        "status": info.normalized,
        "raw": info.raw,
        "returncode": info.returncode,
        "last_ledger_event": last_ledger,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    _append_ledger(
        request.ledger,
        {
            "event": "status",
            "session": request.session,
            "status": info.normalized,
            "returncode": info.returncode,
            "raw_tail": _tail(info.raw),
        },
    )
    return payload


def sync(request: ColabRequest, *, cli: ColabCli | None = None) -> int:
    """Download remote campaign outputs into ``outputs/colab_runner/synced/``."""

    if not request.session:
        raise SystemExit("--session is required for colab sync.")
    colab = cli or ColabCli()
    sessions = _load_sessions(request.sessions_path)
    session_meta = sessions.get(request.session, {})
    campaign = session_meta.get("campaign") or _campaign_from_overrides(
        list(session_meta.get("hydra_overrides") or [])
    )
    remote_campaign = f"{REMOTE_WORKDIR}/outputs/campaigns/{campaign}"
    local_target = request.sync_dir / campaign
    local_target.mkdir(parents=True, exist_ok=True)
    remote_tar = f"/content/{campaign}_sync.tgz"
    archive_script = _sync_archive_script(campaign=campaign, remote_tar=remote_tar)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    ) as handle:
        handle.write(archive_script)
        archive_path = Path(handle.name)
    try:
        archive_result = colab.exec(
            request.session,
            command="",
            timeout=120,
            local_file=archive_path,
        )
    finally:
        archive_path.unlink(missing_ok=True)
    if archive_result.returncode != 0:
        detail = (archive_result.stderr or archive_result.stdout or "").strip()
        raise SystemExit(f"colab sync archive failed: {detail}")

    local_tar = request.sync_dir / f"{campaign}.sync.tgz"
    result = colab.download(
        request.session,
        remote_tar,
        local_tar,
        timeout=max(request.timeout, 600),
    )
    if result.returncode == 0 and local_tar.is_file():
        local_target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(local_tar, "r:gz") as archive:
            archive.extractall(local_target)
        local_tar.unlink(missing_ok=True)

    summary_result = colab.download(
        request.session,
        f"{REMOTE_WORKDIR}/worker-summary.json",
        local_target / "worker-summary.json",
        timeout=120,
    )
    returncode = int(result.returncode)
    if returncode == 0 and summary_result.returncode != 0:
        returncode = int(summary_result.returncode)
    payload = {
        "session": request.session,
        "campaign": campaign,
        "remote_path": remote_campaign,
        "local_path": str(local_target),
        "returncode": returncode,
        "stdout_tail": _tail(result.stdout),
        "stderr_tail": _tail(result.stderr),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    _append_ledger(
        request.ledger,
        {
            "event": "sync",
            "session": request.session,
            "campaign": campaign,
            "local_path": str(local_target),
            "returncode": returncode,
        },
    )
    return returncode


def stop(request: ColabRequest, *, cli: ColabCli | None = None) -> int:
    """Stop a Colab session; idempotent when already stopped."""

    if not request.session:
        raise SystemExit("--session is required for colab stop.")
    colab = cli or ColabCli()
    info = colab.status(request.session)
    if info.normalized == "stopped":
        payload = {"session": request.session, "status": "stopped", "idempotent": True}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    result = colab.stop(request.session)
    payload = {
        "session": request.session,
        "returncode": result.returncode,
        "stdout_tail": _tail(result.stdout),
        "stderr_tail": _tail(result.stderr),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    _append_ledger(
        request.ledger,
        {
            "event": "stop",
            "session": request.session,
            "returncode": result.returncode,
        },
    )
    return int(result.returncode)


def shortlist(request: ColabRequest) -> None:
    """Fetch W&B shortlist rows and write JSON for Colab launch."""

    if not request.sweep_id:
        raise SystemExit("--sweep-id is required for colab shortlist.")
    rows = shortlist_from_api(
        project=request.project,
        entity=request.entity,
        sweep_id=request.sweep_id,
        limit=request.limit,
    )
    payload = [_shortlist_payload(row) for row in rows]
    output = request.output or DEFAULT_SHORTLIST
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def run_preflight(request: ColabRequest) -> int:
    _validate_hydra_overrides(_merge_shortlist_overrides(request))
    payload = preflight(request)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


def run_prepare(request: ColabRequest) -> None:
    merged = _merge_shortlist_overrides(request)
    payload = prepare_package(request, overrides=merged)
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_launch(request: ColabRequest) -> None:
    launch(request)


def run_status(request: ColabRequest) -> None:
    status(request)


def run_sync(request: ColabRequest) -> int:
    return sync(request)


def run_stop(request: ColabRequest) -> int:
    return stop(request)


def run_shortlist(request: ColabRequest) -> None:
    shortlist(request)


def _sync_archive_script(*, campaign: str, remote_tar: str) -> str:
    remote_campaign_dir = f"{REMOTE_WORKDIR}/outputs/campaigns/{campaign}"
    return (
        "import subprocess\n"
        "import sys\n"
        "\n"
        f'REMOTE_CAMPAIGN_DIR = "{remote_campaign_dir}"\n'
        f'REMOTE_TAR = "{remote_tar}"\n'
        "\n"
        "subprocess.run(\n"
        '    ["tar", "-czf", REMOTE_TAR, "-C", REMOTE_CAMPAIGN_DIR, "."],\n'
        "    check=True,\n"
        ")\n"
        "raise SystemExit(0)\n"
    )


def _bootstrap_script() -> str:
    # ``colab exec -f`` runs the file as Python in the Colab kernel, not shell.
    return (
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "import tarfile\n"
        "\n"
        f'REMOTE_WORKDIR = "{REMOTE_WORKDIR}"\n'
        f'REMOTE_TARBALL_PATH = "{REMOTE_TARBALL_PATH}"\n'
        "\n"
        "os.makedirs(REMOTE_WORKDIR, exist_ok=True)\n"
        'with tarfile.open(REMOTE_TARBALL_PATH, "r:gz") as archive:\n'
        "    archive.extractall(REMOTE_WORKDIR)\n"
        "os.chdir(REMOTE_WORKDIR)\n"
        'raise SystemExit(subprocess.call([sys.executable, "scripts/colab_worker_entry.py"]))\n'
    )


def _record_check(
    checks: list[dict[str, str]],
    name: str,
    status: str,
    message: str,
) -> None:
    checks.append({"name": name, "status": status, "message": message})


def _append_ledger(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **dict(record),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _load_sessions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): dict(value)
        for key, value in payload.items()
        if isinstance(value, dict)
    }


def _save_sessions(path: Path, sessions: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sessions, indent=2) + "\n", encoding="utf-8")


def _upsert_session(
    path: Path,
    slug: str,
    record: Mapping[str, object],
) -> None:
    sessions = _load_sessions(path)
    existing = sessions.get(slug, {})
    existing.update(dict(record))
    existing["updated_at"] = datetime.now(timezone.utc).isoformat()
    sessions[slug] = existing
    _save_sessions(path, sessions)


def _last_ledger_event(path: Path, *, session: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("session") == session:
            last = payload
    return last


def _tail(text: str | None, *, limit: int = 2000) -> str:
    return (text or "").strip()[-limit:]
