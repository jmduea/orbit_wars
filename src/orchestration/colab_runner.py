"""Colab training runner orchestration (package, launch, status, sync)."""

from __future__ import annotations

import json
import netrc
import os
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.orchestration.colab_cli import (
    ColabCli,
    ColabCliError,
    parse_session_slug_from_text,
)
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
DEFAULT_MONITOR_DIR = REPO_ROOT / "outputs/colab_runner/monitor"
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
    monitor_dir: Path = field(default_factory=lambda: DEFAULT_MONITOR_DIR)
    monitor_after_launch: bool = False
    interval_seconds: int = 300
    stale_seconds: int = 900
    max_iterations: int | None = None
    once: bool = False
    eval_checkpoints: bool = True
    eval_baselines: str = "noop,random,sniper"
    eval_seeds: str = "0,1,2,3,4"
    eval_formats: str = "2p_vs_baseline"
    eval_games_per_pair: int = 1
    eval_max_steps: int = 500
    eval_write_replays: bool = False
    stop_on_stale: bool = False

    @classmethod
    def from_namespace(cls, args: Any) -> "ColabRequest":
        data = {
            f.name: getattr(args, f.name, None)
            for f in cls.__dataclass_fields__.values()
        }
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
        if data.get("monitor_dir") is None:
            data["monitor_dir"] = DEFAULT_MONITOR_DIR
        if data.get("monitor_after_launch") is None:
            data["monitor_after_launch"] = False
        if data.get("interval_seconds") is None:
            data["interval_seconds"] = 300
        if data.get("stale_seconds") is None:
            data["stale_seconds"] = 900
        if data.get("eval_checkpoints") is None:
            data["eval_checkpoints"] = True
        if data.get("eval_baselines") is None:
            data["eval_baselines"] = "noop,random,sniper"
        if data.get("eval_seeds") is None:
            data["eval_seeds"] = "0,1,2,3,4"
        if data.get("eval_formats") is None:
            data["eval_formats"] = "2p_vs_baseline"
        if data.get("eval_games_per_pair") is None:
            data["eval_games_per_pair"] = 1
        if data.get("eval_max_steps") is None:
            data["eval_max_steps"] = 500
        if data.get("eval_write_replays") is None:
            data["eval_write_replays"] = False
        if data.get("stop_on_stale") is None:
            data["stop_on_stale"] = False
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
    safe_campaign = "".join(
        ch if ch.isalnum() or ch in "-_" else "-" for ch in campaign
    )
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
    data = {
        field: getattr(request, field) for field in ColabRequest.__dataclass_fields__
    }
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
        "TF_GPU_ALLOCATOR": "cuda_malloc_async",
    }
    if wandb_api_key:
        extra["WANDB_API_KEY"] = wandb_api_key
    elif os.environ.get("WANDB_API_KEY"):
        extra["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    else:
        netrc_key = _wandb_api_key_from_netrc()
        if netrc_key:
            extra["WANDB_API_KEY"] = netrc_key
    return worker_env_with_hydra_overrides(overrides, extra=extra)


def _wandb_api_key_from_netrc() -> str | None:
    try:
        auth = netrc.netrc()
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return None
    for host in ("api.wandb.ai", "wandb.ai"):
        entry = auth.authenticators(host)
        if entry and entry[2]:
            return str(entry[2])
    return None


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
        with tempfile.NamedTemporaryFile(
            dir=request.work_dir, prefix=".preflight-", delete=True
        ):
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

    resolved_slug = (
        parse_session_slug_from_text(
            (new_result.stdout or "") + "\n" + (new_result.stderr or "")
        )
        or session_slug
    )

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
        "monitor_hint": f"ow train colab monitor --session {resolved_slug}",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if exec_result.returncode != 0:
        raise SystemExit(exec_result.returncode)
    if request.monitor_after_launch:
        monitor_request = _replace_request(request, session=resolved_slug)
        payload["monitor_payload"] = monitor(monitor_request, cli=colab)
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
            timeout=max(120, min(int(request.timeout), 600)),
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


def monitor(
    request: ColabRequest,
    *,
    cli: ColabCli | None = None,
    evaluator: Any | None = None,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    """Poll/sync a Colab session and evaluate newly synced checkpoints locally."""

    if not request.session:
        raise SystemExit("--session is required for colab monitor.")
    colab = cli or ColabCli()
    state = _load_monitor_state(request)
    iterations = 0
    last_payload: dict[str, Any] = {}
    while True:
        iterations += 1
        payload = monitor_once(request, cli=colab, evaluator=evaluator, state=state)
        last_payload = payload
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        if payload.get("stale") and request.stop_on_stale:
            stop(_replace_request(request, session=request.session), cli=colab)
            payload["stopped_for_stale"] = True
            last_payload = payload
            break
        if request.once:
            break
        if request.max_iterations is not None and iterations >= request.max_iterations:
            break
        if payload.get("remote_status") in {"stopped", "failed"}:
            break
        sleep(max(int(request.interval_seconds), 1))
    _save_monitor_state(request, state)
    return last_payload


def monitor_once(
    request: ColabRequest,
    *,
    cli: ColabCli,
    evaluator: Any | None = None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one monitor iteration: status, sync, stale check, checkpoint eval."""

    if not request.session:
        raise SystemExit("--session is required for colab monitor.")
    state = state if state is not None else _load_monitor_state(request)
    now = datetime.now(timezone.utc)
    status_error: str | None = None
    sync_error: str | None = None
    try:
        status_payload = status(request, cli=cli)
    except Exception as exc:
        status_error = str(exc)
        status_payload = {
            "session": request.session,
            "status": "failed",
            "returncode": 1,
            "error": status_error,
        }
    try:
        sync_returncode = sync(request, cli=cli)
    except SystemExit as exc:
        sync_returncode = int(exc.code) if isinstance(exc.code, int) else 1
        sync_error = str(exc)
    except Exception as exc:
        sync_returncode = 1
        sync_error = str(exc)
    run_dir = _latest_synced_run_dir(request)
    progress = _synced_run_progress(run_dir, now=now)
    stale_reasons = _stale_reasons(
        progress,
        stale_seconds=int(request.stale_seconds),
        remote_status=str(status_payload.get("status") or "unknown"),
    )
    eval_payloads: list[dict[str, Any]] = []
    if request.eval_checkpoints and run_dir is not None:
        for checkpoint in _new_checkpoints_for_eval(run_dir, state):
            eval_payload = _evaluate_checkpoint(
                request,
                checkpoint=checkpoint,
                evaluator=evaluator,
            )
            eval_payloads.append(eval_payload)
            if int(eval_payload["returncode"]) == 0:
                state.setdefault("evaluated_checkpoints", {})[
                    str(checkpoint.resolve())
                ] = eval_payload
            else:
                state.setdefault("failed_evaluations", {})[
                    str(checkpoint.resolve())
                ] = eval_payload
    all_stale_reasons = [
        *stale_reasons,
        *(["status_error"] if status_error else []),
        *(["sync_error"] if sync_error else []),
    ]
    payload = {
        "session": request.session,
        "remote_status": status_payload.get("status"),
        "status_error": status_error,
        "sync_returncode": sync_returncode,
        "sync_error": sync_error,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "progress": progress,
        "stale": bool(all_stale_reasons),
        "stale_reasons": all_stale_reasons,
        "evaluated": eval_payloads,
    }
    state["last_monitor_payload"] = payload
    state["updated_at"] = now.isoformat()
    _save_monitor_state(request, state)
    _append_ledger(
        request.ledger,
        {
            "event": "monitor",
            "session": request.session,
            "remote_status": payload["remote_status"],
            "sync_returncode": sync_returncode,
            "run_dir": payload["run_dir"],
            "stale": payload["stale"],
            "stale_reasons": all_stale_reasons,
            "evaluated_count": len(eval_payloads),
        },
    )
    return payload


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


def run_monitor(request: ColabRequest) -> None:
    monitor(request)


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


def _monitor_state_path(request: ColabRequest) -> Path:
    session = request.session or "unknown"
    safe_session = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in session)
    return request.monitor_dir / f"{safe_session}.json"


def _load_monitor_state(request: ColabRequest) -> dict[str, Any]:
    path = _monitor_state_path(request)
    if not path.is_file():
        return {"evaluated_checkpoints": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"evaluated_checkpoints": {}}
    if not isinstance(payload, dict):
        return {"evaluated_checkpoints": {}}
    payload.setdefault("evaluated_checkpoints", {})
    return payload


def _save_monitor_state(request: ColabRequest, state: Mapping[str, Any]) -> None:
    path = _monitor_state_path(request)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(state), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _session_campaign(request: ColabRequest) -> str:
    if not request.session:
        return _campaign_from_overrides(request.hydra_overrides)
    sessions = _load_sessions(request.sessions_path)
    session_meta = sessions.get(request.session, {})
    campaign = session_meta.get("campaign")
    if isinstance(campaign, str) and campaign:
        return campaign
    overrides = list(session_meta.get("hydra_overrides") or request.hydra_overrides)
    return _campaign_from_overrides(overrides)


def _latest_synced_run_dir(request: ColabRequest) -> Path | None:
    campaign = _session_campaign(request)
    runs_dir = request.sync_dir / campaign / "runs"
    if not runs_dir.is_dir():
        return None
    candidates = [path for path in runs_dir.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _latest_metric_row(log_path: Path) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload.get("update"), int) and "overall_win_rate" in payload:
            latest = payload
    return latest


def _synced_run_progress(run_dir: Path | None, *, now: datetime) -> dict[str, Any]:
    if run_dir is None:
        return {"ok": False, "reason": "no_synced_run"}
    logs = sorted((run_dir / "logs").glob("*_jax.jsonl"))
    checkpoints = sorted((run_dir / "checkpoints").glob("jax_ckpt_*.pkl"))
    latest_log = max(logs, key=lambda path: path.stat().st_mtime) if logs else None
    latest_checkpoint = (
        max(checkpoints, key=lambda path: path.stat().st_mtime) if checkpoints else None
    )
    latest_row = _latest_metric_row(latest_log) if latest_log is not None else None
    latest_mtime = None
    latest_source = None
    for path, source in ((latest_log, "log"), (latest_checkpoint, "checkpoint")):
        if path is None:
            continue
        mtime = path.stat().st_mtime
        if latest_mtime is None or mtime > latest_mtime:
            latest_mtime = mtime
            latest_source = source
    age_seconds = (
        None if latest_mtime is None else max(now.timestamp() - latest_mtime, 0.0)
    )
    return {
        "ok": True,
        "latest_log": str(latest_log) if latest_log is not None else None,
        "latest_checkpoint": str(latest_checkpoint)
        if latest_checkpoint is not None
        else None,
        "checkpoint_count": len(checkpoints),
        "latest_update": latest_row.get("update") if latest_row else None,
        "latest_win_rate": latest_row.get("overall_win_rate") if latest_row else None,
        "latest_preflight_sweep_score": latest_row.get("preflight_sweep_score")
        if latest_row
        else None,
        "latest_activity_source": latest_source,
        "activity_age_seconds": age_seconds,
    }


def _stale_reasons(
    progress: Mapping[str, Any],
    *,
    stale_seconds: int,
    remote_status: str,
) -> list[str]:
    reasons: list[str] = []
    if remote_status in {"stopped", "failed"}:
        reasons.append(f"remote_status_{remote_status}")
    if not progress.get("ok"):
        reasons.append(str(progress.get("reason") or "missing_progress"))
        return reasons
    if progress.get("latest_update") is None:
        reasons.append("no_metric_rows")
    age = progress.get("activity_age_seconds")
    if isinstance(age, (int, float)) and age > stale_seconds:
        reasons.append(f"activity_stale_{int(age)}s")
    return reasons


def _new_checkpoints_for_eval(run_dir: Path, state: Mapping[str, Any]) -> list[Path]:
    evaluated = set((state.get("evaluated_checkpoints") or {}).keys())
    checkpoints = sorted((run_dir / "checkpoints").glob("jax_ckpt_*.pkl"))
    result: list[Path] = []
    for checkpoint in checkpoints:
        if checkpoint.name == "jax_ckpt_last.pkl":
            continue
        if str(checkpoint.resolve()) not in evaluated:
            result.append(checkpoint)
    return result


def _evaluate_checkpoint(
    request: ColabRequest,
    *,
    checkpoint: Path,
    evaluator: Any | None,
) -> dict[str, Any]:
    output_dir = (
        request.monitor_dir / "evals" / checkpoint.parent.parent.name / checkpoint.stem
    )
    command = [
        "uv",
        "run",
        "ow",
        "eval",
        "tournament",
        "--campaign",
        "colab_monitor",
        "--output-dir",
        str(output_dir),
        "--checkpoint",
        str(checkpoint),
        "--seeds",
        request.eval_seeds,
        "--games-per-pair",
        str(int(request.eval_games_per_pair)),
        "--max-steps",
        str(int(request.eval_max_steps)),
        "--formats",
        request.eval_formats,
        "--baselines",
        request.eval_baselines,
    ]
    if request.eval_write_replays:
        command.append("--write-replays")
    if evaluator is None:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    else:
        completed = evaluator(command, output_dir)
    summary = _summarize_eval_output(output_dir)
    return {
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "returncode": int(completed.returncode),
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
        "summary": summary,
    }


def _summarize_eval_output(output_dir: Path) -> dict[str, Any]:
    matches_dir = output_dir / "matches"
    if not matches_dir.is_dir():
        return {"match_count": 0, "by_baseline": {}}
    by_baseline: dict[str, dict[str, Any]] = {}
    for path in sorted(matches_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        checkpoint_ids = [
            str(agent_id)
            for agent_id in payload.get("agent_ids", [])
            if str(agent_id).startswith("jax_ckpt")
        ]
        if not checkpoint_ids:
            continue
        checkpoint_id = checkpoint_ids[0]
        baseline = str(
            payload.get("baseline_name") or payload.get("format_name") or "unknown"
        )
        bucket = by_baseline.setdefault(
            baseline,
            {"wins": 0, "games": 0, "placements": {}},
        )
        bucket["games"] += 1
        result = (payload.get("results") or {}).get(checkpoint_id)
        if result == "win":
            bucket["wins"] += 1
        placement = (payload.get("placements") or {}).get(checkpoint_id)
        if placement is not None:
            key = str(placement)
            bucket["placements"][key] = int(bucket["placements"].get(key, 0)) + 1
    for bucket in by_baseline.values():
        games = max(int(bucket["games"]), 1)
        bucket["win_rate"] = float(bucket["wins"]) / games
    return {
        "match_count": sum(int(bucket["games"]) for bucket in by_baseline.values()),
        "by_baseline": by_baseline,
    }


def _bootstrap_script() -> str:
    # ``colab exec -f`` runs the file as Python in the Colab kernel, not shell.
    return (
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "import tarfile\n"
        "from pathlib import Path\n"
        "\n"
        f'REMOTE_WORKDIR = "{REMOTE_WORKDIR}"\n'
        f'REMOTE_TARBALL_PATH = "{REMOTE_TARBALL_PATH}"\n'
        "\n"
        "os.makedirs(REMOTE_WORKDIR, exist_ok=True)\n"
        'with tarfile.open(REMOTE_TARBALL_PATH, "r:gz") as archive:\n'
        "    archive.extractall(REMOTE_WORKDIR)\n"
        "os.chdir(REMOTE_WORKDIR)\n"
        'stdout = open("colab-worker.stdout.log", "ab", buffering=0)\n'
        'stderr = open("colab-worker.stderr.log", "ab", buffering=0)\n'
        "process = subprocess.Popen(\n"
        '    [sys.executable, "scripts/colab_worker_entry.py"],\n'
        "    stdout=stdout,\n"
        "    stderr=stderr,\n"
        "    start_new_session=True,\n"
        ")\n"
        'Path("colab-worker.pid").write_text(str(process.pid) + "\\n", encoding="utf-8")\n'
        'print(f"colab worker started pid={process.pid}", flush=True)\n'
        "raise SystemExit(0)\n"
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
