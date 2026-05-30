"""Kaggle training runner orchestration (package, launch, status, sync)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.orchestration.kaggle_cli import (
    KaggleCli,
    KaggleKernelRef,
    kaggle_push_supports_secret_flag,
)
from src.orchestration.kernel_package import render_kernel_package
from src.orchestration.population import AcceleratorPreference
from src.orchestration.wandb_sweeps import (
    add_population_metadata,
    create_sweep,
    load_sweep_config,
    shortlist_from_api,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP = REPO_ROOT / "conf/sweeps/wandb/kaggle_runner_mvp.yaml"
DEFAULT_WORK_DIR = REPO_ROOT / "outputs/kaggle_runner/kernel"
DEFAULT_LEDGER = REPO_ROOT / "outputs/kaggle_runner/launches.jsonl"
WORKER_SOURCE = REPO_ROOT / "scripts/kaggle_worker_entry.py"
LAUNCH_DIAGNOSTICS_VERSION = "launch-diagnostics-v1"


@dataclass(slots=True)
class PackageRequest:
    """Worker package and launch options shared across runner commands."""

    work_dir: Path
    kernel_id: str
    title: str
    sweep_yaml: Path
    run_type: str | None = None
    no_wandb: bool = False
    standalone_overrides: list[str] = field(default_factory=list)
    calibration_max_variants: int | None = None
    calibration_warmup: int | None = None
    calibration_updates: int | None = None
    calibration_timeout_seconds: int | None = None
    project: str | None = None
    entity: str | None = None
    sweep_id: str | None = None
    dry_run: bool = False
    create_sweep: bool = False
    timeout_seconds: int = 43200
    accelerators: tuple[str, ...] = ()
    ledger: Path = field(default_factory=lambda: DEFAULT_LEDGER)
    no_accelerator_flag_fallback: bool = False
    output_dir: Path | None = None
    force: bool = False
    kernel_ref: str | None = None
    limit: int = 10
    run_id: str | None = None

    @classmethod
    def from_namespace(cls, args: Any) -> "PackageRequest":
        """Build a request from an argparse namespace."""

        data = {f.name: getattr(args, f.name, None) for f in cls.__dataclass_fields__.values()}
        if data.get("standalone_overrides") is None:
            data["standalone_overrides"] = []
        acc = getattr(args, "accelerators", None)
        if acc:
            data["accelerators"] = tuple(acc)
        elif "accelerators" not in data or data["accelerators"] is None:
            data["accelerators"] = ()
        if data.get("ledger") is None:
            data["ledger"] = DEFAULT_LEDGER
        if data.get("work_dir") is None:
            data["work_dir"] = DEFAULT_WORK_DIR
        if data.get("sweep_yaml") is None:
            data["sweep_yaml"] = DEFAULT_SWEEP
        if data.get("kernel_id") is None:
            data["kernel_id"] = default_kernel_id()
        if data.get("title") is None:
            data["title"] = "orbit wars kaggle runner"
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})




def prepare(
    args: PackageRequest | Any, *, sweep_id: str | None, accelerator: str | None = None
):
    standalone = bool(getattr(args, "no_wandb", False))
    env: dict[str, str] = {
        "WANDB_SWEEP_YAML": _packaged_sweep_path(args.sweep_yaml),
    }
    if standalone:
        env["ORBIT_WARS_KAGGLE_WORKER_MODE"] = "standalone"
    else:
        env["WANDB_SWEEP_ID"] = sweep_id or ""
    project = getattr(args, "project", None)
    entity = getattr(args, "entity", None)
    if project and not standalone:
        env["WANDB_PROJECT"] = str(project)
    if entity and not standalone:
        env["WANDB_ENTITY"] = str(entity)
    if accelerator:
        env["KAGGLE_ACCELERATOR_ID"] = str(accelerator)
    env["ORBIT_WARS_WORKER_VENV"] = os.environ.get(
        "ORBIT_WARS_KAGGLE_WORKER_VENV", "/tmp/orbit_wars_worker_venv"
    )
    if not standalone:
        env["WANDB_API_KEY_SECRET_NAME"] = os.environ.get(
            "ORBIT_WARS_KAGGLE_WANDB_SECRET_NAME", "WANDB_API_KEY"
        )
    env["ORBIT_WARS_KAGGLE_TRUST_BASE_JAX"] = os.environ.get(
        "ORBIT_WARS_KAGGLE_TRUST_BASE_JAX", "1"
    )
    run_type = getattr(args, "run_type", None) or os.environ.get(
        "ORBIT_WARS_KAGGLE_RUN_TYPE", ""
    ).strip()
    if run_type:
        env["ORBIT_WARS_KAGGLE_RUN_TYPE"] = run_type
    _apply_run_type_calibration_defaults(env, run_type)
    _apply_calibration_cli_args(env, args)
    standalone_overrides = list(getattr(args, "standalone_overrides", []) or [])
    if standalone_overrides:
        env["ORBIT_WARS_KAGGLE_STANDALONE_OVERRIDES"] = json.dumps(
            standalone_overrides
        )
    return render_kernel_package(
        package_dir=args.work_dir,
        kernel_id=args.kernel_id,
        title=args.title,
        worker_source=WORKER_SOURCE,
        env=env,
        repo_root=REPO_ROOT,
        accelerator=accelerator,
    )


def _apply_run_type_calibration_defaults(env: dict[str, str], run_type: str) -> None:
    """Set calibration env defaults from run-type without clobbering explicit values."""

    if run_type == "smoke":
        env.setdefault("ORBIT_WARS_KAGGLE_CALIBRATION_MAX_VARIANTS", "1")
        env.setdefault("ORBIT_WARS_KAGGLE_CALIBRATION_UPDATES", "1")
        env.setdefault("ORBIT_WARS_KAGGLE_ALLOW_CALIBRATION_FALLBACK", "1")
        return
    if run_type == "benchmark":
        env.setdefault("ORBIT_WARS_KAGGLE_CALIBRATION_MAX_VARIANTS", "3")
        env.setdefault("ORBIT_WARS_KAGGLE_CALIBRATION_WARMUP", "2")
        env.setdefault("ORBIT_WARS_KAGGLE_CALIBRATION_UPDATES", "30")
        env.setdefault("ORBIT_WARS_KAGGLE_CALIBRATION_TIMEOUT_SECONDS", "3600")
        env.setdefault("ORBIT_WARS_KAGGLE_ALLOW_CALIBRATION_FALLBACK", "1")


def _apply_calibration_cli_args(env: dict[str, str], args: PackageRequest | Any) -> None:
    """Pack explicit calibration CLI flags into worker-env.json."""

    mapping = (
        ("calibration_max_variants", "ORBIT_WARS_KAGGLE_CALIBRATION_MAX_VARIANTS"),
        ("calibration_warmup", "ORBIT_WARS_KAGGLE_CALIBRATION_WARMUP"),
        ("calibration_updates", "ORBIT_WARS_KAGGLE_CALIBRATION_UPDATES"),
        (
            "calibration_timeout_seconds",
            "ORBIT_WARS_KAGGLE_CALIBRATION_TIMEOUT_SECONDS",
        ),
    )
    for attr, key in mapping:
        value = getattr(args, attr, None)
        if value is not None:
            env[key] = str(int(value))


def _packaged_sweep_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def launch(args: PackageRequest | Any, accelerators: tuple[str, ...]) -> None:
    """Prepare and push one package per accelerator with full diagnostics.

    The previous launcher prepared once using the first accelerator and then only
    rewrote worker-env.json for retries. That made failures hard to interpret and
    allowed metadata/runtime intent to diverge. This version regenerates the
    package per accelerator and prints every push attempt as structured JSON.
    """

    attempts: list[dict[str, Any]] = []
    allow_flag_fallback = not bool(
        getattr(args, "no_accelerator_flag_fallback", False)
    )
    allow_flag_fallback = _env_flag(
        "ORBIT_WARS_KAGGLE_ACCELERATOR_FLAG_FALLBACK",
        default=allow_flag_fallback,
    )

    for accelerator in accelerators:
        package = prepare(args, sweep_id=args.sweep_id, accelerator=accelerator)
        result = _push_kernel(
            package.package_dir,
            accelerator=accelerator,
            timeout_seconds=args.timeout_seconds,
            use_accelerator_flag=True,
            dry_run=bool(args.dry_run),
            attach_wandb_secret=not bool(getattr(args, "no_wandb", False)),
        )
        record = _launch_ledger_record(
            args=args,
            package_dir=package.package_dir,
            accelerator=accelerator,
            command=result["command"],
            returncode=int(result["returncode"]),
            stdout=str(result.get("stdout", "")),
            stderr=str(result.get("stderr", "")),
            dry_run=bool(args.dry_run),
            used_accelerator_flag=True,
            fallback_attempt=False,
        )
        attempts.append(record)
        _append_ledger(args.ledger, record)
        _print_launch_attempt(record)

        if int(result["returncode"]) == 0:
            _print_success(
                result,
                kernel_id=args.kernel_id,
                ledger=args.ledger,
                standalone=bool(getattr(args, "no_wandb", False)),
            )
            return

        if (
            allow_flag_fallback
            and accelerator
            and _looks_like_unsupported_accelerator_flag(
                str(result.get("stdout", "")),
                str(result.get("stderr", "")),
            )
        ):
            print(
                "Local Kaggle CLI appears not to support --accelerator; "
                "retrying without the flag while keeping metadata/worker-env GPU enabled.",
                flush=True,
            )
            fallback = _push_kernel(
                package.package_dir,
                accelerator=accelerator,
                timeout_seconds=args.timeout_seconds,
                use_accelerator_flag=False,
                dry_run=bool(args.dry_run),
                attach_wandb_secret=not bool(getattr(args, "no_wandb", False)),
            )
            fallback_record = _launch_ledger_record(
                args=args,
                package_dir=package.package_dir,
                accelerator=accelerator,
                command=fallback["command"],
                returncode=int(fallback["returncode"]),
                stdout=str(fallback.get("stdout", "")),
                stderr=str(fallback.get("stderr", "")),
                dry_run=bool(args.dry_run),
                used_accelerator_flag=False,
                fallback_attempt=True,
            )
            attempts.append(fallback_record)
            _append_ledger(args.ledger, fallback_record)
            _print_launch_attempt(fallback_record)
            if int(fallback["returncode"]) == 0:
                _print_success(
                    fallback,
                    kernel_id=args.kernel_id,
                    ledger=args.ledger,
                    standalone=bool(getattr(args, "no_wandb", False)),
                )
                return

    raise SystemExit(_format_launch_failure(attempts, args.ledger))


def _push_kernel(
    package_dir: Path,
    *,
    accelerator: str | None,
    timeout_seconds: int | None,
    use_accelerator_flag: bool,
    dry_run: bool,
    attach_wandb_secret: bool = True,
) -> dict[str, Any]:
    command = ["kaggle", "kernels", "push", "-p", str(package_dir)]
    if accelerator and use_accelerator_flag:
        command.extend(["--accelerator", accelerator])
    if timeout_seconds is not None:
        command.extend(["--timeout", str(int(timeout_seconds))])
    secret_name = os.environ.get("ORBIT_WARS_KAGGLE_WANDB_SECRET_NAME", "WANDB_API_KEY")
    if attach_wandb_secret and secret_name and kaggle_push_supports_secret_flag():
        command.extend(["--secret", secret_name])

    if dry_run:
        return {
            "command": command,
            "returncode": 0,
            "stdout": " ".join(command),
            "stderr": "",
        }

    completed = subprocess.run(
        command,
        cwd=package_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _looks_like_unsupported_accelerator_flag(stdout: str, stderr: str) -> bool:
    text = f"{stdout}\n{stderr}".lower()
    patterns = (
        "unrecognized arguments: --accelerator",
        "unrecognized argument: --accelerator",
        "no such option: --accelerator",
        "unknown option: --accelerator",
        "unknown flag: --accelerator",
        "unknown command line flag 'accelerator'",
        "invalid option: --accelerator",
    )
    return any(pattern in text for pattern in patterns)


def _print_success(
    result: dict[str, Any],
    *,
    kernel_id: str,
    ledger: Path,
    standalone: bool = False,
) -> None:
    output = str(result.get("stdout", "") or result.get("stderr", "")).strip()
    if output:
        print(output, flush=True)
    kernel_url = f"https://www.kaggle.com/code/{kernel_id}"
    if standalone:
        checklist = {
            "event": "launch_success_checklist",
            "worker_mode": "standalone",
            "kernel_id": kernel_id,
            "kernel_url": kernel_url,
            "required_before_training": (
                "No W&B API key or Kaggle Secret is required in standalone mode."
            ),
            "verify_output": (
                f"uv run ow train kaggle sync {kernel_id} --force"
            ),
            "inspect": (
                "Check worker-summary.json for exit_code, selected_overrides, "
                "and checkpoint_paths under outputs/"
            ),
            "ledger": str(ledger),
        }
    else:
        secret_name = os.environ.get(
            "ORBIT_WARS_KAGGLE_WANDB_SECRET_NAME", "WANDB_API_KEY"
        )
        checklist = {
            "event": "launch_success_checklist",
            "worker_mode": "wandb",
            "kernel_id": kernel_id,
            "kernel_url": kernel_url,
            "required_before_wandb": (
                f"Open the kernel editor → Add-ons → Secrets → enable secret "
                f"{secret_name!r} for this kernel. Kaggle CLI push cannot attach "
                f"secrets automatically."
            ),
            "verify_output": (
                f"uv run ow train kaggle sync {kernel_id} --force"
            ),
            "inspect": "Check worker-summary.json for wandb_secret.loaded=true",
            "ledger": str(ledger),
        }
    print(json.dumps(checklist, indent=2, sort_keys=True), flush=True)


def _print_launch_attempt(record: dict[str, Any]) -> None:
    printable = {
        "event": "launch_attempt",
        "diagnostics_version": LAUNCH_DIAGNOSTICS_VERSION,
        "accelerator": record.get("accelerator"),
        "used_accelerator_flag": record.get("used_accelerator_flag"),
        "fallback_attempt": record.get("fallback_attempt"),
        "returncode": record.get("returncode"),
        "command": record.get("command"),
        "stdout_tail": record.get("stdout_tail"),
        "stderr_tail": record.get("stderr_tail"),
        "package_dir": record.get("package_dir"),
        "kernel_id": record.get("kernel_id"),
        "ledger": str(record.get("ledger", "")),
    }
    print(json.dumps(printable, indent=2, sort_keys=True), flush=True)


def _format_launch_failure(attempts: list[dict[str, Any]], ledger: Path) -> str:
    if not attempts:
        return "No accelerator launch attempt succeeded; no launch attempts were made."
    last = attempts[-1]
    last_detail = str(last.get("stderr_tail") or last.get("stdout_tail") or "").strip()
    payload = {
        "message": "No accelerator launch attempt succeeded.",
        "attempt_count": len(attempts),
        "last_accelerator": last.get("accelerator"),
        "last_returncode": last.get("returncode"),
        "last_command": last.get("command"),
        "last_detail": last_detail,
        "ledger": str(ledger),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def preflight(args: PackageRequest | Any) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    standalone = bool(getattr(args, "no_wandb", False))
    _check_sweep(args.sweep_yaml, checks)
    _check_kernel_id(args.kernel_id, checks)
    _check_writable(args.work_dir, checks)
    _check_kaggle(args.timeout_seconds, checks)
    _check_kaggle_accelerator_flag(checks)
    if standalone:
        _record_check(
            checks,
            "standalone_mode",
            "ok",
            "Standalone mode skips W&B API and Kaggle Secret requirements.",
        )
    else:
        _check_wandb(args.project, args.entity, checks)
        _check_kaggle_secret_attachment(checks)
    if not os.environ.get("KAGGLE_USERNAME"):
        _record_check(
            checks,
            "kaggle_username_env",
            "warning",
            "KAGGLE_USERNAME is not set; pass --kernel-id explicitly if your Kaggle owner differs.",
        )
    return {
        "ok": not any(check["status"] == "error" for check in checks),
        "checks": checks,
    }


def _check_sweep(path: Path, checks: list[dict[str, Any]]) -> None:
    if not path.exists():
        _record_check(
            checks, "sweep_yaml", "error", f"sweep YAML does not exist: {path}"
        )
        return
    try:
        sweep = load_sweep_config(path)
    except Exception as exc:
        _record_check(
            checks, "sweep_yaml", "error", f"failed to parse sweep YAML: {exc}"
        )
        return
    if not isinstance(sweep, dict) or "parameters" not in sweep:
        _record_check(
            checks, "sweep_yaml", "error", "sweep YAML must contain parameters."
        )
        return
    _record_check(checks, "sweep_yaml", "ok", f"parsed {path}")


def _check_kernel_id(kernel_id: str, checks: list[dict[str, Any]]) -> None:
    try:
        ref = KaggleKernelRef.parse(kernel_id)
    except ValueError as exc:
        _record_check(checks, "kernel_id", "error", str(exc))
        return
    if ref.owner == "replace-me":
        _record_check(
            checks,
            "kernel_id",
            "error",
            "kernel ID still uses the placeholder owner replace-me.",
        )
        return
    _record_check(checks, "kernel_id", "ok", str(ref))


def _check_writable(path: Path, checks: list[dict[str, Any]]) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".preflight-", delete=True):
            pass
    except OSError as exc:
        _record_check(checks, "package_dir_writable", "error", f"{path}: {exc}")
        return
    _record_check(checks, "package_dir_writable", "ok", str(path))


def _check_kaggle(timeout_seconds: int, checks: list[dict[str, Any]]) -> None:
    executable = shutil.which("kaggle")
    if executable is None:
        _record_check(
            checks, "kaggle_cli", "error", "kaggle CLI executable not found on PATH."
        )
        return
    _record_check(checks, "kaggle_cli", "ok", executable)
    try:
        completed = subprocess.run(
            ["kaggle", "kernels", "list", "--mine"],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(int(timeout_seconds), 1),
        )
    except subprocess.TimeoutExpired:
        _record_check(checks, "kaggle_auth", "error", "kaggle auth check timed out.")
        return
    if completed.returncode != 0:
        _record_check(
            checks,
            "kaggle_auth",
            "error",
            _tail(completed.stderr or completed.stdout),
        )
        return
    _record_check(checks, "kaggle_auth", "ok", "kaggle kernels list --mine succeeded.")


def _check_kaggle_accelerator_flag(checks: list[dict[str, Any]]) -> None:
    executable = shutil.which("kaggle")
    if executable is None:
        return
    try:
        completed = subprocess.run(
            ["kaggle", "kernels", "push", "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        _record_check(
            checks,
            "kaggle_accelerator_flag",
            "warning",
            "timed out while checking whether kaggle kernels push supports --accelerator.",
        )
        return
    help_text = completed.stdout or completed.stderr or ""
    if "--accelerator" in help_text:
        _record_check(
            checks,
            "kaggle_accelerator_flag",
            "ok",
            "kaggle kernels push help includes --accelerator.",
        )
        return
    _record_check(
        checks,
        "kaggle_accelerator_flag",
        "warning",
        "kaggle kernels push help did not include --accelerator; launch will print full diagnostics and may retry metadata-only GPU push.",
    )


def _check_kaggle_secret_attachment(checks: list[dict[str, Any]]) -> None:
    """Warn that W&B secrets must be attached manually after kernel push."""

    secret_name = os.environ.get("ORBIT_WARS_KAGGLE_WANDB_SECRET_NAME", "WANDB_API_KEY")
    supports_cli_secret = kaggle_push_supports_secret_flag()
    if supports_cli_secret:
        _record_check(
            checks,
            "kaggle_secret_attach",
            "ok",
            f"Local Kaggle CLI supports --secret; launcher will pass {secret_name!r} on push.",
        )
        return
    _record_check(
        checks,
        "kaggle_secret_attach",
        "warning",
        "Kaggle CLI cannot attach secrets on kernels push. After launch, open the kernel "
        f"editor and enable secret {secret_name!r} under Add-ons → Secrets before "
        "expecting W&B runs. Re-attach if Kaggle creates a new kernel version.",
    )


def _check_wandb(
    project: str, entity: str | None, checks: list[dict[str, Any]]
) -> None:
    try:
        import wandb  # type: ignore
    except ImportError as exc:
        _record_check(checks, "wandb_import", "error", f"failed to import wandb: {exc}")
        return
    _record_check(checks, "wandb_import", "ok", "wandb import succeeded.")
    try:
        api = wandb.Api()
        api.project(project, entity=entity)
    except Exception as exc:
        _record_check(
            checks,
            "wandb_api",
            "error",
            f"failed to resolve W&B project {entity + '/' if entity else ''}{project}: {exc}",
        )
        return
    _record_check(
        checks,
        "wandb_api",
        "ok",
        f"resolved W&B project {entity + '/' if entity else ''}{project}",
    )


def _shortlist_payload(row: Any) -> dict[str, Any]:
    return {
        "run_id": row.run_id,
        "name": row.name,
        "state": row.state,
        "checkpoint_artifact": row.checkpoint_artifact,
        "checkpoint_artifact_version": row.checkpoint_artifact_version,
        "checkpoint_artifact_aliases": list(row.checkpoint_artifact_aliases),
        "metrics": dict(row.metrics),
        "score": row.score,
    }


def _launch_ledger_record(
    *,
    args: PackageRequest | Any,
    package_dir: Path,
    accelerator: str,
    command: list[str],
    returncode: int,
    stdout: str,
    stderr: str,
    dry_run: bool,
    used_accelerator_flag: bool,
    fallback_attempt: bool,
) -> dict[str, Any]:
    return {
        "event": "launch",
        "diagnostics_version": LAUNCH_DIAGNOSTICS_VERSION,
        "dry_run": dry_run,
        "package_dir": str(package_dir),
        "kernel_id": args.kernel_id,
        "accelerator": accelerator,
        "used_accelerator_flag": bool(used_accelerator_flag),
        "fallback_attempt": bool(fallback_attempt),
        "command": list(command),
        "returncode": int(returncode),
        "stdout_tail": _tail(stdout, limit=6000),
        "stderr_tail": _tail(stderr, limit=6000),
        "sweep_id": args.sweep_id,
        "ledger": str(args.ledger),
    }


def _append_ledger(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")


def _record_check(
    checks: list[dict[str, Any]],
    name: str,
    status: str,
    message: str,
) -> None:
    checks.append({"name": name, "status": status, "message": message})


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _tail(text: str | None, *, limit: int = 2000) -> str:
    return (text or "").strip()[-limit:]


def default_kernel_id() -> str:
    owner = os.environ.get("KAGGLE_USERNAME", "replace-me")
    return f"{owner}/orbit-wars-kaggle-runner"


def run_status(args: PackageRequest | Any) -> None:
    """Query kernel status and append a ledger record."""

    status = KaggleCli().status(KaggleKernelRef.parse(args.kernel_ref))
    payload = {
        "ref": str(status.ref),
        "status": status.normalized,
        "returncode": status.returncode,
        "raw": status.raw,
    }
    print(json.dumps(payload, indent=2))
    _append_ledger(
        args.ledger,
        {
            "event": "status",
            "kernel_id": str(status.ref),
            "returncode": status.returncode,
            "status": status.normalized,
            "raw_tail": _tail(status.raw),
        },
    )


def run_sync_output(args: PackageRequest | Any) -> int:
    """Download kernel output and append a ledger record."""

    output_dir = args.output_dir or (REPO_ROOT / "outputs/kaggle_runner/synced")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = KaggleCli().output(
        KaggleKernelRef.parse(args.kernel_ref),
        output_dir,
        force=args.force,
    )
    print(result.stdout or result.stderr)
    _append_ledger(
        args.ledger,
        {
            "event": "sync-output",
            "kernel_id": args.kernel_ref,
            "output_dir": str(output_dir),
            "force": bool(args.force),
            "returncode": result.returncode,
            "stdout_tail": _tail(result.stdout),
            "stderr_tail": _tail(result.stderr),
        },
    )
    return int(result.returncode)


def run_shortlist(args: PackageRequest | Any) -> None:
    """Fetch W&B shortlist and write JSON output."""

    rows = shortlist_from_api(
        project=args.project or "orbit_wars",
        entity=args.entity,
        sweep_id=args.sweep_id,
        limit=args.limit,
    )
    payload = [_shortlist_payload(row) for row in rows]
    output = getattr(args, "output", None) or (
        REPO_ROOT / "outputs/kaggle_runner/shortlist.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def run_latest_checkpoint(args: PackageRequest | Any) -> None:
    """Print the latest checkpoint row for a sweep."""

    rows = shortlist_from_api(
        project=args.project or "orbit_wars",
        entity=args.entity,
        sweep_id=args.sweep_id,
        limit=args.limit,
    )
    matches = [
        row
        for row in rows
        if row.has_checkpoint
        and (args.run_id is None or args.run_id in {row.run_id, row.name})
    ]
    if not matches:
        raise SystemExit("No checkpoint artifact found for the requested sweep/run.")
    print(json.dumps(_shortlist_payload(matches[0]), indent=2))


def run_prepare(args: PackageRequest | Any) -> None:
    """Render a worker package and print summary JSON."""

    accelerators = tuple(
        args.accelerators or AcceleratorPreference().accelerator_ids
    )
    package = prepare(args, sweep_id=args.sweep_id, accelerator=accelerators[0])
    print(
        json.dumps(
            {
                "package_dir": str(package.package_dir),
                "summary": str(package.summary_path),
                "accelerator": accelerators[0],
                "diagnostics_version": LAUNCH_DIAGNOSTICS_VERSION,
            },
            indent=2,
            sort_keys=True,
        )
    )


def run_preflight(args: PackageRequest | Any) -> int:
    """Run preflight checks and print JSON; return exit code."""

    payload = preflight(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if any(check["status"] == "error" for check in payload["checks"]):
        return 1
    return 0


def run_launch(args: PackageRequest | Any) -> None:
    """Create optional W&B sweep and launch kernel(s)."""

    if args.create_sweep and args.no_wandb:
        raise SystemExit("--create-sweep cannot be used with --no-wandb.")
    sweep_id = args.sweep_id
    if args.create_sweep:
        sweep = add_population_metadata(
            load_sweep_config(args.sweep_yaml),
            group="kaggle_runner_mvp",
            tags=("kaggle", "population"),
        )
        if args.dry_run:
            sweep_id = "dry-run-sweep"
            print(json.dumps({"would_create_sweep": sweep}, indent=2, sort_keys=True))
        else:
            sweep_id = create_sweep(
                sweep, project=args.project or "orbit_wars", entity=args.entity
            )
    args.sweep_id = sweep_id
    accelerators = tuple(args.accelerators or AcceleratorPreference().accelerator_ids)
    launch(args, accelerators)

