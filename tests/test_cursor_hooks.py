from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path


def _load_terminal_contention():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / ".cursor" / "hooks" / "terminal_contention.py"
    spec = importlib.util.spec_from_file_location("terminal_contention", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_session_start_hook_emits_additional_context() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    hook = repo_root / ".cursor" / "hooks" / "session-start-agent-context.sh"
    assert hook.is_file(), "session-start hook script must exist"

    proc = subprocess.run(
        [str(hook)],
        input='{"session_id":"test"}\n',
        text=True,
        capture_output=True,
        cwd=repo_root,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "additional_context" in payload
    assert isinstance(payload["additional_context"], str)
    assert payload["additional_context"]


def test_hooks_json_declares_session_start() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    hooks_path = repo_root / ".cursor" / "hooks.json"
    config = json.loads(hooks_path.read_text(encoding="utf-8"))
    assert config.get("version") == 1
    session_hooks = config.get("hooks", {}).get("sessionStart", [])
    assert session_hooks
    assert ".cursor/hooks/session-start-agent-context.sh" in session_hooks[0]["command"]


def test_hooks_json_declares_before_shell_terminal_contention() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    hooks_path = repo_root / ".cursor" / "hooks.json"
    config = json.loads(hooks_path.read_text(encoding="utf-8"))
    shell_hooks = config.get("hooks", {}).get("beforeShellExecution", [])
    assert shell_hooks
    assert (
        ".cursor/hooks/before-shell-terminal-contention.sh" in shell_hooks[0]["command"]
    )


def _run_before_shell_hook(
    repo_root: Path,
    *,
    command: str,
    fake_home: Path,
    terminal_files: dict[str, str] | None = None,
) -> dict:
    hook = repo_root / ".cursor" / "hooks" / "before-shell-terminal-contention.sh"
    project_terminals = (
        fake_home / ".cursor" / "projects" / "test-project" / "terminals"
    )
    project_terminals.mkdir(parents=True, exist_ok=True)
    cwd = str(repo_root.resolve())
    for name, body in (terminal_files or {}).items():
        if "cwd:" not in body:
            body = body.replace("---\n", f'---\ncwd: "{cwd}"\n', 1)
        (project_terminals / name).write_text(body, encoding="utf-8")

    env = os.environ.copy()
    env["HOME"] = str(fake_home)

    proc = subprocess.run(
        [str(hook)],
        input=json.dumps(
            {
                "command": command,
                "workspace_roots": [str(repo_root)],
                "hook_event_name": "beforeShellExecution",
            }
        )
        + "\n",
        text=True,
        capture_output=True,
        cwd=repo_root,
        check=False,
        timeout=10,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_before_shell_allows_when_no_active_terminals(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    payload = _run_before_shell_hook(
        repo_root,
        command="make test-fast",
        fake_home=tmp_path / "home-empty",
    )
    assert payload["permission"] == "allow"


def test_before_shell_denies_heavy_command_when_terminal_active(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tc = _load_terminal_contention()
    fake_home = tmp_path / "home-busy"
    cwd = str(repo_root.resolve())
    project_terminals = (
        fake_home / ".cursor" / "projects" / "test-project" / "terminals"
    )
    project_terminals.mkdir(parents=True)
    (project_terminals / "99.txt").write_text(
        f"---\n"
        f'cwd: "{cwd}"\n'
        f"pid: 1\n"
        f'command: "uv run ow train training.total_updates=10"\n'
        f"running_for_ms: 5000\n"
        f"---\n",
        encoding="utf-8",
    )
    payload = tc.evaluate("make test-fast", repo_root, home=fake_home)
    assert payload["permission"] == "deny"
    assert "GPU contention" in payload["user_message"]
    assert "ow train" in payload["agent_message"]


def test_before_shell_allows_heavy_when_terminal_has_command_but_not_running(
    tmp_path: Path,
) -> None:
    """Stale files may retain command: without running_for_ms — must not block."""
    repo_root = Path(__file__).resolve().parents[1]
    tc = _load_terminal_contention()
    fake_home = tmp_path / "home-stale"
    cwd = str(repo_root.resolve())
    project_terminals = (
        fake_home / ".cursor" / "projects" / "test-project" / "terminals"
    )
    project_terminals.mkdir(parents=True)
    (project_terminals / "stale.txt").write_text(
        f'---\ncwd: "{cwd}"\npid: 1\ncommand: "make test-fast"\n---\n',
        encoding="utf-8",
    )
    payload = tc.evaluate("make test-fast", repo_root, home=fake_home)
    assert payload["permission"] == "allow"


def test_before_shell_denies_via_shell_wrapper_when_terminal_active(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    payload = _run_before_shell_hook(
        repo_root,
        command="make test-fast",
        fake_home=tmp_path / "home-busy-shell",
        terminal_files={
            "99.txt": (
                '---\npid: 1\ncommand: "uv run ow train"\nrunning_for_ms: 5000\n---\n'
            )
        },
    )
    assert payload["permission"] == "deny"
    assert "GPU contention" in payload["user_message"]


def test_before_shell_allows_agent_context_when_terminal_active(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tc = _load_terminal_contention()
    fake_home = tmp_path / "home-busy-ctx"
    cwd = str(repo_root.resolve())
    project_terminals = (
        fake_home / ".cursor" / "projects" / "test-project" / "terminals"
    )
    project_terminals.mkdir(parents=True)
    (project_terminals / "1.txt").write_text(
        f'---\ncwd: "{cwd}"\npid: 1\ncommand: "pytest"\nrunning_for_ms: 1000\n---\n',
        encoding="utf-8",
    )
    payload = tc.evaluate("make agent-context", repo_root, home=fake_home)
    assert payload["permission"] == "allow"
