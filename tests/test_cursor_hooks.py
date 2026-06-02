from __future__ import annotations

import json
import subprocess
from pathlib import Path


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
