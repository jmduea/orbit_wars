#!/usr/bin/env bash
# Adapt Cursor postToolUse JSON to OMG Copilot hook format.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INPUT="$(cat)"

python3 - "$ROOT" "$INPUT" <<'PY'
import json
import subprocess
import sys

root, raw_input = sys.argv[1], sys.argv[2]
payload = json.loads(raw_input or "{}")

tool_name = payload.get("tool_name") or payload.get("toolName") or ""
tool_input = payload.get("tool_input") or payload.get("toolInput") or {}
tool_output = payload.get("tool_output") or payload.get("toolOutput") or ""
workspace = payload.get("cwd") or payload.get("workspace") or root

cursor_to_copilot = {
    "Shell": "runInTerminal",
    "Read": "readFile",
    "Write": "editFiles",
    "Delete": "deleteFile",
    "Grep": "grep",
    "Task": "task",
}
copilot_tool = cursor_to_copilot.get(tool_name, tool_name)

omg_input = {
    "toolName": copilot_tool,
    "toolInput": tool_input,
    "toolOutput": tool_output,
    "workspace": workspace,
}

subprocess.run(
    [f"{root}/.github/hooks/post-tool-use.sh"],
    input=json.dumps(omg_input),
    text=True,
    check=False,
)
PY
