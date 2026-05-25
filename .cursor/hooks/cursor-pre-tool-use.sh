#!/usr/bin/env bash
# Adapt Cursor preToolUse JSON to OMG Copilot hook format and translate the response.
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
    "workspace": workspace,
}

proc = subprocess.run(
    [f"{root}/.github/hooks/pre-tool-use.sh"],
    input=json.dumps(omg_input),
    text=True,
    capture_output=True,
    check=False,
)
response = proc.stdout.strip() or '{"decision":"approve"}'
try:
    data = json.loads(response)
except json.JSONDecodeError:
    print('{"permission":"allow"}')
    raise SystemExit(0)

decision = str(data.get("decision", "approve")).lower()
permission = "deny" if decision == "deny" else "allow"
out = {"permission": permission}
reason = data.get("reason") or data.get("advisory")
if reason:
    out["agent_message"] = reason
    if permission == "deny":
        out["user_message"] = reason
print(json.dumps(out))
PY
