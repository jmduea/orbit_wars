#!/usr/bin/env bash
# Adapt OMG stop hook advisory output to Cursor followup_message when useful.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INPUT="$(cat)"

python3 - "$ROOT" "$INPUT" <<'PY'
import json
import subprocess
import sys

root, raw_input = sys.argv[1], sys.argv[2]
payload = json.loads(raw_input or "{}")
workspace = payload.get("cwd") or payload.get("workspace") or root

proc = subprocess.run(
    [f"{root}/.github/hooks/stop.sh"],
    input=json.dumps({"workspace": workspace}),
    text=True,
    capture_output=True,
    check=False,
)
response = proc.stdout.strip() or '{"decision":"approve"}'
try:
    data = json.loads(response)
except json.JSONDecodeError:
    print("{}")
    raise SystemExit(0)

advisory = data.get("advisory")
if advisory:
    print(json.dumps({"followup_message": advisory}))
else:
    print("{}")
PY
