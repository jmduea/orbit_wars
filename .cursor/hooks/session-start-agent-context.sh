#!/usr/bin/env bash
# sessionStart hook: inject make agent-context into the agent session (fail-open).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if ! context_json="$(make -s -C "$ROOT" agent-context 2>/dev/null)"; then
  printf '%s\n' '{"additional_context":"make agent-context failed — check docs/CURSOR.md; defer GPU commands when terminals show ow train / calibrate-seed-scheduler / pytest"}'
  exit 0
fi

python3 - "$context_json" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
text = (
    "# Orbit Wars session context (make agent-context)\n\n"
    "```json\n"
    f"{json.dumps(payload, indent=2, sort_keys=True)}\n"
    "```"
)
print(json.dumps({"additional_context": text}))
PY
