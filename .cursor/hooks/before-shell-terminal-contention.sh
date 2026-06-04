#!/usr/bin/env bash
# beforeShellExecution: block GPU-heavy shell when another agent terminal is active.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python3 "${HOOK_DIR}/terminal_contention.py" "$ROOT"
