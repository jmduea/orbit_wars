#!/bin/bash
# OMG Pre-Tool-Use Hook
# Runs before any tool execution in VS Code Copilot Agent Mode or Copilot CLI
#
# Input sources (auto-detected):
#   VS Code:  TOOL_NAME / TOOL_INPUT / WORKSPACE environment variables
#   CLI:      JSON via stdin with toolName / toolInput / workspace fields
#
# Output JSON: {"decision": "approve"} or {"decision": "deny", "reason": "..."}

# --- Dual-mode input detection ---
# Copilot CLI passes JSON via stdin; VS Code uses environment variables.
if [ ! -t 0 ]; then
  STDIN_DATA=$(cat)
  if [ -n "$STDIN_DATA" ]; then
    TOOL_NAME=$(printf '%s' "$STDIN_DATA" | grep -oE '"toolName"\s*:\s*"[^"]*"' | head -1 | sed 's/.*"toolName"\s*:\s*"//;s/".*//')
    TOOL_INPUT=$(printf '%s' "$STDIN_DATA" | grep -oE '"toolInput"\s*:\s*\{[^}]*\}' | head -1 | sed 's/.*"toolInput"\s*:\s*//')
    WORKSPACE=$(printf '%s' "$STDIN_DATA" | grep -oE '"workspace"\s*:\s*"[^"]*"' | head -1 | sed 's/.*"workspace"\s*:\s*"//;s/".*//')
  fi
fi

TOOL_NAME="${TOOL_NAME:-}"
TOOL_INPUT="${TOOL_INPUT:-}"

# --- Tool name normalization ---
# Map Copilot CLI tool names to VS Code equivalents so guards work on both surfaces.
case "$TOOL_NAME" in
  edit)   TOOL_NAME="editFiles" ;;
  read)   TOOL_NAME="readFile" ;;
  shell)  TOOL_NAME="runInTerminal" ;;
  create) TOOL_NAME="createFile" ;;
  delete) TOOL_NAME="deleteFile" ;;
esac

WORKSPACE="${WORKSPACE:-$(pwd)}"

# Guard: ROADMAP funnel — block src/conf/tests edits without approve-impl
if [ "$TOOL_NAME" = "editFiles" ] || [ "$TOOL_NAME" = "createFile" ] || [ "$TOOL_NAME" = "StrReplace" ] || [ "$TOOL_NAME" = "ApplyPatch" ]; then
  ROADMAP_JSON=""
  if [ -n "${STDIN_DATA:-}" ]; then
    ROADMAP_JSON=$(printf '%s' "$STDIN_DATA" | python3 "$WORKSPACE/scripts/roadmap.py" hook-check 2>/dev/null || true)
  elif [ -n "$TOOL_INPUT" ]; then
    HOOK_ARGS=()
    while IFS= read -r path; do
      [ -n "$path" ] && HOOK_ARGS+=(--path "$path")
    done < <(printf '%s' "$TOOL_INPUT" | grep -oE '(src|conf|tests)/[a-zA-Z0-9_./_-]+' | sort -u | head -8)
    if [ "${#HOOK_ARGS[@]}" -gt 0 ]; then
      ROADMAP_JSON=$(python3 "$WORKSPACE/scripts/roadmap.py" hook-check "${HOOK_ARGS[@]}" 2>/dev/null || true)
    fi
  fi
  if [ -n "$ROADMAP_JSON" ]; then
    ROADMAP_DECISION=$(printf '%s' "$ROADMAP_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin).get('decision','approve'))" 2>/dev/null || echo "approve")
    if [ "$ROADMAP_DECISION" = "deny" ]; then
      ROADMAP_REASON=$(printf '%s' "$ROADMAP_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin).get('reason','ROADMAP funnel blocked'))" 2>/dev/null || echo "ROADMAP funnel blocked")
      python3 -c 'import json,sys; print(json.dumps({"decision":"deny","reason":sys.argv[1]}))' "$ROADMAP_REASON"
      exit 0
    fi
  fi
fi

# Guard: prevent modifications to node_modules
if echo "$TOOL_INPUT" | grep -q "node_modules"; then
  if [ "$TOOL_NAME" = "editFiles" ] || [ "$TOOL_NAME" = "createFile" ]; then
    echo '{"decision": "deny", "reason": "Modifying node_modules is not allowed. Use package.json instead."}'
    exit 0
  fi
fi

# Guard: prevent modifications to .env files with secrets
if echo "$TOOL_INPUT" | grep -qE '\.env(\.local|\.production|\.secret)?'; then
  if [ "$TOOL_NAME" = "editFiles" ] || [ "$TOOL_NAME" = "createFile" ]; then
    echo '{"decision": "deny", "reason": "Direct .env file modification blocked. Review secrets manually."}'
    exit 0
  fi
fi

# Guard: prevent deletion of critical config files
if echo "$TOOL_INPUT" | grep -qE '(package\.json|tsconfig\.json|\.gitignore)'; then
  if [ "$TOOL_NAME" = "deleteFile" ]; then
    echo '{"decision": "deny", "reason": "Cannot delete critical config files."}'
    exit 0
  fi
fi

# Guard: block shell one-liners that rewrite src/conf/tests (bypass editFiles mapping)
if [ "$TOOL_NAME" = "runInTerminal" ]; then
  if echo "$TOOL_INPUT" | grep -qE 'python3?\s+-c' && echo "$TOOL_INPUT" | grep -qE '(src|conf|tests)/'; then
    ROADMAP_JSON=""
    if [ -n "${STDIN_DATA:-}" ]; then
      ROADMAP_JSON=$(printf '%s' "$STDIN_DATA" | python3 "$WORKSPACE/scripts/roadmap.py" hook-check 2>/dev/null || true)
    elif [ -n "$TOOL_INPUT" ]; then
      HOOK_ARGS=()
      while IFS= read -r path; do
        [ -n "$path" ] && HOOK_ARGS+=(--path "$path")
      done < <(printf '%s' "$TOOL_INPUT" | grep -oE '(src|conf|tests)/[a-zA-Z0-9_./_-]+' | sort -u | head -8)
      if [ "${#HOOK_ARGS[@]}" -gt 0 ]; then
        ROADMAP_JSON=$(python3 "$WORKSPACE/scripts/roadmap.py" hook-check "${HOOK_ARGS[@]}" 2>/dev/null || true)
      fi
    fi
    if [ -n "$ROADMAP_JSON" ]; then
      ROADMAP_DECISION=$(printf '%s' "$ROADMAP_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin).get('decision','approve'))" 2>/dev/null || echo "approve")
      if [ "$ROADMAP_DECISION" = "deny" ]; then
        ROADMAP_REASON=$(printf '%s' "$ROADMAP_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin).get('reason','ROADMAP funnel blocked'))" 2>/dev/null || echo "ROADMAP funnel blocked")
        python3 -c 'import json,sys; print(json.dumps({"decision":"deny","reason":sys.argv[1]}))' "$ROADMAP_REASON"
        exit 0
      fi
    fi
  fi
fi

# Guard: block push of agent issue branches (land on main instead)
if [ "$TOOL_NAME" = "runInTerminal" ]; then
  if echo "$TOOL_INPUT" | grep -qE '\bgit\s+push\b'; then
    PUSH_JSON=""
    if [ -n "${STDIN_DATA:-}" ]; then
      PUSH_JSON=$(printf '%s' "$TOOL_INPUT" | python3 "$WORKSPACE/scripts/roadmap.py" push-guard --repo-root "$WORKSPACE" 2>/dev/null || true)
    else
      PUSH_JSON=$(printf '%s' "$TOOL_INPUT" | python3 "$WORKSPACE/scripts/roadmap.py" push-guard --repo-root "$WORKSPACE" 2>/dev/null || true)
    fi
    if [ -n "$PUSH_JSON" ]; then
      PUSH_DECISION=$(printf '%s' "$PUSH_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin).get('decision','approve'))" 2>/dev/null || echo "approve")
      if [ "$PUSH_DECISION" = "deny" ]; then
        PUSH_REASON=$(printf '%s' "$PUSH_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin).get('reason','issue branch push blocked'))" 2>/dev/null || echo "issue branch push blocked")
        python3 -c 'import json,sys; print(json.dumps({"decision":"deny","reason":sys.argv[1]}))' "$PUSH_REASON"
        exit 0
      fi
    fi
  fi
fi

# Guard: prevent force push
if [ "$TOOL_NAME" = "runInTerminal" ]; then
  if echo "$TOOL_INPUT" | grep -qE 'git\s+push\s+.*(--force([^a-zA-Z0-9_-]|$)|-f([^a-zA-Z0-9_-]|$))' && ! echo "$TOOL_INPUT" | grep -qE '\-\-force-with-lease'; then
    echo '{"decision": "deny", "reason": "Force push is not allowed. Use --force-with-lease if necessary."}'
    exit 0
  fi
fi

# Guard: prevent destructive git operations
if [ "$TOOL_NAME" = "runInTerminal" ]; then
  if echo "$TOOL_INPUT" | grep -qE 'git\s+(reset\s+.*--hard|clean\s+.*-[a-z]*f|checkout\s+--\s+\.)'; then
    echo '{"decision": "deny", "reason": "Destructive git operations require manual confirmation."}'
    exit 0
  fi
fi

# Default: approve (with optional checkpoint advisory)
CHECKPOINT_TRIGGER="$WORKSPACE/.omg/state/checkpoint-trigger.json"

if [ -f "$CHECKPOINT_TRIGGER" ]; then
  echo '{"decision": "approve", "advisory": "⚠️ Context threshold reached. Call omg_checkpoint to save session state before continuing."}'
else
  echo '{"decision": "approve"}'
fi
