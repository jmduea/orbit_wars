#!/bin/bash
# OMG Post-Tool-Use Hook
# Runs after tool execution in VS Code Copilot Agent Mode or Copilot CLI
#
# Input sources (auto-detected):
#   VS Code:  TOOL_NAME / TOOL_INPUT / TOOL_OUTPUT / WORKSPACE environment variables
#   CLI:      JSON via stdin with toolName / toolInput / toolOutput / workspace fields
#
# Use for: logging, state updates, completion checks

# --- Dual-mode input detection ---
if [ ! -t 0 ]; then
  STDIN_DATA=$(cat)
  if [ -n "$STDIN_DATA" ]; then
    TOOL_NAME=$(printf '%s' "$STDIN_DATA" | grep -oE '"toolName"\s*:\s*"[^"]*"' | head -1 | sed 's/.*"toolName"\s*:\s*"//;s/".*//')
    TOOL_INPUT=$(printf '%s' "$STDIN_DATA" | grep -oE '"toolInput"\s*:\s*\{[^}]*\}' | head -1 | sed 's/.*"toolInput"\s*:\s*//')
    TOOL_OUTPUT=$(printf '%s' "$STDIN_DATA" | grep -oE '"toolOutput"\s*:\s*\{[^}]*\}' | head -1 | sed 's/.*"toolOutput"\s*:\s*//')
    WORKSPACE=$(printf '%s' "$STDIN_DATA" | grep -oE '"workspace"\s*:\s*"[^"]*"' | head -1 | sed 's/.*"workspace"\s*:\s*"//;s/".*//')
  fi
fi

TOOL_NAME="${TOOL_NAME:-}"
TOOL_INPUT="${TOOL_INPUT:-}"
TOOL_OUTPUT="${TOOL_OUTPUT:-}"
WORKSPACE="${WORKSPACE:-$(pwd)}"

# --- Tool name normalization ---
case "$TOOL_NAME" in
  edit)   TOOL_NAME="editFiles" ;;
  read)   TOOL_NAME="readFile" ;;
  shell)  TOOL_NAME="runInTerminal" ;;
  create) TOOL_NAME="createFile" ;;
  delete) TOOL_NAME="deleteFile" ;;
esac

is_file_mutation_tool() {
  case "$TOOL_NAME" in
    editFiles|createFile|apply_patch|create_file|functions.apply_patch|functions.create_file)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_safe_python_path() {
  local file_path="$1"

  [ -n "$file_path" ] || return 1

  if [[ "$file_path" =~ [\'\"\;\&\|\`\$\(\)\{\}\<\>] ]]; then
    return 1
  fi

  case "$file_path" in
    *../*|../*|*/..) return 1 ;;
    *.py) ;;
    *) return 1 ;;
  esac

  case "$file_path" in
    /*) ;;
    *) file_path="$WORKSPACE/$file_path" ;;
  esac

  case "$file_path" in
    "$WORKSPACE"/*) ;;
    *) return 1 ;;
  esac

  case "$file_path" in
    "$WORKSPACE/.venv"/*|"$WORKSPACE/outputs"/*|"$WORKSPACE/wandb"/*|"$WORKSPACE/artifacts"/*|"$WORKSPACE/.git"/*|"$WORKSPACE/.omg"/*|"$WORKSPACE/.omc"/*)
      return 1
      ;;
  esac

  [ -f "$file_path" ] || return 1
  printf '%s\n' "$file_path"
  return 0
}

collect_python_files_from_tool_input() {
  local files_file="$1"
  local normalized_input

  normalized_input=$(printf '%s' "$TOOL_INPUT" | sed 's/\\n/\
/g')

  printf '%s' "$normalized_input" \
    | grep -oE '"(filePath|path)"[[:space:]]*:[[:space:]]*"[^"]+\.py"' \
    | sed -E 's/.*"(filePath|path)"[[:space:]]*:[[:space:]]*"([^"]+)".*/\2/' \
    | while IFS= read -r file_path; do
        is_safe_python_path "$file_path" || true
      done >> "$files_file"

  printf '%s' "$normalized_input" \
    | grep -oE '\*\*\* (Add|Update) File: [^[:cntrl:]"]+\.py' \
    | sed -E 's/^\*\*\* (Add|Update) File: //' \
    | while IFS= read -r file_path; do
        is_safe_python_path "$file_path" || true
      done >> "$files_file"
}

run_ruff_for_files() {
  local files_file="$1"
  local report_file="$OMG_STATE_DIR/ruff-gate.json"
  local timestamp
  local ruff_output
  local ruff_status="ok"
  local rel_files=()
  local file_path

  [ -s "$files_file" ] || return 0

  sort -u "$files_file" -o "$files_file" 2>/dev/null || true

  while IFS= read -r file_path; do
    rel_files+=("${file_path#"$WORKSPACE/"}")
  done < "$files_file"

  [ "${#rel_files[@]}" -gt 0 ] || return 0

  timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

  if command -v uv >/dev/null 2>&1; then
    ruff_output=$(cd "$WORKSPACE" && uv run --group dev ruff check --fix -- "${rel_files[@]}" 2>&1 && uv run --group dev ruff format -- "${rel_files[@]}" 2>&1 && uv run --group dev ruff check -- "${rel_files[@]}" 2>&1)
    ruff_exit=$?
  elif command -v ruff >/dev/null 2>&1; then
    ruff_output=$(cd "$WORKSPACE" && ruff check --fix -- "${rel_files[@]}" 2>&1 && ruff format -- "${rel_files[@]}" 2>&1 && ruff check -- "${rel_files[@]}" 2>&1)
    ruff_exit=$?
  else
    ruff_status="skipped"
    ruff_output="Ruff was not found on PATH, and uv is not installed."
    ruff_exit=0
  fi

  if [ "$ruff_exit" -ne 0 ]; then
    ruff_status="failed"
  fi

  ruff_output=$(printf '%s' "$ruff_output" | head -40 | tr '"' "'" | tr '\n' ' ')
  printf '{"status":"%s","files":"%s","timestamp":"%s","details":"%s"}\n' \
    "$ruff_status" "$(printf '%s ' "${rel_files[@]}" | sed 's/[[:space:]]*$//')" "$timestamp" "$ruff_output" \
    > "$report_file" 2>/dev/null
}

OMG_STATE_DIR="$WORKSPACE/.omg/state"

# Ensure state directory exists
mkdir -p "$OMG_STATE_DIR" 2>/dev/null

# --- Context byte accumulation for pre-compaction checkpoint ---
# Tracks cumulative TOOL_INPUT + TOOL_OUTPUT bytes to estimate context window usage.
# When threshold is reached (default 400KB ≈ 100K tokens), creates a checkpoint trigger.
# Threshold is configurable via OMG_CONTEXT_THRESHOLD (bytes, default 400000).
CONTEXT_BYTES_FILE="$OMG_STATE_DIR/context-bytes.txt"
CHECKPOINT_TRIGGER="$OMG_STATE_DIR/checkpoint-trigger.json"
OMG_CONTEXT_THRESHOLD="${OMG_CONTEXT_THRESHOLD:-400000}"

# Measure bytes of this tool call's I/O
INPUT_BYTES=$(printf '%s' "$TOOL_INPUT" | wc -c | tr -d ' ')
OUTPUT_BYTES=$(printf '%s' "$TOOL_OUTPUT" | wc -c | tr -d ' ')
CALL_BYTES=$((INPUT_BYTES + OUTPUT_BYTES))

# Read current accumulation
# Note: read-modify-write is not atomic. Acceptable because VS Code hooks run serially.
ACCUMULATED=$(cat "$CONTEXT_BYTES_FILE" 2>/dev/null || echo 0)
ACCUMULATED=$((ACCUMULATED + CALL_BYTES))
echo "$ACCUMULATED" > "$CONTEXT_BYTES_FILE" 2>/dev/null

# Check if threshold reached — create checkpoint trigger
if [ "$ACCUMULATED" -ge "$OMG_CONTEXT_THRESHOLD" ] && [ ! -f "$CHECKPOINT_TRIGGER" ]; then
  ESTIMATED_TOKENS=$((ACCUMULATED / 4))
  echo "{\"checkpoint_due\": true, \"context_bytes\": $ACCUMULATED, \"estimated_tokens\": $ESTIMATED_TOKENS, \"threshold\": $OMG_CONTEXT_THRESHOLD, \"timestamp\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\"}" \
    > "$CHECKPOINT_TRIGGER" 2>/dev/null
fi

# Log tool usage for debugging (optional, enable by setting OMG_DEBUG=1)
if [ "${OMG_DEBUG:-0}" = "1" ]; then
  LOG_FILE="$OMG_STATE_DIR/tool-usage.log"
  echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $TOOL_NAME" >> "$LOG_FILE"
fi

# Track file modifications for autopilot phase tracking
if [ "$TOOL_NAME" = "editFiles" ] || [ "$TOOL_NAME" = "createFile" ]; then
  MODIFIED_FILES="$OMG_STATE_DIR/modified-files.txt"
  # Extract file path from tool input and append to tracking file
  FILE_PATH=$(echo "$TOOL_INPUT" | grep -oE '"filePath"\s*:\s*"[^"]*"' | head -1 | sed 's/.*"filePath"\s*:\s*"//;s/".*//')
  if [ -n "$FILE_PATH" ]; then
    echo "$FILE_PATH" >> "$MODIFIED_FILES" 2>/dev/null
    # Deduplicate
    if [ -f "$MODIFIED_FILES" ]; then
      sort -u "$MODIFIED_FILES" -o "$MODIFIED_FILES" 2>/dev/null
    fi
  fi
fi

# Check for test failures after terminal commands
if [ "$TOOL_NAME" = "runInTerminal" ]; then
  # If a test command was run, check for failures
  if echo "$TOOL_INPUT" | grep -qE '(npm test|jest|vitest|pytest|cargo test|go test)'; then
    if echo "$TOOL_OUTPUT" | grep -qiE '(FAIL|ERROR|failed|error)'; then
      # Write failure marker for ultraqa/autopilot to detect
      echo '{"last_test_run": "failed", "timestamp": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"}' \
        > "$OMG_STATE_DIR/last-test-result.json" 2>/dev/null
    else
      echo '{"last_test_run": "passed", "timestamp": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"}' \
        > "$OMG_STATE_DIR/last-test-result.json" 2>/dev/null
    fi
  fi
fi

# Run Ruff after Python file edits. This keeps agent-written Python formatted and
# applies safe lint fixes immediately after create/edit style tools complete.
if is_file_mutation_tool; then
  RUFF_FILES=$(mktemp 2>/dev/null || printf '/tmp/omg-ruff-files.%s' "$$")
  : > "$RUFF_FILES" 2>/dev/null || true
  collect_python_files_from_tool_input "$RUFF_FILES"
  run_ruff_for_files "$RUFF_FILES"
  rm -f "$RUFF_FILES" 2>/dev/null || true
fi

# --- Plankton: Opt-in type check + lint after file edits ---
# Enable by setting OMG_LINT_ON_EDIT=1 in your environment (opt-in, advisory only)
if [ "${OMG_LINT_ON_EDIT:-0}" = "1" ]; then
  if [ "$TOOL_NAME" = "editFiles" ] || [ "$TOOL_NAME" = "createFile" ]; then
    FILE_PATH=$(echo "$TOOL_INPUT" | grep -oE '"filePath"\s*:\s*"[^"]*"' | head -1 | sed 's/.*"filePath"\s*:\s*"//;s/".*//')

    QUALITY_REPORT="$OMG_STATE_DIR/quality-gate.json"
    QUALITY_STATUS="ok"
    QUALITY_DETAILS=""

    if [ -n "$FILE_PATH" ]; then
      # TypeScript type check (non-blocking, advisory)
      if [ -f "$WORKSPACE/tsconfig.json" ] && echo "$FILE_PATH" | grep -qE '\.(ts|tsx)$'; then
        TS_OUTPUT=$(cd "$WORKSPACE" && npx tsc --noEmit 2>&1 | head -20)
        if echo "$TS_OUTPUT" | grep -qE 'error TS'; then
          QUALITY_STATUS="type-errors"
          QUALITY_DETAILS="$TS_OUTPUT"
        fi
      fi

      # ESLint check (non-blocking, advisory)
      if ls "$WORKSPACE"/.eslintrc* "$WORKSPACE"/eslint.config.* 2>/dev/null | grep -q '.'; then
        if echo "$FILE_PATH" | grep -qE '\.(ts|tsx|js|jsx)$'; then
          # Sanitize FILE_PATH: reject paths containing shell metacharacters or traversal sequences
          if [[ "$FILE_PATH" =~ [\'\"\;\&\|\`\$\(\)\{\}\<\>] ]] || [[ "$FILE_PATH" =~ \.\./ ]]; then
            QUALITY_STATUS="invalid-path"
            QUALITY_DETAILS="FILE_PATH failed sanitization check"
          else
          LINT_OUTPUT=$(cd "$WORKSPACE" && npx eslint "$FILE_PATH" --max-warnings=0 2>&1 | head -20)
          if echo "$LINT_OUTPUT" | grep -qE 'error|warning'; then
            QUALITY_STATUS="${QUALITY_STATUS}+lint-warnings"
            QUALITY_DETAILS="${QUALITY_DETAILS}\n${LINT_OUTPUT}"
          fi
          fi  # end sanitization check
        fi
      fi

      # Write quality gate result (advisory — does NOT block tool execution)
      echo "{\"status\": \"$QUALITY_STATUS\", \"file\": \"$FILE_PATH\", \"timestamp\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\", \"details\": \"$(echo "$QUALITY_DETAILS" | tr '"' "'" | tr '\n' ' ')\"}" \
        > "$QUALITY_REPORT" 2>/dev/null
    fi
  fi
fi

exit 0
