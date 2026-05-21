#!/bin/bash
# OMG Stop Hook
# Runs when the agent is about to finish. Emits a commit reminder when worktree
# changes are present, without creating commits automatically.

if [ ! -t 0 ]; then
  STDIN_DATA=$(cat)
  if [ -n "$STDIN_DATA" ]; then
    WORKSPACE=$(printf '%s' "$STDIN_DATA" | grep -oE '"workspace"\s*:\s*"[^"]*"' | head -1 | sed 's/.*"workspace"\s*:\s*"//;s/".*//')
  fi
fi

WORKSPACE="${WORKSPACE:-$(pwd)}"

if ! command -v git >/dev/null 2>&1; then
  echo '{"decision":"approve"}'
  exit 0
fi

if ! git -C "$WORKSPACE" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo '{"decision":"approve"}'
  exit 0
fi

if [ -n "$(git -C "$WORKSPACE" status --porcelain 2>/dev/null)" ]; then
  echo '{"decision":"approve","advisory":"Reminder: the worktree has uncommitted changes. Commit your work before closing the task."}'
else
  echo '{"decision":"approve"}'
fi