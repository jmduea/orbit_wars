#!/usr/bin/env bash
# Link Understand-Anything skills and agents into .cursor/ for native Cursor commands.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_URL="${UA_REPO_URL:-https://github.com/Lum1104/Understand-Anything.git}"
REPO_DIR="${UA_DIR:-$HOME/.understand-anything/repo}"
PLUGIN_ROOT="$REPO_DIR/understand-anything-plugin"
SKILLS_SRC="$PLUGIN_ROOT/skills"
AGENTS_SRC="$PLUGIN_ROOT/agents"
CURSOR_SKILLS="$ROOT/.cursor/skills"
CURSOR_AGENTS="$ROOT/.cursor/agents"

clone_or_update() {
  if [[ -d "$REPO_DIR/.git" ]]; then
    echo "→ Updating Understand-Anything at $REPO_DIR"
    if ! git -C "$REPO_DIR" pull --ff-only; then
      echo "  (pull skipped — using existing checkout)" >&2
    fi
  else
    echo "→ Cloning Understand-Anything to $REPO_DIR"
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone "$REPO_URL" "$REPO_DIR"
  fi
}

link_skills() {
  mkdir -p "$CURSOR_SKILLS"
  local skill
  for skill in "$SKILLS_SRC"/*/; do
    [[ -d "$skill" ]] || continue
    local name
    name="$(basename "$skill")"
    ln -sfn "$skill" "$CURSOR_SKILLS/$name"
    echo " ✓ skill $name"
  done
}

link_agents() {
  mkdir -p "$CURSOR_AGENTS"
  local agent
  for agent in "$AGENTS_SRC"/*.md; do
    [[ -f "$agent" ]] || continue
    local name
    name="$(basename "$agent" .md)"
    ln -sfn "$agent" "$CURSOR_AGENTS/$name.md"
    echo " ✓ agent $name"
  done
}

main() {
  if [[ ! -d "$ROOT/.cursor" ]]; then
    echo "Missing .cursor/. Run: uv run python scripts/sync_omg_cursor.py" >&2
    exit 1
  fi
  if ! command -v git >/dev/null 2>&1; then
    echo "git is required." >&2
    exit 1
  fi

  clone_or_update
  if [[ ! -d "$SKILLS_SRC" ]]; then
    echo "Understand-Anything plugin tree not found at $SKILLS_SRC" >&2
    exit 1
  fi

  echo "→ Linking Understand-Anything into $ROOT/.cursor"
  link_skills
  link_agents

  echo ""
  echo "Done. Restart Cursor, then use commands such as:"
  echo "  /understand"
  echo "  /understand-dashboard"
  echo "  /understand-chat How does JAX training work?"
  echo ""
  echo "Graph data: $ROOT/.understand-anything/ (already present if previously analyzed)"
}

main "$@"
