#!/usr/bin/env python3
"""Mirror OMG catalog from .github/ into native Cursor project config (.cursor/)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
GITHUB = REPO_ROOT / ".github"
CURSOR = REPO_ROOT / ".cursor"

READONLY_HINTS = (
    "read-only",
    "plans only",
    "git only",
    "terminal only",
)

CURSOR_TOOL_MAP = {
    "Shell": "runInTerminal",
    "Read": "readFile",
    "Write": "editFiles",
    "Delete": "deleteFile",
    "Grep": "grep",
    "Task": "task",
}


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        raise ValueError("expected YAML frontmatter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("unterminated frontmatter")
    meta = yaml.safe_load(text[4:end]) or {}
    body = text[end + 5 :].lstrip("\n")
    return meta, body


def _flatten_description(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return " ".join(_flatten_description(item) for item in value)
    return str(value)


def _is_readonly(meta: dict, body: str) -> bool:
    description = _flatten_description(meta.get("description")).lower()
    if any(hint in description for hint in READONLY_HINTS):
        return True
    if description.endswith("-reviewer") or "-reviewer " in description:
        return True
    name = str(meta.get("name", "")).lower()
    if name.endswith("-reviewer"):
        return True
    tools = meta.get("tools") or []
    if isinstance(tools, list):
        normalized = {str(tool).lower() for tool in tools}
        if normalized and normalized <= {"read", "search", "omg-workflow/*"}:
            return True
    if "you are **read-only**" in body.lower():
        return True
    return False


def _cursor_agent_from_github(agent_path: Path) -> str:
    text = agent_path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(text)
    name = str(meta.get("name") or agent_path.stem.replace(".agent", ""))
    description = _flatten_description(meta.get("description"))
    readonly = _is_readonly(meta, body)
    lines = [
        "---",
        f"name: {name}",
        f"description: {description}",
        "model: inherit",
        f"readonly: {'true' if readonly else 'false'}",
        "---",
        "",
        f"> Synced from `{agent_path.relative_to(REPO_ROOT)}`. Edit `.github/agents/` and re-run `scripts/sync_omg_cursor.py`.",
        "",
        body.rstrip(),
        "",
    ]
    return "\n".join(lines)


def _link_or_copy(source: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        else:
            shutil.rmtree(dest)
    try:
        dest.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, dest)
        else:
            shutil.copy2(source, dest)


def sync_agents() -> int:
    src_dir = GITHUB / "agents"
    dest_dir = CURSOR / "agents"
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for agent_path in sorted(src_dir.glob("*.agent.md")):
        out_path = dest_dir / f"{agent_path.stem.replace('.agent', '')}.md"
        out_path.write_text(_cursor_agent_from_github(agent_path), encoding="utf-8")
        written += 1
    stale = [
        path
        for path in dest_dir.glob("*.md")
        if not (src_dir / f"{path.stem}.agent.md").exists()
    ]
    for path in stale:
        path.unlink()
    return written


def sync_skills() -> int:
    dest_root = CURSOR / "skills"
    dest_root.mkdir(parents=True, exist_ok=True)
    linked = 0
    for skill_dir in sorted((GITHUB / "skills").iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            continue
        _link_or_copy(skill_dir, dest_root / skill_dir.name)
        linked += 1
    for prompt_path in sorted((GITHUB / "prompts").glob("*.prompt.md")):
        skill_name = prompt_path.stem.replace(".prompt", "")
        dest_dir = dest_root / skill_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        _link_or_copy(prompt_path, dest_dir / "SKILL.md")
        linked += 1
    managed_names = {
        *(d.name for d in (GITHUB / "skills").iterdir() if d.is_dir()),
        *(p.stem.replace(".prompt", "") for p in (GITHUB / "prompts").glob("*.prompt.md")),
    }
    stale = [
        path
        for path in dest_root.iterdir()
        if path.name not in managed_names and not path.name.startswith("understand")
    ]
    for path in stale:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    return linked


def sync_rules() -> None:
    rules_dir = CURSOR / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    instructions = (GITHUB / "copilot-instructions.md").read_text(encoding="utf-8")
    body = instructions.replace(
        "oh-my-githubcopilot (OMG), a multi-agent orchestration layer for GitHub Copilot.",
        "oh-my-githubcopilot (OMG), a multi-agent orchestration layer. In Cursor, use project skills, subagents, hooks, and the OMG MCP server.",
    )
    rule = "\n".join(
        [
            "---",
            "description: OMG multi-agent orchestration, skills, and completion protocol",
            "alwaysApply: true",
            "---",
            "",
            body.rstrip(),
            "",
        ]
    )
    (rules_dir / "omg-orchestration.mdc").write_text(rule, encoding="utf-8")


def sync_hooks_json() -> None:
    hooks = {
        "version": 1,
        "hooks": {
            "preToolUse": [{"command": ".cursor/hooks/cursor-pre-tool-use.sh"}],
            "postToolUse": [{"command": ".cursor/hooks/cursor-post-tool-use.sh"}],
            "stop": [{"command": ".cursor/hooks/cursor-stop.sh", "loop_limit": 1}],
        },
    }
    (CURSOR / "hooks.json").write_text(
        json.dumps(hooks, indent=2) + "\n",
        encoding="utf-8",
    )


def sync_mcp() -> None:
    mcp = {
        "mcpServers": {
            "omg-workflow": {
                "command": "node",
                "args": ["${workspaceFolder}/mcp-server/dist/index.js"],
                "env": {"WORKSPACE_ROOT": "${workspaceFolder}"},
            }
        }
    }
    (CURSOR / "mcp.json").write_text(
        json.dumps(mcp, indent=2) + "\n",
        encoding="utf-8",
    )


def ensure_hook_scripts_executable() -> None:
    hooks_dir = CURSOR / "hooks"
    for path in hooks_dir.glob("*.sh"):
        path.chmod(path.stat().st_mode | 0o111)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if generated Cursor bridge files are out of date",
    )
    args = parser.parse_args()

    before_agents = {
        path.name: path.read_text(encoding="utf-8")
        for path in (CURSOR / "agents").glob("*.md")
    } if (CURSOR / "agents").is_dir() else {}

    CURSOR.mkdir(parents=True, exist_ok=True)
    agent_count = sync_agents()
    skill_count = sync_skills()
    sync_rules()
    sync_hooks_json()
    sync_mcp()
    ensure_hook_scripts_executable()

    if args.check:
        changed = False
        for path in sorted((CURSOR / "agents").glob("*.md")):
            expected = _cursor_agent_from_github(
                GITHUB / "agents" / f"{path.stem}.agent.md"
            )
            if before_agents.get(path.name) != expected and path.read_text(encoding="utf-8") != expected:
                changed = True
        if changed:
            print("Cursor bridge is out of date. Run: uv run python scripts/sync_omg_cursor.py")
            return 1
        print("Cursor bridge is up to date.")
        return 0

    print(
        f"Synced OMG → Cursor: {agent_count} agents, {skill_count} skills, "
        "rules, hooks.json, mcp.json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
