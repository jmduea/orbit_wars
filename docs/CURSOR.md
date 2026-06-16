# Cursor setup (Orbit Wars)

Agent orchestration uses **Cursor plugins** (user-installed), not repo-owned OMG/MCP tooling. Historical OMG/MCP mirror docs were removed from the tree once Cursor plugins replaced them.

## Required plugins

Install via **Cursor Settings → Plugins**:

| Plugin | Purpose |
|--------|---------|
| [pstack](https://github.com/cursor/plugins/tree/main/pstack) | Workflow: `/poteto-mode`, `/how`, `/why`, `/interrogate`, `/architect`, `/tdd`, `/unslop` |
| [Understand-Anything](https://github.com/Lum1104/Understand-Anything) | Codebase graph: `/understand`, `/understand-chat`, `/understand-explain` |
| [continual-learning](https://github.com/cursor/plugins/tree/main/continual-learning) | Mines transcripts → updates `AGENTS.md` learned sections |

Optional: **agent-compatibility** (repo audit), **cli-for-agent** (CLI design), **cursor-team-kit** (`/deslop` referenced by poteto-mode).

Do **not** run `scripts/install_understand_anything_cursor.sh` — use native plugin install only.

## Repo-owned agent config

| Path | Role |
|------|------|
| `AGENTS.md` | Short project index; continual-learning appends learned bullets |
| `.cursor/rules/orbit-wars.mdc` | Always-on commands + test policy |
| `.cursor/rules/python-standards.mdc` | Python/Hydra coding standards (file-scoped) |
| `.cursor/rules/hydra-config.mdc` | Config editing (file-scoped) |
| `docs/ROADMAP.md` | Human priority index (not an agent gate) |
| `.cursorignore` | Hard block: secrets + personal scratch (no `@` / search) |
| `.cursorindexingignore` | Soft block: `outputs/`, caches; keeps `src/`, `conf/`, `tests/`, active `docs/` |

## Codebase indexing

Cursor respects `.gitignore` by default; this repo also ships explicit ignore files so local **`outputs/`** (~GB) and **`.venv/`** do not crowd semantic search while implementation paths stay indexed.

| File | Use |
|------|-----|
| [`.cursorindexingignore`](../.cursorindexingignore) | Training artifacts, caches, `.understand-anything/` graph JSON — still readable via `Read` / `@file` |
| [`.cursorignore`](../.cursorignore) | `.env*` and retired personal scratch paths — do not reference in chat |

**Kept in search:** `src/`, `tests/`, `conf/`, `scripts/`, `AGENTS.md`, `docs/solutions/`, `docs/brainstorms/`, `docs/benchmarks/`, `.cursor/rules/`.

**Kept discoverable under `outputs/`:** `outputs/indexes/` (e.g. `runs.jsonl` for `make agent-context`), `outputs/_meta/`.

**Kaggle parity:** `.venv/lib/python3.12/site-packages/kaggle_environments/` is re-included (negation under `.venv/`) so agents can compare `orbit_wars.py` with `src/jax/env.py` without indexing the full 4GB venv. After `uv sync`, re-index if the path is missing.

**Verify:** Cursor Settings → Features → Codebase indexing (file count should be hundreds, not thousands). `make agent-context` should still list `recent_runs` when `outputs/indexes/runs.jsonl` exists.

## Governance

- No new `alwaysApply: true` rules without removing one.
- No repo-local skill/agent catalogs — use plugins.
- No implementation hooks blocking `src/` edits.

## Session-start hook (recommended)

Agents cold-start faster when session context is loaded automatically. Cursor session-start is product-level; the repo ships an optional project hook (no user secrets).

**Project hooks (copy as-is):** `.cursor/hooks.json` per [Cursor hooks docs](https://cursor.com/docs/hooks). Cursor reloads on save.

| Hook | Script | Behavior |
|------|--------|----------|
| `sessionStart` | `.cursor/hooks/session-start-agent-context.sh` | Runs `make agent-context` → `additional_context` |
| `beforeShellExecution` | `.cursor/hooks/before-shell-terminal-contention.sh` | **Denies GPU-heavy shell** when another agent has a terminal whose `cwd` is under this repo, the terminal file still has `running_for_ms:` (no `exit_code:` footer), **and** that terminal’s `command:` is also GPU-heavy (`ow train`, `make test*`, `pytest`, `wandb agent`, calibration, etc.). Light background work (`python -m http.server`, `git status`, `make agent-context`) does not block. Design and pitfalls: [`docs/solutions/developer-experience/cursor-before-shell-gpu-terminal-contention.md`](solutions/developer-experience/cursor-before-shell-gpu-terminal-contention.md). |

Manual alternative — **Cursor Settings → Hooks → Session Start**:

```json
{
  "command": "make -C /absolute/path/to/orbit_wars agent-context"
}
```

Equivalent one-liner:

```bash
make agent-context
# or: uv run python scripts/agent_context.py
```

**Failure modes**

| Symptom | Cause | Mitigation |
|---------|-------|------------|
| Empty `recent_runs` | No `outputs/campaigns/*/runs/` yet | Run a smoke train or ignore until first campaign |
| Missing calibration excerpt | `docs/benchmarks/preflight-calibration.json` absent | Run `make preflight-calibrate` when gates matter |
| Slow hook | Large `outputs/` scan | Hook is read-only; do not run GPU train in the hook |
| GPU contention | Another session running pytest/train | Check terminals folder before starting heavy jobs |

Do **not** delete `.audit/` or `.cursor/hooks/state/` in automation.
