# Cursor setup (Orbit Wars)

Agent orchestration uses **Cursor plugins** (user-installed), not repo-owned OMG/MCP tooling. Retired OMG files live under `docs/archive/omg/`.

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

## Governance

- No new `alwaysApply: true` rules without removing one.
- No repo-local skill/agent catalogs — use plugins.
- No implementation hooks blocking `src/` edits.

## Session-start hook (recommended)

Agents cold-start faster when session context is loaded automatically. Cursor session-start is product-level; the repo ships an optional project hook (no user secrets).

**Project hook (copy as-is):** `.cursor/hooks.json` runs `.cursor/hooks/session-start-agent-context.sh`, which calls `make agent-context` and returns `additional_context` per [Cursor hooks docs](https://cursor.com/docs/hooks). Enable by keeping those files in the repo root — Cursor reloads on save.

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

