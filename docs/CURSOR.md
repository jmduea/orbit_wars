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
