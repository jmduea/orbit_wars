# Documentation index

Canonical map of doc types, top-level folders, and navigation chains for Orbit Wars.
Update this index when adding a new top-level folder or changing doc-type policy.

## Start here (human)

1. [ONBOARDING.md](ONBOARDING.md) — codebase tour, hotspots, verification matrix
2. [AGENT_CAPABILITIES.md](AGENT_CAPABILITIES.md) — operator task prompts and capability map
3. [CURSOR.md](CURSOR.md) — Cursor plugins and session-start hooks
4. [ROADMAP.md](ROADMAP.md) — human priorities (Now / Next / Done). When Now/Next are empty, see Done and live trackers: GitHub [#205](https://github.com/jmduea/orbit_wars/issues/205) (SSOT pipeline), [cherry-pick manifest](benchmarks/cherry-pick-manifest.json) (admission picks).

## Agent policy chain

1. [`AGENTS.md`](../AGENTS.md) — repo root agent guide (commands, test tiers, invariants)
2. **`docs/README.md`** (this file) — doc-type map and folder index
3. [AGENT_CAPABILITIES.md](AGENT_CAPABILITIES.md) — task-specific CLI prompts
4. [solutions/](solutions/) — resolved bugs and patterns (search by category frontmatter)
5. [plans/](plans/) — active and completed implementation plans

## Config → Kaggle pipeline (SSOT)

Canonical requirements (supersedes parallel preflight / hybrid / bracket doc paths pending implementation):

- [brainstorms/2026-06-03-training-pipeline-ssot-requirements.md](brainstorms/2026-06-03-training-pipeline-ssot-requirements.md) — single pipeline spine, teardown policy
- [competition/COMPETITION_OVERVIEW.md](competition/COMPETITION_OVERVIEW.md) — game rules and scoring
- [competition/COMPETITION_SUBMISSION.md](competition/COMPETITION_SUBMISSION.md) — agent packaging and submission

Tracker: GitHub [#205](https://github.com/jmduea/orbit_wars/issues/205). Interactive spine: [tools/ssot-training-pipeline-flowchart.html](tools/ssot-training-pipeline-flowchart.html).

## Doc-type lifecycle

| Type | Folder | When to write | Typical next step |
| --- | --- | --- | --- |
| Ideation | [ideation/](ideation/) | Unscoped exploration | brainstorm or discard |
| Requirements | [brainstorms/](brainstorms/) | Problem + acceptance before code | `/ce-plan` → `plans/` |
| Plan | [plans/](plans/) | How-to-build with units | implementation → `solutions/` on resolve |
| Solution | [solutions/](solutions/) | Resolved bug/pattern with frontmatter | link from `AGENTS.md` if durable |
| Architecture | [architecture/](architecture/) | Stable subsystem design | update when code owner changes |
| Benchmarks | [benchmarks/](benchmarks/) | Committed calibration JSON + runbook MD | never invent thresholds (see `AGENTS.md` preflight section) |
| Audits | [audits/](audits/) | Point-in-time reviews | link from this index; do not duplicate in ONBOARDING |
| Session handoff | [session-handoff/](session-handoff/) | Ephemeral operator notes during multi-session work | move to `archive/session-handoff/` after originating plan merges |

## Top-level folders

| Folder | Purpose |
| --- | --- |
| [architecture/](architecture/) | Stage-level design notes with curated index ([architecture/README.md](architecture/README.md)) |
| [archive/](archive/) | Retired docs and historical mirrors (e.g. OMG/MCP) |
| [audits/](audits/) | Point-in-time architecture and agent-native reviews; operator ship tracker: [audits/agent-native-status.md](audits/agent-native-status.md) |
| [benchmarks/](benchmarks/) | Committed calibration JSON and benchmark runbooks; see [benchmarks/README.md](benchmarks/README.md) |
| [brainstorms/](brainstorms/) | Requirements-style docs before planning |
| [competition/](competition/) | Kaggle competition rules and submission packaging (SSOT for lane A/C) |
| [ideation/](ideation/) | Early exploration before scoped requirements |
| [plans/](plans/) | Dated feat/fix implementation plans with units |
| [solutions/](solutions/) | Documented learnings organized by category; see [solutions/README.md](solutions/README.md) |
| [session-handoff/](session-handoff/) | Ephemeral operator session notes (not Start here); archive when work lands |
| [tools/](tools/) | Local maintainer HTML tools (SSOT flowchart, config picker); see [tools/README.md](tools/README.md) |

## Root evergreen docs

Long-lived references at `docs/` root (not folder indexes):

- [operator-runbook.md](operator-runbook.md) — operator workflows and run commands
- [feature-encoding-v2.md](feature-encoding-v2.md) — planet-edge observation encoding spec
- [hydra_migration.md](hydra_migration.md) — Hydra/config migration history
- [adding-observation-features.md](adding-observation-features.md) — guide for extending feature encoding

## Maintenance

When adding a stage architecture doc, update [architecture/README.md](architecture/README.md) per that folder's index pattern.
When changing doc-type policy or navigation chains, update this file and verify links with `tests/test_docs_navigation.py`.

### Retention policy

| Tier | Examples | Rule |
| --- | --- | --- |
| **Immutable** | `docs/benchmarks/*.json` gate sources, committed calibration artifacts | Never delete; recalibrate before changing thresholds |
| **Pipeline artifacts** | `docs/brainstorms/`, `docs/plans/`, `docs/solutions/` | Keep; update links and frontmatter on supersession — do not delete |
| **Archivable** | Manual issue snapshots, superseded status docs, session handoffs after merge | Move to `docs/archive/` with dated names |
| **Off-model** | `docs/archive/omg/`, archived snapshots | Listed in `.cursorignore`; grep may still find paths |

Live issue tracking: GitHub issues and [ROADMAP.md](ROADMAP.md). Historical issue dump: [archive/issues-snapshot-2026-06.md](archive/issues-snapshot-2026-06.md).
