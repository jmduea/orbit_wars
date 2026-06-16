# Documentation index

Canonical map of doc types, top-level folders, and navigation chains for Orbit Wars.
Update this index when adding a new top-level folder or changing doc-type policy.

## Start here (human)

1. [ONBOARDING.md](ONBOARDING.md) — codebase tour, hotspots, verification matrix
2. [AGENT_CAPABILITIES.md](AGENT_CAPABILITIES.md) — operator task prompts and capability map
3. [CURSOR.md](CURSOR.md) — Cursor plugins and session-start hooks
4. [ROADMAP.md](ROADMAP.md) — human priorities (Now / Next / Done)

## Agent policy chain

1. [`AGENTS.md`](../AGENTS.md) — repo root agent guide (commands, test tiers, invariants)
2. **`docs/README.md`** (this file) — doc-type map and folder index
3. [AGENT_CAPABILITIES.md](AGENT_CAPABILITIES.md) — task-specific CLI prompts
4. [solutions/](solutions/) — resolved bugs, patterns, and shipped design learnings (search by category frontmatter)
5. [brainstorms/](brainstorms/) — durable requirements specs (SSOT spine, Gate 5 tournament)

## Config → Kaggle pipeline (SSOT)

Canonical requirements (supersedes legacy preflight / hybrid / bracket doc paths pending full SSOT landing):

- [brainstorms/2026-06-03-training-pipeline-ssot-requirements.md](brainstorms/2026-06-03-training-pipeline-ssot-requirements.md) — single pipeline spine, teardown policy
- [solutions/architecture-patterns/ssot-training-pipeline-config-to-kaggle-submission.md](solutions/architecture-patterns/ssot-training-pipeline-config-to-kaggle-submission.md) — shipped learnings and operator map
- [competition/COMPETITION_OVERVIEW.md](competition/COMPETITION_OVERVIEW.md) — game rules and scoring
- [competition/COMPETITION_SUBMISSION.md](competition/COMPETITION_SUBMISSION.md) — agent packaging and submission

Tracker: GitHub [#205](https://github.com/jmduea/orbit_wars/issues/205). Interactive spine: [tools/ssot-training-pipeline-flowchart.html](tools/ssot-training-pipeline-flowchart.html).

## Doc-type lifecycle

| Type | Folder | When to write | Typical next step |
| --- | --- | --- | --- |
| Requirements | [brainstorms/](brainstorms/) | Problem + acceptance before code | implement → record in `solutions/` |
| Solution | [solutions/](solutions/) | Resolved bug/pattern with YAML frontmatter | link from `AGENTS.md` if durable |
| Architecture | [architecture/](architecture/) | Stable subsystem design | update when code owner changes |
| Benchmarks | [benchmarks/](benchmarks/) | Committed calibration JSON + runbook MD | never invent thresholds (see `AGENTS.md` preflight section) |

Historical implementation plans and ideation notes were removed from the tree once their learnings landed in `solutions/` — keep new docs in the folders listed above only.

## Top-level folders

| Folder | Purpose |
| --- | --- |
| [architecture/](architecture/) | Stage-level design notes with curated index ([architecture/README.md](architecture/README.md)) |
| [benchmarks/](benchmarks/) | Committed calibration JSON and benchmark runbooks; see [benchmarks/README.md](benchmarks/README.md) |
| [brainstorms/](brainstorms/) | Durable requirements specs (SSOT, Gate 5) |
| [competition/](competition/) | Kaggle competition rules and submission packaging |
| [solutions/](solutions/) | Documented learnings organized by category |
| [tools/](tools/) | Local maintainer tools and runbooks (SSOT flowchart, config picker); see [tools/README.md](tools/README.md) |

## Root evergreen docs

Long-lived references at `docs/` root (not folder indexes):

- [operator-runbook.md](operator-runbook.md) — operator workflows and run commands
- [colab_runner.md](colab_runner.md) — Colab long-run train host (`ow train colab`)
- [feature-encoding-v2.md](feature-encoding-v2.md) — planet-edge observation encoding spec
- [hydra_migration.md](hydra_migration.md) — Hydra/config migration history
- [kaggle_runner.md](kaggle_runner.md) — Kaggle notebook / submission runner notes
- [adding-observation-features.md](adding-observation-features.md) — guide for extending feature encoding
- [agent-native-phase3-status.md](agent-native-phase3-status.md) — shipped benchmark/gate/sweep CLI primitives
- [nomenclature-rfc.md](nomenclature-rfc.md) — user-facing term mappings

Repo root: [COLAB_LAUNCH_AND_INTEGRATION_PROMOTION.md](../COLAB_LAUNCH_AND_INTEGRATION_PROMOTION.md) — live Colab pilot tracker; [CONCEPTS.md](../CONCEPTS.md) — glossary.

## Maintenance

When adding a stage architecture doc, update [architecture/README.md](architecture/README.md) per that folder's index pattern.
When changing doc-type policy or navigation chains, update this file and run `make test-fast` (includes `tests/test_docs_navigation.py`).
