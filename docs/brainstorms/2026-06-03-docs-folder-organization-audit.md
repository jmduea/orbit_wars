---
status: partial-shipped
date: 2026-06-03
scope: docs/ information architecture (not a single feature plan)
actors:
  - human-maintainer
  - coding-agent (AGENTS.md + CURSOR session-start)
  - new-contributor (ONBOARDING path)
---

# Docs folder organization audit

## Shipped status (2026-06-06 re-baseline)

**P0/P1 navigation IA shipped** via plans `2026-06-03-010` and `2026-06-04-007`:

- `docs/README.md` — canonical doc-type map and folder index
- `AGENTS.md` pointer to `docs/README.md`
- ONBOARDING § Documentation — hand-maintained blocks, zero broken stale links (A3)
- `docs/benchmarks/README.md` — committed calibration manifest (A4a)
- `scripts/agent_context.py` `docs.readme` key (F1 agent-context chain)

**Remaining work** (this audit's open scope for consolidation/cleanup):

- P2 **R4** — relocate or archive root clutter (`Issues.md`, phase-status trackers, retired stubs)
- Optional stub READMEs in `plans/`, `brainstorms/`, `solutions/` — **deferred** (central `docs/README.md` suffices; see plan 007 KTD1)
- **Retention policy** — explicit rules for archive/delete vs immutable pipeline artifacts (benchmarks JSON)
- **`session-handoff/`** — classify in doc-type lifecycle and index in `docs/README.md`

## Problem frame (2026-06-03 baseline)

The `docs/` tree had grown to **267+ files** across **eight** top-level folders plus loose root markdown with **no `docs/README.md` index**. Entry points overlapped and some cross-links were stale. Agents cold-started via `make agent-context` and `AGENTS.md`, yet discoverability of **plans vs brainstorms vs solutions vs ideation vs audits** was implicit.

**Current tree (2026-06-06):** **331 files** across **eleven** top-level folders (`architecture/`, `archive/`, `audits/`, `benchmarks/`, `brainstorms/`, `competition/`, `ideation/`, `plans/`, `session-handoff/`, `solutions/`, `tools/`) plus **17 loose markdown files at `docs/` root**. Navigation indexes now exist; the remaining pain is **stale root clutter**, **uncategorized ephemeral docs** (`session-handoff/`), and **search noise** from `archive/omg/` (~148 files) — not missing READMEs.

This audit proposes clearer information architecture **and** explicit retirement rules. **Immutable pipeline artifacts** (committed benchmarks JSON, gate calibration sources) must not be deleted; **manual snapshots, retired stubs, and superseded status docs** are eligible for archive or delete.

## Current layout (inventory)

| Location | Count / role | Discoverability |
|----------|--------------|-----------------|
| `docs/` root (loose `.md`) | 17 files incl. ONBOARDING, ROADMAP, AGENT_CAPABILITIES, phase-status trackers, Issues.md | Indexed via `docs/README.md` Root evergreen; mixed evergreen vs status vs retired |
| `docs/plans/` | 52 dated feat/fix plans | Filename convention; central index in `docs/README.md` |
| `docs/brainstorms/` | 12 requirements-style docs | Same |
| `docs/solutions/` | 28 learnings in category subdirs | Referenced from AGENTS.md; no `solutions/README.md` stub |
| `docs/ideation/` | Exploratory docs | Defined in `docs/README.md` lifecycle table |
| `docs/architecture/` | Stage docs + README index | **Good pattern** — small curated index |
| `docs/benchmarks/` | Calibration JSON/MD artifacts | `benchmarks/README.md` lists gate sources |
| `docs/competition/` | Kaggle rules + submission SSOT | Indexed in `docs/README.md` |
| `docs/tools/` | Maintainer HTML tools | Indexed in `docs/README.md` |
| `docs/session-handoff/` | Ephemeral operator session notes | **Unclassified** — referenced as plan origins, not in lifecycle table |
| `docs/audits/` | Agent-native architecture audits | Linked from `docs/README.md` |
| `docs/archive/omg/` | ~148 retired OMG/MCP mirror files | Quarantined; CURSOR.md points here; still in-repo search path |
| `docs/Issues.md` | 42KB issue dump | `.cursorignore`d; unclear vs GitHub / ROADMAP |

**ONBOARDING (resolved):** Stale links to `experiments.md`, `baseline_sweep.md`, `config_migration.md` were removed; hand-maintained blocks point at `docs/README.md`.

**ROADMAP:** Now/Next empty; Done points to recent plans — human index works but agents may treat empty Now as "no priorities."

## Proposed information architecture

### R1 — Add `docs/README.md` as canonical map

Single landing page with:

1. **Start here** links: ONBOARDING → AGENT_CAPABILITIES → CURSOR → ROADMAP
2. **Doc types** table explaining lifecycle:

| Type | Folder | When to write | Typical next step |
|------|--------|---------------|-------------------|
| Ideation | `ideation/` | Unscoped exploration | brainstorm or discard |
| Requirements | `brainstorms/` | Problem + acceptance before code | `/ce-plan` → `plans/` |
| Plan | `plans/` | How-to-build with units | implementation → `solutions/` on resolve |
| Solution | `solutions/<category>/` | Resolved bug/pattern with frontmatter | linked from AGENTS.md if durable |
| Architecture | `architecture/` | Stable subsystem design | update when code owner changes |
| Benchmarks | `benchmarks/` | Committed calibration JSON + runbook MD | never invent thresholds |
| Audits | `audits/` | Point-in-time reviews | link from README; don't duplicate in ONBOARDING |
| Session handoff | `session-handoff/` | Ephemeral operator notes during multi-session work | archive after plan merges or fold into `solutions/` |

3. **Folder indexes** — central `docs/README.md` is the primary index. Optional per-folder stub READMEs are **P2/deferred** (see A4b). **Benchmarks index: see R5** (architecture/ already has a curated README pattern).

### R2 — Trim ONBOARDING § Documentation to live links only

Replace stale experiment/sweep/migration table with pointers to `docs/README.md` + architecture index + operator-runbook. Keep code tour in ONBOARDING; move doc-type policy out. **Shipped.**

### R3 — Clarify agent navigation chain

Two explicit chains (do not merge):

| Audience | Chain |
|----------|-------|
| **Human Start here** (R1) | ONBOARDING → AGENT_CAPABILITIES → CURSOR → ROADMAP |
| **Agent policy** (README + AGENTS.md) | `AGENTS.md` → `docs/README.md` → `docs/AGENT_CAPABILITIES.md` → `docs/solutions/` → `docs/plans/` |

Add one line in `AGENTS.md` pointing to `docs/README.md` as the doc-type map. **Shipped.**

`make agent-context` emits `docs/README.md` via `docs.readme` key. **Shipped** (plan 007).

### R4 — Retire or relocate root clutter

- `brain_dump.md`: already retired — move to `docs/archive/brain_dump-stub.md` or delete stub after README links archive
- `agent-native-phase2-status.md` / `agent-native-phase3-status.md`: consolidate under `docs/audits/` or single `docs/agent-native-status.md` (update `AGENTS.md` links if relocated)
- `Issues.md`: archive snapshot (`archive/issues-snapshot-YYYY-MM.md`) or delete in favor of GitHub-only tracking

### R5 — Benchmarks discoverability

Add `docs/benchmarks/README.md` listing committed artifacts (`preflight-calibration.json`, `seed-scheduler-calibration.json`, `unified-tournament-calibration.json`, launch-hygiene baselines) with "do not invent thresholds" policy pointer to AGENTS.md. **Shipped.** **Acceptance: A4a.**

## Recommended rollout (incremental)

1. **P0 (done):** R1 + R3 + R5/A4a + R2/A3
2. **P1 (done):** agent-context `docs.readme` pointer (plan 007)
3. **P2 (open):** R4 relocations and archival policy (A8, A9)
4. **Deferred:** Per-folder stub READMEs in `plans/`, `brainstorms/`, `solutions/` (A4b) — rejected as duplicate-index maintenance (plan 007 KTD1)

## Acceptance criteria

**Priority tiers:** P0 = ship first; P1 = follow-on; P2 = stretch.

| ID | Tier | Criterion | Status |
|----|------|-----------|--------|
| A1 | P0 | `docs/README.md` exists and links all committed top-level folders with one-sentence purpose | Done (11 folders incl. `session-handoff/`) |
| A2 | P1 | New agent can reach AGENT_CAPABILITIES, solutions index, and ROADMAP-linked active plans without grep | Done — `solutions/README.md` category index + agent policy chain |
| A3 | P0 | ONBOARDING § Documentation has zero broken relative links | Done |
| A4a | P0 | `benchmarks/README.md` lists committed calibration artifacts per R5 | Done |
| A4b | P2 | Optional stub READMEs in `plans/`, `brainstorms/`, `solutions/` | Deferred |
| A5 | P0 | Doc-type lifecycle stated in `docs/README.md` | Done |
| A6 | P0 | Retired stubs (`brain_dump.md`) no longer appear in "start here" paths | Done |
| A7 | P0 | `AGENTS.md` includes one line pointing to agent policy chain via `docs/README.md` | Done |
| A8 | P2 | R4 disposition for phase-status docs and `Issues.md` (relocation or archive) | Done |
| A9 | P2 | Archival policy: when to move root docs to `archive/`, when to delete stubs; `.cursorignore` dumps out-of-band for agents | Done |

## Key flows

**F1 — Agent cold start:** session hook → `make agent-context` → AGENTS.md → `docs/README.md` → AGENT_CAPABILITIES → `docs/solutions/` → `docs/plans/` as needed.

**F2 — Human plans feature:** ROADMAP Later → issue → brainstorm → plan → implementation → solution doc on non-obvious fix.

## Outstanding questions

- Should `docs/Issues.md` remain a manual dump or be deleted in favor of GitHub-only?
- Auto-generate folder READMEs from frontmatter vs hand-maintained tables? **Default: hand-maintained central index only** (reject auto-gen unless churn exceeds manual upkeep).
- Move phase trackers under `docs/audits/` vs consolidate to single `docs/agent-native-status.md`? (R4 lists both options)
- Should `archive/omg/` leave the main repo or stay quarantined in-tree with expanded `.cursorignore`?

### From 2026-06-06 doc review (deferred)

- **R4 priority:** Promote root clutter cleanup (`Issues.md`, phase-status consolidation) to P0/P1, or keep P2 and accept search-noise tradeoff?
- **Subtraction budget:** Should cleanup success be measured by file count reduction, agent token exposure, or navigation test pass rate?
- **archive/omg shrink:** Move ~148 retired OMG files out of default search path (submodule, separate repo, or expanded `.cursorignore`) vs keep in-tree quarantine?
- **session-handoff TTL:** Auto-archive handoff docs after originating plan merges?

## Resolve before planning

- ~~Whether ONBOARDING stays Understand-generated or becomes hand-curated nav + code map split~~ — **resolved:** hand-maintained documentation blocks (A3 green)
- Maintainer preference on Issues.md fate (blocks R4)
