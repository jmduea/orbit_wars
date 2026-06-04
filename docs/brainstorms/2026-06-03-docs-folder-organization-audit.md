---
status: draft-audit
date: 2026-06-03
scope: docs/ information architecture (not a single feature plan)
actors:
  - human-maintainer
  - coding-agent (AGENTS.md + CURSOR session-start)
  - new-contributor (ONBOARDING path)
---

# Docs folder organization audit

## Problem frame

The `docs/` tree has grown to **267+ files** across **eight** top-level folders (`architecture/`, `archive/`, `audits/`, `benchmarks/`, `brainstorms/`, `ideation/`, `plans/`, `solutions/`) plus **15 loose markdown files at `docs/` root**, but there is **no `docs/README.md` index**. Entry points (`ONBOARDING.md`, `AGENT_CAPABILITIES.md`, `ROADMAP.md`, `CURSOR.md`) partially overlap and some cross-links are stale. Agents cold-start via `make agent-context` and `AGENTS.md`, yet discoverability of **plans vs brainstorms vs solutions vs ideation vs audits** is implicit. This audit proposes a clearer information architecture without deleting pipeline artifacts.

## Current layout (inventory)

| Location | Count / role | Discoverability |
|----------|--------------|-----------------|
| `docs/` root (loose `.md`) | 15 files incl. ONBOARDING, ROADMAP, AGENT_CAPABILITIES, feature-encoding-v2, operator-runbook, retired brain_dump | No index; mixed evergreen vs status vs retired |
| `docs/plans/` | 35 dated feat/fix plans | Filename convention only; no README |
| `docs/brainstorms/` | 8 requirements-style docs | Same; boundary with plans unclear to newcomers |
| `docs/solutions/` | 16 learnings in 6 category subdirs | Referenced from AGENTS.md; no top-level solutions README |
| `docs/ideation/` | 2 exploratory docs | No definition vs brainstorm |
| `docs/architecture/` | 3 stage docs + README index | **Good pattern** — small curated index |
| `docs/benchmarks/` | 38 JSON/MD calibration artifacts | Partial runbooks; no single artifact index |
| `docs/audits/` | Agent-native architecture audits | Orphan from main navigation |
| `docs/archive/omg/` | Large retired OMG/MCP mirror | Correctly quarantined; CURSOR.md points here |
| `docs/Issues.md` | 42KB issue dump | Unclear relationship to GitHub issues / ROADMAP |

**ONBOARDING staleness:** § Documentation table lists `docs/experiments.md`, `docs/baseline_sweep.md`, `docs/config_migration.md` — **none exist** on disk (only `hydra_migration.md` exists among migration docs).

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

3. **Folder indexes** — add minimal README stubs in `plans/`, `brainstorms/`, `solutions/` (one-sentence purpose + link back to `docs/README.md`). **Benchmarks index content: see R5** (architecture/ already has a curated README pattern).

### R2 — Trim ONBOARDING § Documentation to live links only

Replace stale experiment/sweep/migration table with pointers to `docs/README.md` + architecture index + operator-runbook. Keep code tour in ONBOARDING; move doc-type policy out.

### R3 — Clarify agent navigation chain

Two explicit chains (do not merge):

| Audience | Chain |
|----------|-------|
| **Human Start here** (R1) | ONBOARDING → AGENT_CAPABILITIES → CURSOR → ROADMAP |
| **Agent policy** (README + AGENTS.md) | `AGENTS.md` → `docs/README.md` → `docs/AGENT_CAPABILITIES.md` → `docs/solutions/` → `docs/plans/` |

Add one line in `AGENTS.md` pointing to `docs/README.md` as the doc-type map.

`make agent-context` emitting `docs/README.md` is a **follow-up code change** (not this doc-only pass); F1 is partially satisfied until `scripts/agent_context.py` adds the pointer.

### R4 — Retire or relocate root clutter

- `brain_dump.md`: already retired — move to `docs/archive/brain_dump-stub.md` or delete stub after README links archive
- `agent-native-phase2-status.md` / `phase3-status.md`: consolidate under `docs/audits/` or single `docs/agent-native-status.md`
- `Issues.md`: either generate from GitHub or rename to `archive/issues-snapshot-YYYY-MM.md`

### R5 — Benchmarks discoverability

Add `docs/benchmarks/README.md` listing committed artifacts (`preflight-calibration.json`, `seed-scheduler-calibration.json`, `unified-tournament-calibration.json`, launch-hygiene baselines) with "do not invent thresholds" policy pointer to AGENTS.md. **Acceptance: A4 benchmarks row.**

## Recommended rollout (incremental)

1. **P0:** R1 (`docs/README.md`) + R3 AGENTS.md line (A1, A5, A7) in parallel with R5/A4 benchmarks README
2. **P0:** Resolve ONBOARDING regen policy, then R2 (A3)
3. **P1:** Stub folder READMEs (A4); agent-context `docs/README.md` pointer (code follow-up)
4. **P2:** R4 relocations (A8) after Issues.md / status-file decisions

## Acceptance criteria

**Priority tiers:** P0 = ship first; P1 = follow-on; P2 = stretch.

| ID | Tier | Criterion |
|----|------|-----------|
| A1 | P0 | `docs/README.md` exists and links all eight top-level folders with one-sentence purpose |
| A2 | P1 | New agent can reach AGENT_CAPABILITIES, solutions index, and ROADMAP-linked active plans without grep |
| A3 | P0 | ONBOARDING § Documentation has zero broken relative links (resolve Understand regen policy before R2 — see Resolve before planning) |
| A4 | P1 | Each of `plans/`, `brainstorms/`, `solutions/` has a stub README; `benchmarks/README.md` lists committed calibration artifacts per R5 |
| A5 | P0 | Doc-type lifecycle (ideation → brainstorm → plan → solution) is stated in one place (`docs/README.md`) |
| A6 | P0 | Retired stubs (`brain_dump.md`) no longer appear in "start here" paths |
| A7 | P0 | `AGENTS.md` includes one line pointing to the agent policy chain via `docs/README.md` |
| A8 | P2 | R4 disposition for phase-status docs and `Issues.md` decided (link from README without relocation acceptable for MVP) |

## Key flows

**F1 — Agent cold start:** session hook → `make agent-context` → AGENTS.md → (`docs/README.md` when added to agent-context) → AGENT_CAPABILITIES → solutions lookup as needed.

**F2 — Human plans feature:** ROADMAP Later → issue → brainstorm → plan → implementation → solution doc on non-obvious fix.

## Outstanding questions

- Should `docs/Issues.md` remain a manual dump or be deleted in favor of GitHub-only?
- Auto-generate folder READMEs from frontmatter vs hand-maintained tables?
- Move phase trackers under `docs/audits/` vs consolidate to single `docs/agent-native-status.md`? (R4 lists both options)

## Resolve before planning

- Maintainer preference on Issues.md fate
- Whether ONBOARDING stays Understand-generated or becomes hand-curated nav + code map split
