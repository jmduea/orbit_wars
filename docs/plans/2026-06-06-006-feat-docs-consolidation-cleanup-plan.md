---
title: "feat: Docs consolidation and stale-doc cleanup"
type: feat
status: completed
date: 2026-06-06
origin: docs/brainstorms/2026-06-03-docs-folder-organization-audit.md
---

# Plan: Docs consolidation and stale-doc cleanup

## Summary

Close the remaining **subtraction** work from the docs folder organization audit after P0/P1 navigation IA shipped (plans `010`, `007`, and 2026-06-06 doc review applies). Relocate or archive root clutter, document an explicit retention policy, consolidate ephemeral status docs, and reduce agent search noise from retired material — without touching committed benchmarks JSON or pipeline artifacts in `brainstorms/`, `plans/`, or `solutions/`.

## Problem Frame

Navigation indexes are live (`docs/README.md`, `solutions/README.md`, `benchmarks/README.md`, `session-handoff/` classified). The tree still has **331 files** with **~45%** under `docs/archive/omg/`, plus root-level noise (`Issues.md`, phase-status trackers) that agents can still grep even when `.cursorignore` excludes some paths from chat context.

The audit's original scope stopped at "better maps." The user's goal is **consolidation/cleanup of stale, deprecated, and never-used docs** — measurable reduction in confusing surfaces, not more README stubs.

## Requirements

| ID | Requirement |
|----|-------------|
| R1 | **Archival policy** documented in `docs/README.md` Maintenance section: immutable (benchmarks JSON, gate sources) vs eligible-for-archive (manual snapshots, superseded status docs, session handoffs after merge). |
| R2 | **`docs/Issues.md` disposition:** archive to `docs/archive/issues-snapshot-2026-06.md` and remove from root; update `docs/README.md`, `.cursorignore`, and `docs/CURSOR.md` references. GitHub + ROADMAP remain live trackers. |
| R3 | **Phase-status consolidation:** merge durable content from `docs/agent-native-phase2-status.md` and `docs/agent-native-phase3-status.md` into a single `docs/audits/agent-native-status.md`; update all inbound links (`AGENTS.md`, solutions, plans); delete or archive superseded root copies. |
| R4 | **Session-handoff TTL:** move handoffs whose originating plans are `status: completed` to `docs/archive/session-handoff/`; add one-line policy to `docs/README.md`. |
| R5 | **Plan frontmatter hygiene:** flip `status: active` → `completed` on merged plans still marked active (grep-driven pass; keep legitimately open plans active). |
| R6 | **`.cursorignore` accuracy:** fix stale `docs/brain_dump.md` path (file already at `docs/archive/brain_dump.md`); optionally add `docs/archive/omg/` to reduce agent context noise. |
| R7 | **Regression tests:** extend `tests/test_docs_navigation.py` for new archive paths and consolidated status doc; assert no Start-here links to archived root clutter. |

### Out of scope

- Deleting files under `docs/brainstorms/`, `docs/plans/`, or `docs/solutions/` (pipeline artifacts — link/update only).
- Auto-generated per-folder README stubs (deferred per plan 007 KTD1).
- Moving `archive/omg/` out of the repo (submodule/separate repo) — optional follow-up if `.cursorignore` insufficient.
- Rewriting `AGENT_CAPABILITIES.md` operator paths or SSOT #205 spine content.

## Key Technical Decisions

**KTD1 — Archive over delete for historical snapshots.** `Issues.md` and session handoffs move under `docs/archive/` with dated names so git history preserves context; delete only empty stubs (e.g. retired `brain_dump` at wrong path).

**KTD2 — Single agent-native status under audits/.** Phase 2/3 status docs are point-in-time ship trackers, not evergreen root docs. Consolidation matches audit R4 and keeps `docs/` root for long-lived references only.

**KTD3 — `.cursorignore` as fast noise reduction.** Expanding ignore for `docs/archive/omg/` is lower risk than mass deletion; document in `docs/CURSOR.md` that archive is off-model by default.

**KTD4 — Plan status pass is mechanical.** Use `rg 'status: active' docs/plans/` and cross-check merged PR evidence / plan Verification sections; do not flip plans for in-flight work (pick4, opponent rollout, SSOT slices still active).

## Implementation Units

### U1. Document archival policy (R1)

**Goal:** One place states what must never be deleted vs what can be archived.

**Files:** `docs/README.md`, `docs/brainstorms/2026-06-03-docs-folder-organization-audit.md` (mark A9 done)

**Approach:** Add **Retention policy** subsection under Maintenance with two tiers and examples. Close audit A9.

**Test scenarios:**
- Policy section names benchmarks JSON as immutable and Issues/session-handoff as archivable.

**Verification:** Manual read; no code change.

---

### U2. Archive Issues.md (R2)

**Goal:** Remove 42KB manual dump from docs root.

**Files:** `docs/Issues.md` → `docs/archive/issues-snapshot-2026-06.md`, `docs/README.md`, `.cursorignore`, `docs/CURSOR.md`

**Approach:** `git mv` to archive with date stamp. Remove root link from README Other root docs (or point to archive). Update `.cursorignore` path. CURSOR.md table row → archived path.

**Patterns:** `docs/archive/brain_dump.md` precedent.

**Test scenarios:**
- `test_docs_readme_links_resolve` — no broken href after README edit.
- No test asserts Issues.md at root.

**Verification:** `uv run pytest tests/test_docs_navigation.py -q`

---

### U3. Consolidate agent-native status docs (R3)

**Goal:** One status doc under audits; fix inbound links.

**Files:** Create `docs/audits/agent-native-status.md`; archive `docs/agent-native-phase2-status.md`, `docs/agent-native-phase3-status.md`; update `AGENTS.md`, `docs/solutions/developer-experience/agent-native-operator-cli-phase1.md`, `docs/solutions/architecture-patterns/benchmark-cli-package-split-agent-native-parity.md`, `docs/solutions/developer-experience/seed-scheduler-calibration-agent-native-operator-phase2.md`, any plan cross-links found via `rg agent-native-phase`.

**Approach:** Merge shipped tables and operator primitive blocks from phase2+phase3 into single doc; add "supersedes phase2/phase3 root files" note. Redirect old paths with one-line stubs at old locations only if external links exist — prefer pure `git mv` to archive + link updates only.

**Test scenarios:**
- `rg 'agent-native-phase[23]-status' docs/ AGENTS.md` returns zero after migration (or only archive paths).

**Verification:** `uv run pytest tests/test_docs_navigation.py -q`; `make test-fast`

---

### U4. Archive stale session handoffs (R4)

**Goal:** Ephemeral notes don't accumulate at top level.

**Files:** `docs/session-handoff/*.md` → `docs/archive/session-handoff/` for completed-origin handoffs; `docs/README.md` lifecycle row TTL note

**Approach:** For each handoff, check if referenced plan is `completed` or work merged; move completed ones. Keep active handoffs (e.g. ongoing env-parity) in place.

**Test scenarios:**
- README session-handoff row mentions archive-after-merge policy.

**Verification:** `tests/test_docs_navigation.py` if archive README links added.

---

### U5. Plan frontmatter hygiene pass (R5)

**Goal:** `status: active` reflects reality.

**Files:** Subset of `docs/plans/*.md` still `active` but merged (e.g. `2026-06-04-007`, `2026-06-03-010` after this cleanup slice ships)

**Approach:** `rg 'status: active' docs/plans/`; for each, confirm Verification or main merge; flip to `completed` with date note in plan if needed.

**Test scenarios:**
- After pass, active count matches genuinely open tracks (pick4, opponent rollout, SSOT, env-parity AB, etc.).

**Verification:** `rg 'status: active' docs/plans/` manual review

---

### U6. Fix .cursorignore and optional archive/omg ignore (R6)

**Goal:** Ignore rules match on-disk paths; reduce OMG mirror noise.

**Files:** `.cursorignore`, `docs/CURSOR.md`

**Approach:** Replace `docs/brain_dump.md` with `docs/archive/brain_dump.md` or remove if redundant. Add `docs/archive/omg/` block with comment. Document in CURSOR.md.

**Test scenarios:**
- CURSOR.md table matches `.cursorignore` entries.

**Verification:** Manual; hooks unchanged.

---

### U7. Navigation regression tests (R7)

**Goal:** Guard consolidated paths.

**Files:** `tests/test_docs_navigation.py`

**Approach:** Parametrize `docs/audits/agent-native-status.md` links if added to README; assert Start here excludes `Issues.md` at root; optional test that `docs/archive/issues-snapshot-2026-06.md` exists after U2.

**Dependencies:** U2, U3

**Verification:** `make test-fast` filtered to `test_docs_navigation.py`

## Scope Boundaries

### Deferred

- `archive/omg/` removal from repo (submodule / separate repo).
- Auto-generated plan/brainstorm indexes.
- Full ROADMAP Now/Next population (human-only per `docs/ROADMAP.md` policy).

## Risks & Dependencies

- **Link churn:** U3 touches many cross-references — use `rg` sweep before merge.
- **AGENTS.md is high-traffic:** single-line update must point to new audits path.
- **Low risk:** no JAX/training behavior; doc-only slice suitable for `make test-fast`.

## Verification

```bash
make test-fast
rg 'status: active' docs/plans/
rg 'docs/Issues.md' docs/ AGENTS.md .cursorignore
uv run pytest tests/test_docs_navigation.py -q
```
