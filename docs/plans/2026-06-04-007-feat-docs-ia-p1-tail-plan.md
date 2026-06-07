---
title: "feat: Docs IA P1 tail (agent-context, benchmarks index, tools/)"
type: feat
status: active
date: 2026-06-04
origin: docs/brainstorms/2026-06-03-docs-folder-organization-audit.md
---

# Plan: Docs IA P1 tail

## Summary

Close the remaining P1 items from the docs folder organization audit after P0 shipped in plan `2026-06-03-010`: wire `docs/README.md` into `make agent-context`, add a committed-calibration index under `docs/benchmarks/`, and list `docs/tools/` in the canonical folder map. Defer optional stub READMEs in `plans/`, `brainstorms/`, and `solutions/`.

## Problem Frame

P0 navigation IA is live (`docs/README.md`, `AGENTS.md` pointer, ONBOARDING hand-maintained blocks, `tests/test_docs_navigation.py`). Agents running `make agent-context` still omit the doc-type map from session JSON. `docs/benchmarks/` holds 39+ calibration artifacts with no folder README. `docs/tools/` exists with its own README but is missing from `docs/README.md` Top-level folders.

## Requirements

- R1. `scripts/agent_context.py` `build_context()["docs"]` includes `docs/README.md` (origin F1, audit R3 follow-up).
- R2. `docs/benchmarks/README.md` lists committed calibration JSON/runbooks and points to `AGENTS.md` for threshold policy (origin R5, A4 benchmarks portion).
- R3. `docs/README.md` Top-level folders table includes `tools/` with one-sentence purpose (origin inventory gap).
- R4. Regression tests cover agent-context docs map and new README links (origin A2 partial).

### Out of scope (P1 deferred)

- Stub READMEs in `docs/plans/`, `docs/brainstorms/`, `docs/solutions/` (audit A4 folder portion).
- R4 root relocations (phase-status consolidation, `Issues.md` archive).

## Key Technical Decisions

**KTD1 — Minimal P1 slice.** Ship the three high-ROI items above; skip hand-maintained per-folder stubs to avoid duplicate index maintenance (doc review consensus).

**KTD2 — Benchmarks README is a manifest, not a runbook duplicate.** Table committed artifacts agents gate on; link to existing `.md` runbooks (`preflight-calibration.md`, `seed-scheduler-calibration.md`) rather than inlining thresholds.

**KTD3 — agent_context key name `readme`.** Match sibling keys (`onboarding`, `agent_capabilities`) with value `docs/README.md`.

## Implementation Units

### U1. Add `docs/README.md` to agent context

**Goal:** Complete F1 agent cold-start chain in session JSON.

**Requirements:** R1

**Files:** `scripts/agent_context.py`, `tests/test_agent_context.py`

**Approach:** Add `"readme": "docs/README.md"` under `docs` in `build_context()`. Extend `test_build_context_includes_preflight_and_roadmap` (or add focused test) to assert `payload["docs"]["readme"] == "docs/README.md"`.

**Patterns to follow:** Existing `docs` dict in `build_context()`.

**Test scenarios:**
- Happy path: `build_context(limit_runs=0)["docs"]["readme"]` equals `docs/README.md`.

**Verification:** `uv run pytest tests/test_agent_context.py -q`

---

### U2. Create `docs/benchmarks/README.md`

**Goal:** Discoverability for committed calibration artifacts and policy pointer.

**Requirements:** R2

**Files:** `docs/benchmarks/README.md` (create)

**Approach:** Short index patterned after `docs/architecture/README.md`: purpose line, policy pointer to `AGENTS.md` preflight section, table of primary committed JSON (`preflight-calibration.json`, `seed-scheduler-calibration.json`, `unified-tournament-calibration.json`, `preflight-profiles.json`, launch-hygiene baselines, `qualifier-seed-calibration.json`), links to companion `.md` runbooks. Note that historical/issue-specific JSON files remain in-folder but are not gate sources.

**Patterns to follow:** `docs/architecture/README.md`, audit R5 artifact list.

**Test scenarios:**
- Happy path: every markdown link in `docs/benchmarks/README.md` resolves.

**Verification:** Included in U4 navigation test parametrization or dedicated link test.

---

### U3. Index `docs/tools/` in `docs/README.md`

**Goal:** Close top-level folder gap for SSOT flowchart and config picker.

**Requirements:** R3

**Files:** `docs/README.md`

**Approach:** Add row to Top-level folders table linking `tools/` with purpose (local maintainer HTML tools, SSOT flowchart). Optionally cross-link from Config → Kaggle SSOT section to `tools/ssot-training-pipeline-flowchart.html`.

**Test scenarios:**
- Happy path: `tools/` link in README resolves to directory.

**Verification:** U4 passes.

---

### U4. Extend docs navigation tests

**Goal:** Guard new links and agent-context wiring.

**Requirements:** R4

**Dependencies:** U1, U2, U3

**Files:** `tests/test_docs_navigation.py`, `tests/test_agent_context.py`

**Approach:** Parametrize benchmarks README links if created; assert `tools/` in README folder section resolves; agent-context test from U1.

**Test scenarios:**
- Happy path: `test_docs_readme_links_resolve` covers new README hrefs after benchmarks README exists.

**Verification:** `make test-fast` filtered to `test_docs_navigation.py` and `test_agent_context.py`.

## Scope Boundaries

### Deferred to Follow-Up Work

- Per-folder stub READMEs (`plans/`, `brainstorms/`, `solutions/`).
- Phase-status file consolidation under `docs/audits/`.
- Auto-generated folder indexes from frontmatter.

## Risks & Dependencies

- Low risk doc-only change; no JAX/training behavior.
- Coordinate with SSOT doc spine work (#205) — do not rewrite `AGENT_CAPABILITIES.md` operator paths in this slice.
