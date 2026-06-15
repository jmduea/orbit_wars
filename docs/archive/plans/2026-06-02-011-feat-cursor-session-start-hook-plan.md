---
title: "feat: Cursor session-start hook example"
date: 2026-06-02
status: completed
type: feat
origin: "docs/ROADMAP.md Next — Phase 3 plan §4"
depends_on: "docs/plans/2026-06-02-agent-native-phase3-refactors.md"
---

# feat: Cursor Session-Start Hook Example

## Summary

Close ROADMAP **Next** item: ship an optional project-level `.cursor/hooks.json` + hook script that runs `make agent-context` on `sessionStart`, complementing the existing copy-paste docs in `docs/CURSOR.md` (added in PR #175).

---

## Problem Frame

Agents cold-start without preflight thresholds, roadmap excerpts, or recent run pointers unless operators manually run `make agent-context`. PR #175 documented the hook in `docs/CURSOR.md` but did not commit a repo-relative example; ROADMAP still lists the item in **Next**.

---

## Requirements

- **R1:** Add `.cursor/hooks.json` (version 1) with `sessionStart` → repo hook script.
- **R2:** Add `.cursor/hooks/session-start-agent-context.sh` — runs `make agent-context`, returns JSON `{ "additional_context": "..." }` per Cursor hooks schema; fail-open on errors.
- **R3:** Update `docs/CURSOR.md` to reference committed files (not absolute paths only).
- **R4:** Triage `docs/ROADMAP.md`: move item from **Next** → **Done** (last 5 cap).
- **R5:** Lightweight test: hook script exits 0 and stdout is valid JSON with `additional_context` key.

---

## Scope Boundaries

- No user secrets or machine-specific paths in committed files.
- No blocking enforcement hooks on `src/` edits (governance unchanged).
- Launch hygiene tier-2 e2e throughput — out of scope (not in ROADMAP Now/Next).

---

## Implementation Units

### U1. Project hook files

**Files:** `.cursor/hooks.json`, `.cursor/hooks/session-start-agent-context.sh`

**Decisions:** Script resolves repo root from `.cursor/hooks/`; uses `make agent-context` (uv-backed); formats payload as markdown JSON block in `additional_context`.

**Test scenarios:** Script produces parseable JSON; `additional_context` non-empty when repo has calibration + roadmap.

### U2. Docs + ROADMAP

**Files:** `docs/CURSOR.md`, `docs/ROADMAP.md`

**Test scenarios:** ROADMAP **Next** empty or replaced; **Done** includes session-start hook with plan link.

### U3. Hook smoke test

**Files:** `tests/test_cursor_hooks.py`

**Test scenarios:** Subprocess hook script from repo root; assert exit 0 and JSON schema.

---

## Verification

```bash
make test-fast
bash .cursor/hooks/session-start-agent-context.sh </dev/null | python -m json.tool
```

---

## References

- Phase 3 §4: `docs/plans/2026-06-02-agent-native-phase3-refactors.md`
- Cursor hooks docs: https://cursor.com/docs/hooks
- Existing: `docs/CURSOR.md`, `scripts/agent_context.py`, `tests/test_agent_context.py`
