---
title: "feat: Plan hygiene and close Planet Flow residual issues"
type: feat
status: completed
date: 2026-06-02
origin: LFG next steps — ROADMAP Now/Next empty; GitHub #168–#170 open after #177
---

# Plan: Plan hygiene and close Planet Flow residual issues

## Summary

Mark merged implementation plans as `completed`, document Agent-native Phase 3 items 1–4 as shipped on `main`, and close GitHub issues #168–#170 with verification references (code already landed in PR #177 / follow-ups).

## Problem Frame

ROADMAP **Now/Next** are empty; multiple `docs/plans/*.md` still show `status: active` after merge. Issues #168–#170 remain open though shared PPO finalization, Planet Flow metric descriptors, and compiler-control tests exist on `main`.

## Requirements

| ID | Requirement |
|----|-------------|
| R1 | Flip `status: active` → `completed` on plans whose work is merged (operator, observability, planet-flow sweep/reachability/angle, preflight profiles, launch hygiene e2e). |
| R2 | Update `docs/plans/2026-06-02-agent-native-phase3-refactors.md` with shipped status for items 1–4. |
| R3 | Add `docs/agent-native-phase3-status.md` pointing agents to primitives (`ow benchmark gate run`, `ow sweep`, gate YAML). |
| R4 | PR closes #168, #169, #170 with evidence paths. |
| R5 | `make test-fast` green. |

## Out of scope

- Seed scheduler GPU calibration (`2026-06-01-003` stays active).
- Launch hygiene Phase B hot-path recovery (ROADMAP Later).
- New Planet Flow features.

## Implementation Units

### U1. Complete stale plan frontmatter

**Files:** Nine plan files under `docs/plans/` currently `status: active` but merged.

**Verification:** Grep shows no erroneous `active` on completed tracks except seed-scheduler plan.

### U2. Phase 3 status doc + backlog update

**Files:** `docs/agent-native-phase3-status.md` (new), `docs/plans/2026-06-02-agent-native-phase3-refactors.md`

### U3. Cross-link AGENTS.md

**Files:** `AGENTS.md` — one line under agent workflow pointing to phase 3 status.

### U4. Issue closure in PR

**Verification:** PR body lists Closes #168 #169 #170 with file references.

## Test scenarios

- `make test-fast` — no behavioral change expected.

## Verification

```bash
make test-fast
rg 'status: active' docs/plans/
```
