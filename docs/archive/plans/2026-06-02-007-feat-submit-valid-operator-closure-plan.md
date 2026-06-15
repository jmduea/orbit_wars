---
date: 2026-06-02
topic: submit-valid-operator-closure
status: completed
origin: docs/ROADMAP.md (Now: submit-valid path, hybrid promotion)
---

# feat: Submit-valid operator closure

## Summary

Close the remaining **Now** roadmap gaps for submit-valid and hybrid promotion after CLI hardening (#160/#161) and preflight primitives (PR #175). Agents currently need a two-step poll (`ow eval status` → `ow eval results show`) because status JSON omits `validation_ok`. This plan inlines checkpoint-eval proof fields in status/results list, adds a Hydra contract test for `artifacts=hybrid_promotion`, and triages ROADMAP Now/Done.

## Problem frame

Submit-valid proof requires manifest-backed `validation_ok` from hybrid `checkpoint_eval` jobs or `ow eval package --validate-docker`. Docs and CLI help are in place; the operator gap is status introspection and profile assurance.

## Requirements

| ID | Requirement |
|----|-------------|
| R1 | `ow eval status` JSON includes per-`checkpoint_eval` summary with `validation_ok`, `update`, `status`, and `result_dir` when evaluation manifests exist. |
| R2 | `ow eval results list` rows include `validation_ok` for checkpoint_eval manifests (alongside existing kind/update/status fields). |
| R3 | Hydra resolves `artifacts=hybrid_promotion` to hybrid strategy, tournament enabled, checkpoint_eval_async on, replay off. |
| R4 | ROADMAP reflects completed Now items (preflight primitives, submit-valid docs/CLI); hybrid promotion marked done when R1–R3 land. |

## Non-goals

- Changing hybrid worker behavior or promotion thresholds
- Gate 5 tournament-proof semantics or calibration JSON
- Inlining full tournament manifests in status (summary fields only)
- Cursor session-start hook (Later in ROADMAP)

## Key technical decisions

**KTD1 — Status carries a `checkpoint_evals` array, not a single field.** Runs may queue multiple checkpoint evals; expose all completed/failed manifests sorted by update so agents can pick the latest or scan history without a second CLI call.

**KTD2 — Read evaluation manifests, not queue job files, for proof fields.** Queue jobs may be stale; `evaluations/checkpoint_eval_u*/manifest.json` is the durable submit-valid record (same source as `results show`).

**KTD3 — Config test uses existing `test_config_consolidation.py` pattern.** Membership assertions on promotion strategy and artifact pipeline flags; no full resolved-config equality.

## Implementation units

### U1. Inline checkpoint_eval summaries in run status

**Goal:** Agents read `validation_ok` from `ow eval status` without `results show`.

**Files:** `src/cli/run_status.py`, `tests/test_cli_eval_status.py`

**Approach:** Add `_load_checkpoint_eval_summaries(evaluations_dir)` that globs `checkpoint_eval_*/manifest.json`, parses `validation_ok`, `update`, `status`, `promoted`, `tournament_id`, and attaches as `checkpoint_evals` on `summarize_run_status` output. Preserve existing job rows unchanged.

**Test scenarios:**
- Happy path: run dir with completed checkpoint_eval manifest → summary includes `validation_ok: true` and matching update.
- Edge: no evaluations dir → `checkpoint_evals` is empty list.
- Edge: docker failure manifest with `validation_ok: false` → surfaced in summary.

**Verification:** `make test-domain-artifacts` subset passes; status CLI JSON contains `checkpoint_evals`.

### U2. Expose validation_ok in results list

**Goal:** `ow eval results list` rows include proof field for checkpoint_eval entries.

**Files:** `src/cli/run_status.py`, `tests/test_cli_eval_results.py`

**Approach:** Extend `list_evaluation_results` key extraction to include `validation_ok` when present in manifest.

**Test scenarios:**
- checkpoint_eval row includes `validation_ok`.
- tournament-only row omits `validation_ok` when absent.

**Verification:** results list CLI test asserts `validation_ok` in JSON output.

### U3. Hybrid promotion Hydra contract test

**Goal:** CI guards `artifacts=hybrid_promotion` profile invariants.

**Files:** `tests/test_config_consolidation.py`

**Approach:** Compose `artifacts=hybrid_promotion` via Hydra; assert promotion.strategy hybrid, tournament.enabled true, checkpoint_eval_async true, replay.enabled false, docker_validation_async false, replay_async false.

**Verification:** `make test-domain-config` passes.

### U4. Docs and ROADMAP triage

**Goal:** Docs match behavior; ROADMAP Now reflects reality.

**Files:** `AGENTS.md`, `docs/AGENT_CAPABILITIES.md`, `docs/ROADMAP.md`

**Approach:** Update hybrid poll contract to note status inlines `checkpoint_evals[].validation_ok`; move preflight primitives and submit-valid path to Done; remove or narrow remaining Now entries.

**Verification:** Doc grep for "does not yet inline" removed or updated.

## Acceptance criteria

- `make test-fast` green after U1–U3
- Status JSON documents submit-valid proof without mandatory `results show` for idle poll
- ROADMAP Now ≤3 with accurate Done entries
