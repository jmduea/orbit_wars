---
title: "Agent-native audit gap closure (CRUD, context, primitives)"
status: completed
date: 2026-06-02
origin: docs/audits/agent-native-architecture-2026-06-02.md
---

# Agent-native audit gap closure

## Summary

Ship one PR-sized slice addressing the audit's top three operator gaps: extend `make agent-context` JSON (gate ids, resolved-config hash, GPU contention hint), add `ow sweep cancel` for W&B sweep teardown, and decompose `learn-proof` with `--steps` / `--print-primitives` plus documented primitive-only paths in `docs/AGENT_CAPABILITIES.md`.

## Problem frame

The 2026-06-02 agent-native audit scored Context Injection (67%), CRUD (83% operator-adequate), and Tools as Primitives (80%). Agents still re-discover gate ids, run monolithic learn-proof, and lack sweep cancel. This plan closes the highest-ROI slices without touching seed-scheduler GPU calibration or hybrid promotion.

## Requirements

| ID | Requirement |
|----|-------------|
| R1 | `make agent-context` includes preflight gate ids, resolved-config SHA prefix, GPU contention hint |
| R2 | Optional `make agent-context RESOLVED=smoke` embeds truncated resolved-config snapshot |
| R3 | `ow sweep cancel --backend wandb --sweep-id <id>` cancels active sweep runs (dry-run supported) |
| R4 | `ow benchmark learn-proof --print-primitives` emits primitive command chain JSON |
| R5 | `ow benchmark learn-proof --steps beat_noop,beat_random` runs subset in ladder order |
| R6 | `docs/AGENT_CAPABILITIES.md` documents new primitives and CRUD boundaries |

## Key technical decisions

**KTD1 — No JAX in agent_context.** Gate ids read from `conf/benchmark/gates/*.yaml` stems only; do not import `benchmark_gates` (pulls JAX via loader).

**KTD2 — Resolved config via existing Hydra path.** Subprocess `uv run ow train print_resolved_config=true training=smoke ...`; store SHA256 prefix by default, full truncated snapshot only when `RESOLVED=smoke`.

**KTD3 — GPU contention via pgrep.** Lightweight process-pattern scan (`ow train`, `calibrate-seed-scheduler`, `pytest`); single-GPU note always present.

**KTD4 — Sweep cancel cancels running W&B runs.** Use `wandb.Api()` run.cancel() for active runs; kaggle backend documents as unsupported.

**KTD5 — learn-proof steps preserve GATE_ORDER.** `--steps` filters `GATE_ORDER`; does not reorder.

## Scope boundaries

### In scope

U1–U3 below.

### Deferred to follow-up work

- `ow runs archive` / checkpoint delete primitives
- `ow benchmark factorized-sampler` Makefile wrapper
- Deprecation stderr on demoted scripts
- Capability map parity regression test

---

## Implementation units

### U1. Extend agent-context JSON

**Goal:** Inject gate ids, resolved-config pointer/hash, GPU contention into session JSON.

**Requirements:** R1, R2

**Files:** `scripts/agent_context.py`, `Makefile`, `tests/test_agent_context.py`

**Approach:** Add helpers for gate stems, resolved-config subprocess+hash, pgrep contention. Makefile passes `--resolved smoke` when `RESOLVED=smoke`.

**Test scenarios:**

- Happy: `build_context()` includes `preflight_gates.gate_ids`, `gpu_contention`, `resolved_config.sha256_prefix`
- Edge: missing gates dir returns empty list
- Edge: resolved subprocess failure sets `present: false`

**Verification:** `make test-fast` includes updated `test_agent_context.py`.

### U2. ow sweep cancel

**Goal:** Operator-safe W&B sweep teardown primitive.

**Requirements:** R3

**Files:** `src/cli/sweep.py`, `tests/test_cli_sweep.py`, `docs/AGENT_CAPABILITIES.md`

**Approach:** Add `cancel` subcommand; dry-run JSON; cancel running/pending runs via W&B API.

**Test scenarios:**

- Parser accepts cancel with required flags
- Dry-run returns JSON without API calls (mock wandb or dry-run path)

**Verification:** `tests/test_cli_sweep.py` passes.

### U3. learn-proof primitive decomposition

**Goal:** Document and expose primitive chain for learn-proof workflow.

**Requirements:** R4, R5, R6

**Files:** `src/cli/benchmark.py`, `tests/test_benchmark_cli.py` (or new test file), `docs/AGENT_CAPABILITIES.md`

**Approach:** Add `--print-primitives`, `--steps`; expand AGENT_CAPABILITIES with primitive ladder table and CRUD notes.

**Test scenarios:**

- `--print-primitives` exits 0 with primitives array
- `--steps beat_noop --dry-run` runs single gate only

**Verification:** targeted benchmark CLI tests + `make test-fast`.

---

## Open questions

- W&B sweep entity-level cancel API varies by version; run.cancel loop is sufficient for operator use.
