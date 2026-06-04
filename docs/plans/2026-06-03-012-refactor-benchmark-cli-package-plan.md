---
date: 2026-06-03
topic: benchmark-cli-package
status: active
type: refactor
origin: docs/plans/2026-06-03-011-refactor-src-simplification-followup-plan.md (U3); GitHub issue #194
---

# refactor: split ow benchmark CLI into src/cli/benchmark/ package

## Summary

Split the ~1510-line `src/cli/benchmark.py` monolith into `src/cli/benchmark/` per follow-up plan U3 and issue #194, preserving `ow benchmark` CLI surface, help text, and `LEARN_PROOF_PRIMITIVES` strings byte-for-byte.

## Problem Frame

`src/cli/benchmark.py` is unmaintainable as a single module. Gate logic already lives in `src/cli/benchmark_gates.py`. Factorized sampler benchmarking is inlined in `src/jax/factorized_sampler_benchmark.py` (no subprocess). This change is structural only — no behavior change.

## Requirements

| ID | Requirement |
|----|-------------|
| R1 | Package layout: `parser.py`, `training.py`, `sanity.py`, `calibrate.py`, `learn_proof.py`, `planet_flow.py`, `factorized.py`, `gate.py`, shared `common.py` |
| R2 | Update `src/cli/__init__.py` to import `main` from package; delete monolith `src/cli/benchmark.py` |
| R3 | `LEARN_PROOF_PRIMITIVES` and `print_benchmark_help()` text unchanged |
| R4 | `build_parser()` argparse tree unchanged (tests parse args) |
| R5 | Lazy JAX imports remain inside command handlers, not package `__init__` |
| R6 | Do not reintroduce subprocess for factorized-sampler |

## Key Technical Decisions

**KTD1 — Mirror `benchmark_gates.py`.** Thin `gate.py` delegates to `benchmark_gates.run_gate_cli`; parser registration colocated per command family.

**KTD2 — `src.cli.benchmark` is the package.** Tests using `from src.cli import benchmark` and `from src.cli.benchmark import build_parser` continue to work without a shim file at `benchmark.py`.

**KTD3 — Parser registration helpers.** Each command module exposes `register_parser(subparsers)`; `parser.build_parser()` orchestrates.

## Implementation Units

### U1. Create package skeleton and common helpers

**Goal:** `src/cli/benchmark/common.py` with `REPO_ROOT`, `LEARN_PROOF_PRIMITIVES`, `_git_head_sha`, `_init_benchmark_runtime`, `print_benchmark_help`.

**Files:** `src/cli/benchmark/__init__.py`, `src/cli/benchmark/common.py`

**Verification:** Import package without loading JAX.

### U2. Split parsers and command runners

**Goal:** Move `build_parser` and each `run_*_cli` into dedicated modules; wire `main()` dispatch in `__init__.py`.

**Files:** `parser.py`, `training.py`, `sanity.py`, `calibrate.py`, `learn_proof.py`, `planet_flow.py`, `factorized.py`, `gate.py`

**Verification:** `uv run ow benchmark --help`; pytest benchmark CLI tests.

### U3. Remove monolith and fix imports

**Goal:** Delete `src/cli/benchmark.py`; ensure `__init__.py` re-exports symbols tests import (`build_parser`, `run_calibrate_unified_tournament_cli`, etc.).

**Files:** delete `src/cli/benchmark.py`, `src/cli/__init__.py` unchanged if package path suffices

**Verification:** `make test-fast`; `uv run pytest tests/test_benchmark_cli.py tests/test_cli_benchmark_gate.py tests/test_agent_capability_map.py -q`

## Scope Boundaries

**In scope:** U3 only from parent plan 011.

**Deferred:** Other units (metric registry shard, eval lazy import, etc.).

## Risks

- Accidental help-string drift — mitigated by copying blocks verbatim and CLI tests.
- Circular imports — mitigated by lazy imports inside runners and parser-only imports in `parser.py`.

## Sources

- `docs/plans/2026-06-03-011-refactor-src-simplification-followup-plan.md` U3
- GitHub issue #194
- Pattern: `src/cli/benchmark_gates.py`
