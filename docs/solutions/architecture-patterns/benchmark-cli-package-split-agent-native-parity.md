---
title: Split ow benchmark CLI into package with agent-native parity
date: 2026-06-03
category: architecture-patterns
module: cli
problem_type: architecture_pattern
component: tooling
severity: medium
applies_when:
  - "Refactoring a large ow subcommand module without changing CLI behavior"
  - "Adding benchmark subcommands after a package split"
  - "Extending docs/AGENT_CAPABILITIES.md capability map rows"
  - "Agent-native review flags missing operator commands or stale help paths"
tags:
  - benchmark-cli
  - ow-cli
  - package-split
  - agent-native
  - capability-map
  - argparse
related_components:
  - src/cli/benchmark/
  - src/cli/benchmark_gates.py
  - docs/AGENT_CAPABILITIES.md
  - tests/test_agent_capability_map.py
  - tests/test_benchmark_cli.py
---

# Split ow benchmark CLI into package with agent-native parity

## Context

Issue [#194](https://github.com/jmduea/orbit_wars/issues/194) / PR [#202](https://github.com/jmduea/orbit_wars/pull/202) (branch `issue/194-benchmark-cli-package`) completes follow-up plan U3 from `docs/plans/2026-06-03-012-refactor-benchmark-cli-package-plan.md`: the ~1510-line monolith `src/cli/benchmark.py` was unmaintainable and blocked further operator CLI work.

Phase 1 src simplification (PR [#201](https://github.com/jmduea/orbit_wars/pull/201)) already extracted gates and other concerns; this change is **structural only** — same `ow benchmark` surface, help text, `LEARN_PROOF_PRIMITIVES`, and test import paths. Review follow-up commit `607e825` hardened agent-native discoverability (capability map rows, help registry tests, stale help strings).

Do not duplicate the phase 1 simplification narrative; use this doc when touching the benchmark CLI package or keeping the capability map aligned with argparse.

## Guidance

### Package layout (mirror `benchmark_gates.py`)

| Module | Role |
|--------|------|
| `common.py` | `REPO_ROOT`, `LEARN_PROOF_PRIMITIVES`, `_init_benchmark_runtime`, `print_benchmark_help` |
| `parser.py` | `build_parser()` — argparse tree only, no JAX |
| `training.py`, `sanity.py`, `calibrate.py`, `calibrate_seed.py`, `calibrate_unified.py`, `learn_proof.py`, `planet_flow.py`, `factorized.py`, `gate.py`, `tournament_proof.py` | `register_parser` + `run_*_cli` per command family |
| `__init__.py` | Re-export symbols tests import; `main()` dispatches on `args.command` |

**KTDs from the plan:**

1. **`src.cli.benchmark` is the package** — no shim file at `src/cli/benchmark.py`. `from src.cli import benchmark` and `from src.cli.benchmark import build_parser` keep working via the package.
2. **Thin `gate.py`** delegates to `src.cli.benchmark_gates.run_gate_cli`; do not duplicate gate logic in the package.
3. **Lazy JAX** — import JAX only inside command runners, not in `__init__.py` or `parser.py`.
4. **Factorized sampler** stays in-process via `src/jax/factorized_sampler_benchmark.py` (no subprocess reintroduction).

Wire `src/cli/__init__.py` to `benchmark.main` as before; delete the monolith only after re-exports match `tests/test_benchmark_cli.py` and gate tests.

### Preserve CLI contract byte-for-byte

- Copy help strings and `LEARN_PROOF_PRIMITIVES` verbatim when splitting; drift breaks `tests/test_benchmark_cli.py` and operator docs.
- `build_parser()` choice names and nesting must stay identical so existing Hydra override paths and CI gates unchanged.

### Agent-native parity after a split

1. **Capability map** — every non-`(planned)` `ow …` row in `docs/AGENT_CAPABILITIES.md` § Capability map must resolve in the live argparse tree. After adding commands (e.g. `calibrate-unified-tournament`, `shortlist-planet-flow-sweep`, `planet-flow-noop-smoke`), add table rows in the same PR.
2. **`tests/test_agent_capability_map.py`** — builds a help registry by walking `ow <cmd> --help` and nested `{choices}`. Extend `_EXTRA_NESTED_TOKENS` when subcommands are not expressed as argparse choices (e.g. `ow benchmark gate` → `list`, `run`). Skip `(planned)` rows in the table parser.
3. **Planned commands** — `shape-calibrate` is documented as planned but must **not** appear in benchmark `--help`; `test_shape_calibrate_not_registered_in_benchmark_help` guards against false positives where the registry would otherwise accept a non-existent subcommand.
4. **Help hygiene** — update `print_benchmark_help` and `docs/audits/agent-native-status.md` paths from `src/cli/benchmark.py` to `src/cli/benchmark/`; fix stale factorized-sampler help in `parser.py`; remove dead duplicate handlers in `gate.py`.

## Why This Matters

A monolith CLI file compounds merge conflicts and hides which commands own JAX imports. Agents rely on `docs/AGENT_CAPABILITIES.md` as the operator index; if the map lists commands that `--help` does not expose (or omits shipped commands), automation invokes the wrong primitive or fails mid-run.

Splitting without re-export discipline breaks tests that import `run_calibrate_unified_tournament_cli` from `src.cli.benchmark`. Agent-native tests are the guardrail that docs and argparse stay in sync without manual `ow benchmark --help` audits every PR.

## When to Apply

- Splitting any other large `src/cli/*.py` module into a package (eval, train helpers) using the same `parser.py` + per-command runners pattern.
- Adding a new `ow benchmark` subcommand after this refactor.
- Review feedback that capability map rows or help text drifted from implementation.
- Refreshing older solution docs that still cite `src/cli/benchmark.py` — update paths to `src/cli/benchmark/<module>.py`.

## Examples

**Import surface (unchanged for tests):**

```python
from src.cli.benchmark import build_parser, LEARN_PROOF_PRIMITIVES
from src.cli.benchmark import run_calibrate_unified_tournament_cli
```

**Dispatch in package `main()`:**

```python
match args.command:
    case "calibrate-unified-tournament":
        return run_calibrate_unified_tournament_cli(args)
    case "gate":
        return run_gate_cli(args)
    # ...
```

**Verification (targeted, then daily CPU):**

```bash
uv run pytest tests/test_benchmark_cli.py tests/test_cli_benchmark_gate.py tests/test_agent_capability_map.py -q
make test-fast
```

**Capability map: skip planned, assert shipped commands:**

```python
# tests/test_agent_capability_map.py — (planned) rows excluded from registration check
if "(planned)" in cells[0].lower():
    continue
```

## Related

- Plan: `docs/plans/2026-06-03-012-refactor-benchmark-cli-package-plan.md`
- GitHub: [#194](https://github.com/jmduea/orbit_wars/issues/194), [#202](https://github.com/jmduea/orbit_wars/pull/202)
- Agent-native operator phases: `docs/solutions/developer-experience/agent-native-operator-cli-phase1.md`, `docs/solutions/developer-experience/seed-scheduler-calibration-agent-native-operator-phase2.md`
- Benchmark subprocess UX (separate concern; update stale `src/cli/benchmark.py` refs when refreshing): `docs/solutions/developer-experience/benchmark-subprocess-training-observability.md`
- **Canonical training spine (SSOT):** `docs/solutions/architecture-patterns/ssot-training-pipeline-config-to-kaggle-submission.md`
- Gate 5 / unified tournament (legacy commands in package): `docs/solutions/architecture-patterns/gate5-unified-tournament-submit-valid-funnel.md`
