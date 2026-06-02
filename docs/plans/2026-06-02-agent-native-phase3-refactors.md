---
date: 2026-06-02
topic: agent-native-operator-cli
status: scheduled
origin: docs/agent-native-phase2-status.md
depends_on: agent-native-phase2 (PR)
---

# Agent-native Phase 3 — large refactors (scheduled)

Phase 2 shipped composable gate **recipes** and operator **primitives**; Phase 3 decomposes remaining workflow monoliths and externalizes gate specs. **No implementation in Phase 2 PR** — this document is the ordered backlog.

## Suggested order

| # | Item | Effort | Depends on | Blocks |
|---|------|--------|------------|--------|
| 1 | `PreflightGateSpec` → YAML (train overrides) | **L** | Phase 2 gate YAML metadata | #2 |
| 2 | Atomic `ow benchmark learn-proof` split | **L** | #1 (shared gate loader) | CI/Makefile ladder |
| 3 | `ow wandb sweep` / Kaggle create-sweep unification | **M** | — | W&B + Kaggle operators |
| 4 | Cursor session-start hook (product) | **S** (repo docs only) | Cursor hooks API | Agent onboarding |

Rationale: gate YAML first so learn-proof composers read one source of truth; sweep unification is parallelizable once #1–#2 contracts are stable; session-start hook is mostly Cursor product config with a thin repo doc + optional `.cursor/hooks` example.

---

## 1. Move `PreflightGateSpec` tuples to YAML

**Problem:** `conf/benchmark/gates/*.yaml` describes gates for agents (`ow benchmark gate --dry-run`) but train-time overrides and window semantics still live as tuples in `src/jax/preflight.py`.

**Scope:**
- Define schema for train overrides (metric keys, window, min/max, optional Hydra override fragments).
- Load gate definitions from YAML at preflight boundary; delete duplicated tuple tables where YAML is authoritative.
- Keep `docs/benchmarks/preflight-calibration.json` as threshold source; YAML references gate names + metric paths only.

**Acceptance criteria:**
- `uv run ow benchmark gate --list` and dry-run for all gates read the same YAML used by JAX preflight.
- `make test-fast` green; extend `tests/test_cli_benchmark_gates.py` (or domain benchmark tests) for loader round-trip.
- `AGENTS.md` / `docs/AGENT_CAPABILITIES.md` state: no manual edits to tuple tables in `preflight.py` for new gates.

**Effort:** L (touch preflight train path, gate registry, tests).

---

## 2. Atomic split of `ow benchmark learn-proof`

**Problem:** `learn-proof` is a workflow wrapper (Gates 2–5: calibrate trend, curriculum, tournament proof). Phase 2 added `ow benchmark gate <name>` recipes but the ladder still invokes monolithic code paths.

**Scope:**
- Primitives: `ow benchmark gate run <name>`, `ow benchmark preflight-train` (or reuse existing train smoke entry), `ow benchmark tournament-proof` — each independently invocable with `--dry-run` and JSON status exit codes.
- Composer: `ow benchmark learn-proof` becomes thin orchestration calling primitives in order; `make preflight-learn-proof` documents primitive sequence for agents.
- Preserve `docs/benchmarks/preflight-calibration.json` threshold injection and `make agent-context` excerpt.

**Acceptance criteria:**
- Agent can run Gate 3 only without running Gate 5 tournament (documented in `docs/AGENT_CAPABILITIES.md`).
- `ow benchmark learn-proof` behavior unchanged for CI/humans (same default ladder).
- `make test-fast` + targeted benchmark CLI tests; no new slow tier unless user requests GPU proof run.

**Effort:** L (depends on #1 for shared gate config).

---

## 3. Unify `ow wandb sweep` and Kaggle create-sweep

**Problem:** W&B sweep creation and Kaggle notebook/sweep flows use separate CLIs and docs; agents duplicate env and campaign naming.

**Scope:**
- Single `ow sweep` (or `ow orchestration sweep`) subcommand group: `create`, `status`, `list` with `--backend wandb|kaggle`.
- Shared campaign/run naming via `src/artifacts/run_paths.py` conventions.
- Deprecate parallel entrypoints with one-release warning strings.

**Acceptance criteria:**
- `ow sweep --help` documents both backends with copy-paste examples.
- Existing W&B and Kaggle scripts either delegate to `ow` or are removed (no parallel `scripts/*.py` for new work per `AGENTS.md`).
- Docs: link from `docs/AGENT_CAPABILITIES.md` and planet-flow sweep plans.

**Effort:** M (mostly CLI + orchestration; minimal JAX).

---

## 4. Cursor session-start hook

**Problem:** Agents cold-start without `make agent-context` unless prompted; session-start is product-level.

**Scope (repo):**
- Document recommended hook: run `make agent-context` (or `uv run python scripts/agent_context.py`) on session start.
- Optional example hook under `docs/CURSOR.md` — not required to commit `.cursor/hooks` secrets or user-specific paths.
- Do **not** delete `.audit/` or `.cursor/hooks/state/` in automation.

**Acceptance criteria:**
- `docs/CURSOR.md` has copy-paste hook snippet and failure modes (missing `outputs/`, no GPU contention note).
- No enforcement test in repo (human/product choice).

**Effort:** S for docs; **M** if shipping a default hook template in-repo.

---

## Verification baseline (Phase 2 PR)

```bash
make test-fast   # 495 passed locally on 2026-06-02
uv run ow benchmark gate --list
```

## References

- Phase 2 completion: `docs/agent-native-phase2-status.md`
- Phase 1 plan: `docs/plans/2026-06-01-agent-native-operator-cli-plan.md`
- Gate recipes: `conf/benchmark/gates/*.yaml`
