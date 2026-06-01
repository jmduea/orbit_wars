---
date: 2026-06-01
topic: agent-native-operator-cli
status: completed
origin: agent-native-architecture-audit-2026-06-01
---

# Plan: Agent-Native Operator CLI (Phase 1)

## Summary

Ship Phase 1 of the agent-native audit: operator-facing docs, session context (`make agent-context`), run introspection (`ow runs`), eval queue status (`ow eval status`), CLI discovery UX (`ow --help`, empty states, `make help`), and training/worker observability banners—without refactoring benchmark workflows or preflight gate YAML.

## Problem Frame

Coding agents share the same `ow` CLI and `outputs/` tree as humans, but lack first-class **list/show/status** commands, dynamic session context, and immediate terminal feedback for async artifact work. The 2026-06-01 agent-native audit scored CRUD (0/8), context injection (4/8), and UI integration (10/18) lowest among operator concerns.

## Requirements

- **R1** — `docs/AGENT_CAPABILITIES.md` exists with task-oriented workflows, copy-paste prompts, and links to `AGENTS.md`, `conf/README.md`, preflight docs.
- **R2** — `make help` prints test tiers, domain targets, preflight shortcuts, and `make agent-context`.
- **R3** — `make agent-context` emits JSON (stdout) with: preflight threshold excerpt from `docs/benchmarks/preflight-calibration.json`, ROADMAP Now/Next lines, latest N entries from `outputs/indexes/runs.jsonl` when present, and repo-relative pointers (no secrets).
- **R4** — `ow runs list` and `ow runs show --run <path>` read campaign manifests / run dirs; `ow runs logs --run <path> [--tail N]` tails `logs/*_jax.jsonl`.
- **R5** — `ow eval status --run <path>` summarizes queue job JSON statuses, latest promotion manifest path, and last JSONL event type from lean log when readable.
- **R6** — `print_ow_help()` links `ow benchmark --help`, `ow eval --help`, `docs/AGENT_CAPABILITIES.md`; unknown commands suggest `ow --help`.
- **R7** — `ow eval` / `ow benchmark` with no subcommand print subcommand list + example (exit 0), not argparse error.
- **R8** — Local train prints one startup banner (run_dir, log_path, queue_dir, wandb on/off) and one completion summary (final update, log_path, checkpoint hint).
- **R9** — `ow eval worker --verbose` prints per-job start/done lines; autostart worker prints one notice with log path.
- **R10** — Fix doc drift: ONBOARDING canonical `ow train`, README link to `conf/README.md` not missing `config/GUIDANCE.md`; fix `.cursor/rules/orbit-wars.mdc` broken fence; add short **config vs code** table to `AGENTS.md`.
- **R11** — All new CLI registered in `src/cli/__init__.py`; tests in `tests/test_cli_runs.py`, `tests/test_cli_eval_status.py`, extend `tests/test_cli_train_hosts.py` for help strings.
- **R12** — Verification: `make test-domain-config` + targeted new tests; no slow/JAX smokes in this phase.

## Key Technical Decisions

**KTD1 — Phase 1 only; defer benchmark decomposition.** `ow benchmark learn-proof` / `calibrate` stay workflow commands; no `conf/benchmark/gates/` migration in this plan. (see audit items 4, 6, 9)

**KTD2 — `ow runs` as thin filesystem introspection.** No new database; parse `manifest.json`, `campaign_manifest.json`, glob `outputs/campaigns/*/runs/*`. Implementation in `src/cli/runs.py` with argparse subparsers.

**KTD3 — `make agent-context` as Makefile + small Python script under `scripts/agent_context.py`.** Keeps logic testable; Makefile target documents agent usage. Script must run without JAX import.

**KTD4 — JSON stdout for machine-readable context.** Human-readable sections optional via `--format text`; default `json` for agents.

**KTD5 — Branch isolation.** Implement on `feat/agent-native-operator-cli` branched from current default to avoid mixing with in-progress seed-scheduler calibration edits on `feat/seed-scheduler-calibration`.

## Scope Boundaries

**In scope:** R1–R12 above.

**Deferred (Phase 2):**
- Split `ow benchmark` into atomic primitives + CI composers
- `conf/benchmark/gates/*.yaml` for preflight recipes
- `ow eval jobs cancel`, `ow promote demote`
- Auto-regenerate threshold bullets into `AGENTS.md` on calibrate
- Remove `.audit/` / `.omg/state/` legacy dirs

**Out of scope:** Web UI, MCP server restore, W&B/Kaggle browser automation.

## Implementation Units

### U1 — Documentation and Makefile discovery

**Files:** `docs/AGENT_CAPABILITIES.md`, `Makefile`, `AGENTS.md`, `docs/ONBOARDING.md`, `README.md`, `.cursor/rules/orbit-wars.mdc`

**Test scenarios:** Manual review; `make help` runs; link check for README.

**Verification:** `make help`

### U2 — `scripts/agent_context.py` + `make agent-context`

**Files:** `scripts/agent_context.py`, `Makefile`, `tests/test_agent_context.py`

**Test scenarios:** Missing indexes file → empty list; calibration JSON parsed; ROADMAP excerpt capped.

**Verification:** `uv run pytest tests/test_agent_context.py -q`

### U3 — `ow runs` subcommands

**Files:** `src/cli/runs.py`, `src/cli/__init__.py`, `tests/test_cli_runs.py`

**Test scenarios:** list with fixture tmp campaign dir; show manifest fields; logs tail returns last lines.

**Verification:** `uv run pytest tests/test_cli_runs.py -q`

### U4 — `ow eval status` + worker verbose

**Files:** `src/cli/eval.py`, `src/jax/train/queue.py`, `src/jax/train/loop.py`, `tests/test_cli_eval_status.py`

**Test scenarios:** status with queued/running job fixtures; verbose worker prints job kind.

**Verification:** `uv run pytest tests/test_cli_eval_status.py tests/test_artifact_pipeline.py -q -k "not slow"`

### U5 — Help UX and train banners

**Files:** `src/cli/train_hosts.py`, `src/cli/__init__.py`, `src/cli/benchmark.py`, `tests/test_cli_train_hosts.py`

**Test scenarios:** eval/benchmark no-args help; unknown command message; help mentions AGENT_CAPABILITIES.

**Verification:** `uv run pytest tests/test_cli_train_hosts.py -q`

## Dependencies and sequencing

U1 → U2 → U3 → U4 → U5 (U3/U4 can parallelize after U2).

## Risks

- WIP on current branch: use **KTD5** dedicated branch.
- `outputs/indexes/runs.jsonl` may be absent on fresh clones — script must tolerate.

## Verification (plan-level)

```bash
make test-domain-config
uv run pytest tests/test_cli_runs.py tests/test_cli_eval_status.py tests/test_agent_context.py tests/test_cli_train_hosts.py -q
```
