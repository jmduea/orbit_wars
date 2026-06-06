---
date: 2026-06-06
topic: agent-native-operator-cli
status: consolidated
supersedes:
  - docs/archive/agent-native-phase2-status.md
  - docs/archive/agent-native-phase3-status.md
---

# Agent-native operator status

Consolidated ship tracker for Phases 1–3 of the agent-native operator CLI. Supersedes root `agent-native-phase2-status.md` and `agent-native-phase3-status.md` (archived under `docs/archive/`).

## Phase 2 — shipped

| Item | Notes |
| --- | --- |
| `ow eval jobs cancel` | `--all-queued`, `--job-id`, `--dry-run` |
| `ow eval status --watch` | Poll queue + last log marker |
| `make agent-context` extensions | Eval queue excerpt, git branch |
| `ow benchmark gate beat_noop` / `beat_random` | YAML recipes + dry-run |
| `ow benchmark gate curriculum_staged` | Gate 4 YAML recipe |
| `ow promote show` / `history` / `demote` | Promotion rollback primitives |
| AGENTS.md continual-learning | Learned sections policy |
| Calibrate → AGENTS.md thresholds | `<!-- preflight-thresholds -->` block refresh on default calibrate write |
| `ow eval results list\|show --run` | Glob `evaluations/**/manifest.json` |
| `ow runs watch --run` | Poll run status (queue + log marker) |
| Metric promotion terminal line | Mirrors tournament promote UX in training |
| `manifest.json` `produced_artifacts` | Appends checkpoint records at commit boundaries |

## Phase 3 — shipped

| # | Item | Evidence |
| --- | --- | --- |
| 1 | `PreflightGateSpec` from YAML | `conf/benchmark/gates/*.yaml`, `src/jax/preflight_gate_loader.py` |
| 2 | Atomic `ow benchmark` primitives | `ow benchmark gate run <name>`, `ow benchmark tournament-proof`; `learn-proof` thin composer (`src/cli/benchmark/`) |
| 3 | `ow sweep` unification | `src/cli/sweep.py` |
| 4 | Cursor session-start hook | `docs/CURSOR.md`, `.cursor/hooks.json` example — PR #180 |
| 5 | Seed-scheduler calibration + operator phase 2 | `training.reseed_every_updates: 50`; `docs/benchmarks/seed-scheduler-calibration.json`; PR #184 |

**Completed plans:** `docs/plans/2026-06-01-003-feat-seed-scheduler-calibration-plan.md`, `docs/plans/2026-06-02-015-feat-agent-native-audit-gaps-plan.md`, `docs/plans/2026-06-02-016-feat-agent-native-deferred-crud-plan.md`, `docs/plans/2026-06-02-017-feat-seed-u2-u3-capability-map-plan.md`.

## Operator primitives (prefer over workflows)

```bash
uv run ow benchmark gate run beat_noop --dry-run
uv run ow benchmark gate run beat_random
uv run ow benchmark tournament-proof --eval-checkpoint <pkl> --baselines noop
uv run ow sweep create --backend wandb --make wandb_sweep=planet_flow_ppo_signal_short
make agent-context
```

Phase 2 operator CRUD (PR #184) — details in `docs/solutions/developer-experience/seed-scheduler-calibration-agent-native-operator-phase2.md`:

```bash
uv run ow runs archive --run <path> --dry-run
uv run ow runs checkpoint delete --run <path> --checkpoint <pkl> --dry-run
uv run ow sweep cancel --backend wandb --sweep-id <id> --dry-run
uv run ow benchmark learn-proof --steps beat_noop,beat_random --eval-checkpoint <pkl>
uv run ow benchmark calibrate-seed-scheduler --analyze-only --dry-run
```

## Still open / deferred

- **Launch hygiene Phase B** — optional throughput recovery; ROADMAP Later only.
- **Planet Flow pipeline relaunch (U7)** — operator GPU after reachability mask.

## Related

- Phase 1 learning: `docs/solutions/developer-experience/agent-native-operator-cli-phase1.md`
- Calibration + phase-2 CLI: `docs/solutions/developer-experience/seed-scheduler-calibration-agent-native-operator-phase2.md`
- Historical backlog spec: `docs/plans/2026-06-02-agent-native-phase3-refactors.md`

## Verification

```bash
uv run ow benchmark gate --list
make test-fast
```
