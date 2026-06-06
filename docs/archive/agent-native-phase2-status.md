---
date: 2026-06-02
topic: agent-native-operator-cli
status: phase2-complete
origin: 2026-06-01-agent-native-operator-cli-plan.md
---

# Agent-native Phase 2 status

Phase 2 extends Phase 1 (`docs/plans/2026-06-01-agent-native-operator-cli-plan.md`) with composable benchmark gates, calibrate→AGENTS.md sync, and additional operator primitives.

## Done

| Item | Notes |
|------|-------|
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

## Shipped in Phase 3 (2026-06-03)

Large refactors from [`docs/plans/2026-06-02-agent-native-phase3-refactors.md`](plans/2026-06-02-agent-native-phase3-refactors.md) items 1–4 are on `main`. Seed-scheduler calibration and deferred operator CRUD closed in PR #184. See [`docs/agent-native-phase3-status.md`](agent-native-phase3-status.md) and [`docs/solutions/developer-experience/seed-scheduler-calibration-agent-native-operator-phase2.md`](solutions/developer-experience/seed-scheduler-calibration-agent-native-operator-phase2.md).

| Item | Status |
|------|--------|
| Full atomic split of `ow benchmark learn-proof` | Shipped — `ow benchmark gate run`, `tournament-proof`; `learn-proof --steps` |
| Move all `PreflightGateSpec` tuples to YAML | Shipped — `conf/benchmark/gates/*.yaml`, `preflight_gate_loader.py` |
| Cursor session-start hooks | Shipped — `docs/CURSOR.md`, PR #180 |
| `ow sweep` unification | Shipped — `src/cli/sweep.py` |
| Seed-scheduler calibration default | Shipped — `reseed_every_updates: 50` |
| Delete `.audit/` / `.omg/state/` | Still local gitignored operator data; docs-only cleanup optional |

## Verification

```bash
uv run ow benchmark gate --list
make test-fast
```

`.audit/` remains gitignored and local-only — not removed by Phase 2.
