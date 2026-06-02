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

## Deferred (not natural small slices)

| Item | Why deferred |
|------|----------------|
| Full atomic split of `ow benchmark learn-proof` | Large refactor across preflight train/eval paths |
| Move all `PreflightGateSpec` tuples to YAML | Overrides still in `src/jax/preflight.py`; YAML is metadata only |
| Cursor session-start hooks | Product-level; out of repo scope |
| `ow wandb sweep` / Kaggle create-sweep unification | Separate tooling surface |
| Delete `.audit/` / `.omg/state/` | Local gitignored operator data; docs-only cleanup |

## Verification

```bash
uv run ow benchmark gate --list
make test-fast
```

`.audit/` remains gitignored and local-only — not removed by Phase 2.
