# Agent-native Phase 3 — status (2026-06-03)

Phase 3 backlog items from `docs/plans/2026-06-02-agent-native-phase3-refactors.md` that are **shipped on `main`**, plus seed-scheduler calibration closed in PR [#184](https://github.com/jmduea/orbit_wars/pull/184) (merge `191fef3`).

## Shipped

| # | Item | Evidence |
|---|------|----------|
| 1 | `PreflightGateSpec` from YAML | `conf/benchmark/gates/*.yaml`, `src/jax/preflight_gate_loader.py`; no tuple tables in `preflight.py` for gate recipes |
| 2 | Atomic `ow benchmark` primitives | `ow benchmark gate run <name>`, `ow benchmark tournament-proof`; `learn-proof` is thin composer (`src/cli/benchmark/`) |
| 3 | `ow sweep` unification | `src/cli/sweep.py`; deprecated paths noted in `ow train` / wandb help |
| 4 | Cursor session-start hook | `docs/CURSOR.md`, `.cursor/hooks.json` example — PR #180 |
| 5 | Seed-scheduler calibration + operator phase 2 | `training.reseed_every_updates: 50` in `conf/training/base.yaml` and `src/config/schema.py`; `docs/benchmarks/seed-scheduler-calibration.json` (`decision.chosen_interval: 50`); learning doc below |

**Completed plans:** `docs/plans/2026-06-01-003-feat-seed-scheduler-calibration-plan.md`, `docs/plans/2026-06-02-015-feat-agent-native-audit-gaps-plan.md`, `docs/plans/2026-06-02-016-feat-agent-native-deferred-crud-plan.md`, `docs/plans/2026-06-02-017-feat-seed-u2-u3-capability-map-plan.md`.

## Operator primitives (prefer over workflows)

Phase 3 benchmark / sweep entry points:

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
- **Planet Flow pipeline relaunch (U7)** — operator GPU after reachability mask; see reachability plan U7.

## Related

- **Canonical learning (calibration + phase-2 CLI):** `docs/solutions/developer-experience/seed-scheduler-calibration-agent-native-operator-phase2.md`
- Phase 2 status: `docs/agent-native-phase2-status.md`
- Phase 1 learning: `docs/solutions/developer-experience/agent-native-operator-cli-phase1.md`
- Planet Flow residuals closed: #168–#170 (see PR closing those issues)
