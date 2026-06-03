# Agent-native Phase 3 — status (2026-06-02)

Phase 3 backlog items from `docs/plans/2026-06-02-agent-native-phase3-refactors.md` that are **shipped on `main`**:

| # | Item | Evidence |
|---|------|----------|
| 1 | `PreflightGateSpec` from YAML | `conf/benchmark/gates/*.yaml`, `src/jax/preflight_gate_loader.py`; no tuple tables in `preflight.py` for gate recipes |
| 2 | Atomic `ow benchmark` primitives | `ow benchmark gate run <name>`, `ow benchmark tournament-proof`; `learn-proof` is thin composer (`src/cli/benchmark.py`) |
| 3 | `ow sweep` unification | `src/cli/sweep.py`; deprecated paths noted in `ow train` / wandb help |
| 4 | Cursor session-start hook | `docs/CURSOR.md`, `.cursor/hooks.json` example — PR #180 |

## Operator primitives (prefer over workflows)

```bash
uv run ow benchmark gate run beat_noop --dry-run
uv run ow benchmark gate run beat_random
uv run ow benchmark tournament-proof --eval-checkpoint <pkl> --baselines noop
uv run ow sweep create --backend wandb --make wandb_sweep=planet_flow_ppo_signal_short
make agent-context
```

## Still open / deferred

- **Seed scheduler calibration** — GPU sweep + tournament eval; plan `docs/plans/2026-06-01-003-feat-seed-scheduler-calibration-plan.md` remains `active`.
- **Launch hygiene Phase B** — optional throughput recovery; ROADMAP Later only.
- **Planet Flow pipeline relaunch (U7)** — operator GPU after reachability mask; see reachability plan U7.

## Related

- Phase 2: `docs/agent-native-phase2-status.md`
- Planet Flow residuals closed: #168–#170 (see PR closing those issues)
