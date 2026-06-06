---
title: Seed-scheduler calibration (interval 50) and agent-native operator phase 2
date: 2026-06-03
last_updated: 2026-06-03
category: developer-experience
module: jax-training
problem_type: developer_experience
component: development_workflow
severity: medium
applies_when:
  - "Changing training.reseed_every_updates or validating reseed cadence against eval win rates"
  - "Agents need CRUD-style run cleanup, sweep cancel, or partial learn-proof gates without re-reading long workflow docs"
  - "Adding rows to docs/AGENT_CAPABILITIES.md capability map — must stay wired in ow --help"
tags:
  - seed-scheduler
  - reseed-every-updates
  - calibrate-seed-scheduler
  - agent-native
  - ow-cli
  - make-agent-context
  - capability-map
related_components:
  - conf/training/base.yaml
  - src/config/schema.py
  - src/jax/seed_scheduler_calibration.py
  - src/cli/benchmark/calibrate_seed.py
  - src/cli/runs.py
  - src/cli/sweep.py
  - scripts/agent_context.py
  - tests/test_agent_capability_map.py
---

# Seed-scheduler calibration (interval 50) and agent-native operator phase 2

## Context

Plan 003 left seed-scheduler calibration partially done: training arms and `ow benchmark calibrate-seed-scheduler` existed, but the default `training.reseed_every_updates` was still auto-scale (`-1`) until U1–U3 GPU sweeps and held-out tournament eval finished. In parallel, the 2026-06-02 agent-native audit (`docs/audits/agent-native-architecture-2026-06-02.md`) scored high CLI parity but weak CRUD, thin session context, and workflow-only benchmark entry points.

PR [#184](https://github.com/jmduea/orbit_wars/pull/184) (merge `191fef3` on `main`) closed plans 003, 015, 016, and 017: measured reseed interval **50**, locked defaults in Hydra/schema, and shipped deferred operator primitives plus a capability-map regression test.

Phase 1 operator CLI remains documented in `docs/solutions/developer-experience/agent-native-operator-cli-phase1.md` (PR #164). Subprocess progress during long calibrations is in `docs/solutions/developer-experience/benchmark-subprocess-training-observability.md`.

## Guidance

### Seed scheduler — calibrate, analyze, lock default

**Reproduce the sweep** (GPU; long JAX compile on first update — see observability doc):

```bash
uv run ow benchmark calibrate-seed-scheduler \
  --opponents noop_only,random_only,self_play_only \
  --reseed-intervals 0,25,50,100 \
  --total-updates 500
```

**Re-analyze existing run dirs** without retraining:

```bash
uv run ow benchmark calibrate-seed-scheduler --analyze-only
```

Canonical artifacts:

- Decision JSON: `docs/benchmarks/seed-scheduler-calibration.json` (`decision.chosen_interval: 50`, `chosen_effective_interval: 50`)
- Human summary: `docs/benchmarks/seed-scheduler-calibration.md`

**Locked defaults** after calibration:

- `conf/training/base.yaml`: `reseed_every_updates: 50`
- `src/config/schema.py`: dataclass default `50` with comment pointing at calibration JSON
- `AGENTS.md` seed-scheduler bullet cites the JSON and `ow benchmark calibrate-seed-scheduler` before changing the value

Semantics: reseed resets **rollout env state**, not only the PRNG key. Use `-1` for auto `max(25, total_updates // 10)`; `0` disables periodic reseed.

### Agent-native operator phase 2

| Capability | Command / hook | Notes |
|------------|----------------|-------|
| Richer session JSON | `make agent-context` | Preflight excerpt, roadmap, recent runs, **W&B sweep summary** (`wandb_sweeps`), GPU contention hint (`pgrep` patterns). **Enforcement:** project `beforeShellExecution` hook — [`cursor-before-shell-gpu-terminal-contention.md`](cursor-before-shell-gpu-terminal-contention.md) |
| Archive completed run | `ow runs archive --run <path> [--dry-run] [--confirm]` | Refuses if eval queue has active jobs; moves tree under `outputs/archived/` |
| Checkpoint delete | `ow runs checkpoint delete --run <path> --checkpoint <pkl> [--confirm]` | Blocks promoted incumbent; `--dry-run` previews |
| Cancel W&B sweep runs | `ow sweep cancel --backend wandb --sweep-id <id> [--dry-run]` | JSON lists `cancelled_run_ids` |
| Partial learn-proof ladder | `ow benchmark learn-proof --steps beat_noop,beat_random` | Comma-separated gate ids; mutually exclusive with `--gate` / `--through` |
| Capability map drift guard | `tests/test_agent_capability_map.py` | Every `ow …` row in `docs/AGENT_CAPABILITIES.md` § Capability map must resolve via `ow … --help` |

Prefer these **primitives** over monolithic workflows (`learn-proof` without `--steps`, hybrid promotion train) when automating agent loops — task prompts live in `docs/AGENT_CAPABILITIES.md`.

### Verification expectations

- `tests/test_seed_scheduler_calibration.py` asserts `decision["chosen_interval"] == 50` against committed JSON.
- `make test-fast` covers CLI/archive/sweep/agent-context tests added in PR #184.
- Re-run `ow benchmark calibrate-seed-scheduler` before changing the pinned interval; do not relax thresholds to make a run pass.

## Why This Matters

An uncalibrated reseed interval couples training stability to guesswork: too-frequent reseeds waste sample efficiency; too-rare reseeds let stale rollout state dominate PPO updates. Locking **50** from measured eval win rates (not self-play ~50% training metrics) keeps Hydra defaults honest.

Agent-native phase 2 closes the audit gap between “agents can invoke `ow train`” and “agents can **manage** campaign artifacts safely” — archive/delete/cancel without shell `rm`, plus `make agent-context` surfacing sweep state so parallel GPU work is visible before starting another calibration.

## When to Apply

- Before editing `training.reseed_every_updates` in `conf/` or schema defaults.
- When extending operator CLI: add subcommand in `src/cli/`, register in `src/cli/__init__.py`, document in `docs/AGENT_CAPABILITIES.md` capability map, and extend `tests/test_agent_capability_map.py` if the map lists the command.
- When changing the pinned reseed interval — re-run calibration and update `docs/benchmarks/seed-scheduler-calibration.json` before editing `conf/training/base.yaml` or schema defaults; see `docs/audits/agent-native-status.md` for the current shipped default (**50**).

## Examples

**Inspect calibration decision without GPU:**

```bash
uv run ow benchmark calibrate-seed-scheduler --analyze-only --dry-run
python3 -c "import json; print(json.load(open('docs/benchmarks/seed-scheduler-calibration.json'))['decision'])"
```

**Archive a finished run (preview then confirm):**

```bash
uv run ow runs archive --run outputs/campaigns/<c>/runs/<id> --dry-run
uv run ow runs archive --run outputs/campaigns/<c>/runs/<id> --confirm
```

**Run only Gates 2–3 via explicit steps:**

```bash
uv run ow benchmark learn-proof --steps beat_noop,beat_random --eval-checkpoint <pkl>
```

## Related

- PR [#184](https://github.com/jmduea/orbit_wars/pull/184) — shipped work
- Plans (completed): `docs/plans/2026-06-01-003-feat-seed-scheduler-calibration-plan.md`, `docs/plans/2026-06-02-015-feat-agent-native-audit-gaps-plan.md`, `docs/plans/2026-06-02-016-feat-agent-native-deferred-crud-plan.md`, `docs/plans/2026-06-02-017-feat-seed-u2-u3-capability-map-plan.md`
- Audit: `docs/audits/agent-native-architecture-2026-06-02.md`
- Phase 1 CLI: `docs/solutions/developer-experience/agent-native-operator-cli-phase1.md`
- Calibration subprocess UX: `docs/solutions/developer-experience/benchmark-subprocess-training-observability.md`
- Joint env shaping calibration (planned; extends this pattern to reward×opponent×reseed): `docs/solutions/developer-experience/shape-calibrate-env-shaping-calibration-operator.md` — ideation `docs/ideation/2026-06-03-searchable-measurable-env-shaping-ideation.md`, plan `docs/plans/2026-06-03-003-feat-shape-calibrate-plan.md`
- Benchmark CLI package (calibrate-seed-scheduler runner): `docs/solutions/architecture-patterns/benchmark-cli-package-split-agent-native-parity.md`
- Stash recovery during parallel branches: `docs/solutions/workflow-issues/git-stash-recovery-after-parallel-branch-cleanup.md`
