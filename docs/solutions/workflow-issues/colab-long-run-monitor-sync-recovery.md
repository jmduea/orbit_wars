---
title: Colab long-run monitor, sync tolerance, and recovery
date: 2026-06-16
category: workflow-issues
module: orchestration
problem_type: workflow_issue
component: tooling
severity: medium
applies_when:
  - "Launching multi-hour Colab training after local W&B preflight (Colab v1 is training-only)"
  - "Using --monitor-after-launch or restarting ow train colab monitor after terminal loss"
  - "Sync archive exec is slow while the kernel is busy during long runs"
  - "Recovering artifacts from a dead, OOM, or stopped session via partial sync harvest"
  - "Selecting the fixed-path pilot recipe instead of map_pool or plain defaults for first long run"
symptoms:
  - "Monitor marks run stale while training is still healthy because sync archive exec exceeded 120s"
  - "Long run OOM on T4 when launched with plain defaults instead of the known-good fixed-path pilot overrides"
  - "Checkpoints and metrics only discoverable after manual sync once monitor or session dies"
  - "Preflight shortlist built from sweep 85u3e192 selects wrong opponent mix (production_mix / latest self-play, not noop/random)"
tags:
  - colab
  - long-run
  - monitor-after-launch
  - sync-tolerance
  - partial-sync-harvest
  - dead-session-recovery
  - fixed-path-pilot
  - preflight-sweep
related_components:
  - src/orchestration/colab_runner.py
  - src/cli/colab_runner.py
  - src/cli/train_hosts.py
  - docs/colab_runner.md
  - COLAB_LAUNCH_AND_INTEGRATION_PROMOTION.md
  - conf/wandb_sweep/fixed/preflight.yaml
  - src/jax/train/sweep_score.py
---

# Colab long-run monitor, sync tolerance, and recovery

## Context

Orbit Wars uses **`ow train colab`** for remote GPU long runs after **local** W&B preflight sweeps. The June 2026 operator arc exposed a gap between “Colab host works” (U6 smoke proof, June 7) and “Colab long run succeeds end-to-end”: sync archive timeouts under busy kernels, no automatic checkpoint pull/eval during multi-hour runs, wrong sweep/recipe choices wasting credits, GPU OOM from default geometry, and W&B tag side effects on non-preflight runs.

Prior architecture doc [`colab-train-host-preflight-long-run.md`](../architecture-patterns/colab-train-host-preflight-long-run.md) covers the train-host shape (tarball bootstrap, shortlist workflow). **This doc captures June 2026 long-run operator learnings** that doc does not cover.

Three commits landed the operational fixes:

| Commit | Change |
|--------|--------|
| `2ceca69` | `ow train colab monitor` + `--monitor-after-launch` — poll status, sync, stale detection, local checkpoint eval |
| `bca7074` | Sync archive `colab exec` timeout raised from fixed 120s to `max(120, min(timeout, 600))` so busy kernels do not false-trigger stale alerts |
| `ea13249` | Tracker doc + integration promotion readiness script (promotion gates; not Colab runtime) |

Session history (June 14–16) adds failed approaches: `task=map_pool` + `production_mix` pilots showed operational PASS but learning FAIL; auto long run with plain defaults OOM'd at update 47 on T4; sweep `85u3e192` was invalidated after verification it fixed `production_mix` (latest self-play) instead of noop/random recovery — corrected sweep is `0mn8n6g0`.

## Guidance

### 1. Always use `--monitor-after-launch` for long runs

Do not launch a multi-hour Colab run and walk away expecting a final `sync` at the end. Pass **`--monitor-after-launch`** so the launcher enters the monitor loop immediately after the worker bootstrap succeeds.

The monitor loop (`monitor_once`) on each interval:

1. **`colab status`** — surface whether the VM still exists.
2. **`colab sync`** — remote tarball archive + download into `outputs/colab_runner/synced/<campaign>/`; exec also keeps the session from going idle.
3. **Stale check** — read newest synced `logs/*_jax.jsonl` and checkpoint mtimes; flag stale when no activity for `--stale-seconds` (default 900).
4. **Local checkpoint eval** — run `ow eval tournament` on newly synced numbered checkpoints; results under `outputs/colab_runner/monitor/evals/`; state in `outputs/colab_runner/monitor/<session>.json` so restarts skip already-evaluated ckpts.

Recommended defaults for long runs:

- `--interval-seconds 300` (5 min poll)
- `--stale-seconds 900` (15 min no log/ckpt change → stale)
- `--eval-baselines noop,random,sniper --eval-seeds 0,1,2,3,4 --eval-formats 2p_vs_baseline`

If the monitor terminal closes, **restart against the same session slug** — do not relaunch a duplicate worker unless the session is dead.

### 2. Busy sync archive calls (fix `bca7074`)

`sync()` archives the remote campaign directory via a small Python script uploaded with `colab exec -f`. Under load (large checkpoints, concurrent training I/O), a **120s exec timeout** caused `colab sync archive failed` and monitor `sync_error` stale reasons even though training was healthy.

Fix: archive exec timeout is now **`max(120, min(int(request.timeout), 600))`** — up to 10 minutes for long-run launch defaults (`--timeout 86400` → 600s cap). Pass a generous `--timeout` on launch/sync/monitor so sync inherits the higher bound.

**Symptom:** monitor JSON shows `"sync_error": "colab sync archive failed: …"` and `"stale_reasons"` includes `"sync_error"` while W&B still shows live metrics.

**Action:** upgrade to `bca7074+`; ensure launch `--timeout` is not left at a short smoke value for long runs.

### 3. Dead Colab session recovery

Colab VMs can die (OOM, preemption, idle prune, API failure). Partial progress is recoverable if sync ran at least once.

**Synced artifact layout:**

```
outputs/colab_runner/synced/<campaign>/
  runs/<run_id>/
    checkpoints/jax_ckpt_*.pkl
    logs/*_jax.jsonl
  worker-summary.json   # when download succeeds
```

**Recovery paths:**

| Situation | Action |
|-----------|--------|
| Session dead, partial sync exists | Inspect `ow runs show --run outputs/colab_runner/synced/<campaign>/runs/<run_id>`; use last numbered ckpt or `jax_ckpt_last.pkl` |
| Resume training locally | `resume_checkpoint=outputs/colab_runner/synced/.../checkpoints/jax_ckpt_last.pkl` (Hydra override on `ow train`) |
| Resume on fresh Colab VM | **Not supported in v1** — Colab launch does not upload local checkpoints; relaunch with proven fixed-path overrides and train from scratch, or resume locally via `resume_checkpoint` on `ow train` |
| Session may still be alive | `ow train colab status --session <slug>` then restart `ow train colab monitor --session <slug> …` |

Do not assume the remote run dir survives after VM loss — **local synced copy is the audit source** for packaging, gates, and resume. Both W&B (live monitoring) and synced local campaign outputs (audit/package/tournament) are required (session history).

### 4. Failed long-run recipes (June 2026 evidence)

**Do not launch full long runs from these shapes without new pilot evidence:**

| Recipe | Outcome | Why it failed |
|--------|---------|---------------|
| `task=map_pool` + `production_mix` + `2p4p_32_split` (pilots `colab_pilot`, `colab_pilot_iter2`) | Operational PASS, **learning FAIL/ambiguous** | `production_mix` → latest self-play; `overall_win_rate` and reward trends worsened over 50–100 updates (session history) |
| Auto long run `colab_fixed_path_long_auto` (defaults, not pilot recipe) | **OOM at update 47** | Default/random-heavy geometry on T4; ~10.1 GiB alloc failure copying PPO metrics to host before ckpt 50 |

**Known-good fixed-path pilot recipe** (synced overrides from `colab_fixed_path_pilot`; long run `colab_fixed_path_long_pilot_recipe` reached strong noop/random eval by update 96):

- `training=2p_32` (not `2p4p_32_split` for this credit budget)
- `curriculum=scripted_heavy`
- `model=transformer_factorized_small`, `model.max_moves_k=3`
- Shaped rewards + `task.trajectory_shield_mode=tiered`
- `training.rollout_steps=512`, `training.reseed_every_updates=100`
- `artifacts=disabled` (or artifact pipeline off), replay off
- `artifacts.checkpoint_every=50`
- W&B enabled for live monitoring — **without** `preflight` tag (see §6)

Copy overrides from a successful pilot's `.hydra/overrides.yaml` rather than hand-assembling from memory.

### 5. Preflight sweep selection

Shortlist **only** from sweeps that test **noop/random recovery**, not self-play production mix.

| Sweep ID | Status | Notes |
|----------|--------|-------|
| `85u3e192` | **Invalidated** | Fixed axis `production_mix` → latest self-play; final `preflight_sweep_score=-1.0`; do not use for Colab launch (session history) |
| `0mn8n6g0` | **Use this** | Corrected recipe (`curriculum=noop_only` in `conf/wandb_sweep/fixed/preflight.yaml`); noop recovery objective |

Workflow:

```bash
uv run ow train colab shortlist --sweep-id 0mn8n6g0 \
  --out outputs/colab_runner/shortlist.json

uv run ow train colab launch \
  --from-shortlist outputs/colab_runner/shortlist.json --rank 0 \
  --gpu T4 --timeout 86400 \
  --monitor-after-launch \
  --interval-seconds 300 --stale-seconds 900 \
  training.total_updates=2000 \
  training=2p_32 \
  curriculum=scripted_heavy \
  model=transformer_factorized_small \
  model.max_moves_k=3 \
  task.trajectory_shield_mode=tiered \
  training.rollout_steps=512 \
  training.reseed_every_updates=100 \
  artifacts=disabled \
  artifacts.checkpoint_every=50 \
  output.campaign=colab_long
```

Merge rule: shortlist supplies swept PPO hyperparams; **fixed-path geometry overrides above must still be present** (CLI wins on same-prefix keys).

**Gate before full run:** pilot must pass operational sync/W&B/checkpoint gates **and** show interpretable learning on the intended opponent mix. Two failed map_pool+production_mix pilots triggered reassessment — do not skip straight to 2000 updates.

### 6. W&B tags — do not tag long runs with `preflight`

`is_preflight_sweep()` returns true when **`preflight` appears in `telemetry.wandb.tags`**, which activates `PreflightSweepScoreTracker` and preflight guardrail metrics in the training loop.

That is correct for **local 100-update sweep agents** (recipe sets `tags: [preflight, v2, …]`). It is **wrong for Colab long runs**:

- Long-run metrics get preflight sweep scoring semantics (`preflight_sweep_score` stays `-1` until window 10 or when guardrails fail).
- Misleading W&B dashboards and operator confusion.

**Do:** `telemetry.wandb.group=colab_long` (or campaign-specific group), tags like `colab`, `long-run`, `fixed-path-pilot`.

**Do not:** copy `telemetry.wandb.tags=[preflight]` from sweep YAML onto Colab long runs — the preflight tag activates `PreflightSweepScoreTracker` on non-sweep runs.

## Why This Matters

Colab credits and wall clock are scarce. Without monitor-after-launch, operators lose checkpoints when VMs die silently, discover learning failure only after sync, and waste days on recipes that failed cheap pilots. Without corrected sweep selection, `preflight_sweep_score` optimizes the wrong opponent mix. Without sync timeout tolerance, healthy runs look stale and trigger panic stops. Without W&B tag discipline, long-run metrics lie about preflight eligibility.

The June 2026 sequence — failed map_pool pilots → invalidated sweep → OOM auto long run → fixed-path pilot recipe with monitor — is the template for future Colab long runs.

## When to Apply

- Any Colab run expected to exceed ~1 hour or 100 updates.
- After changing sync/monitor code or upgrading `google-colab-cli`.
- When choosing between shortlist-driven launch vs hand-picked overrides (prefer shortlist from **`0mn8n6g0`-class** noop-recovery sweeps, then validate with a medium pilot).
- When resuming after session death, API 503, or operator terminal loss.
- **Not** for local preflight sweeps, admission gates, Docker packaging, or Kaggle submit — those stay local per existing Colab scope boundaries.

## Examples

### Canonical long run (monitor + fixed-path overrides)

```bash
uv run ow train colab launch --gpu T4 --timeout 86400 \
  --monitor-after-launch \
  --interval-seconds 300 \
  --stale-seconds 900 \
  --eval-baselines noop,random,sniper \
  --eval-seeds 0,1,2,3,4 \
  --eval-formats 2p_vs_baseline \
  training.total_updates=2000 \
  training=2p_32 \
  curriculum=scripted_heavy \
  model=transformer_factorized_small \
  model.max_moves_k=3 \
  task.trajectory_shield_mode=tiered \
  training.rollout_steps=512 \
  training.reseed_every_updates=100 \
  artifacts=disabled \
  artifacts.checkpoint_every=50 \
  output.campaign=colab_fixed_path_long \
  telemetry.wandb.enabled=true \
  telemetry.wandb.group=colab_fixed_path_long
```

### Restart monitor after terminal interrupt

```bash
uv run ow train colab monitor --session ow-colab_fixed_path_long-<sha> \
  --interval-seconds 300 \
  --stale-seconds 900 \
  --eval-baselines noop,random,sniper \
  --eval-seeds 0,1,2,3,4 \
  --eval-formats 2p_vs_baseline
```

### One-shot sync (recovery or cron)

```bash
uv run ow train colab sync --session ow-colab_fixed_path_long-<sha> --timeout 86400
uv run ow runs show --run outputs/colab_runner/synced/colab_fixed_path_long/runs/<run_id>
```

### Resume locally from synced checkpoint

```bash
uv run ow train \
  resume_checkpoint=outputs/colab_runner/synced/colab_fixed_path_long/runs/<run_id>/checkpoints/jax_ckpt_last.pkl \
  training.total_updates=2000 \
  output.campaign=colab_resume_local
```

### Monitor options reference

| Flag | Purpose |
|------|---------|
| `--once` | Single sync/eval/stale pass |
| `--max-iterations N` | Bounded watch loop |
| `--no-eval-checkpoints` | Liveness/sync only |
| `--stop-on-stale` | Stop Colab VM on stale (use sparingly; prefer manual investigation) |
| `--eval-write-replays` | HTML replays during monitor eval (large disk) |

Eval artifacts: `outputs/colab_runner/monitor/evals/<run_id>/jax_ckpt_<n>/`. Monitor state: `outputs/colab_runner/monitor/<session>.json`.

### Anti-patterns

**Wrong:** Launch 2000-update `task=map_pool` + `production_mix` because pilots “proved Colab works.”

**Right:** Operational pilot ≠ learning pilot; use noop-recovery preflight shortlist + fixed-path recipe with scripted curriculum.

**Wrong:** `telemetry.wandb.tags=[preflight]` on Colab long runs.

**Right:** Campaign-specific W&B group; no `preflight` tag unless running a local sweep agent.

**Wrong:** Ignore `sync_error` in monitor JSON during heavy checkpoint writes on pre-`bca7074` code.

**Right:** Upgrade sync timeout fix; increase `--timeout`; distinguish sync failures from true training stall via W&B.

**Wrong:** Pipe `ow train colab launch` through `tail`/`head`.

**Right:** JSON on stdout; follow monitor state at `outputs/colab_runner/monitor/<session>.json` or restart `ow train colab monitor --session <slug>`.

## Related

- Operator reference: [`docs/colab_runner.md`](../../colab_runner.md) — monitor section
- Launch tracker: [`COLAB_LAUNCH_AND_INTEGRATION_PROMOTION.md`](../../../COLAB_LAUNCH_AND_INTEGRATION_PROMOTION.md) — pilot gates, sweep invalidation, June 16 long-run state
- Train host architecture: [`colab-train-host-preflight-long-run.md`](../architecture-patterns/colab-train-host-preflight-long-run.md)
- CLI observability (no tail pipe): [`ow-long-cli-stderr-progress-no-tail-pipe.md`](../developer-experience/ow-long-cli-stderr-progress-no-tail-pipe.md) — prefer `ow train colab monitor` over piping `launch`
- Sweep selection anti-patterns (generalized): [`planet-flow-sweep-gameable-objective.md`](../logic-errors/planet-flow-sweep-gameable-objective.md)
- Preflight sweep recipe: [`conf/wandb_sweep/fixed/preflight.yaml`](../../../conf/wandb_sweep/fixed/preflight.yaml)
- Preflight score tracker: `src/jax/train/sweep_score.py` (`is_preflight_sweep`, `PreflightSweepScoreTracker`)
