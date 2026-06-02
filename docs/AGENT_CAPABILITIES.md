# Agent capabilities (Orbit Wars)

Task-oriented guide for coding agents. Canonical policy: `AGENTS.md`. Architecture: `docs/ONBOARDING.md`.

## Session context

```bash
make agent-context          # JSON: git branch, preflight thresholds, roadmap, recent runs, latest eval queue
uv run ow train print_resolved_config=true
```

## Command tiers

| Tier | Examples | Use when |
|------|----------|----------|
| **Primitive** | `ow runs list`, `ow eval status`, `ow eval jobs cancel`, `ow promote demote`, `ow benchmark gate beat_noop`, `ow benchmark gate beat_random` | Inspect or mutate one artifact; compose in agent scripts |
| **Workflow** | `ow benchmark learn-proof`, `make preflight-learn-proof`, `ow train ... artifacts=hybrid_promotion` | Human/CI end-to-end gates; prefer primitives for targeted agent loops |

## Train

```bash
uv run ow train training=smoke training.total_updates=10 curriculum=off
uv run ow train print_resolved_config=true
uv run ow train kaggle preflight
```

After a run, inspect:

```bash
uv run ow runs list --limit 10
uv run ow runs show --run outputs/campaigns/<campaign>/runs/<run_id>
uv run ow runs logs --run outputs/campaigns/<campaign>/runs/<run_id> --tail 5
```

## Verify (tests)

```bash
make test-fast
make test-domain-config    # after conf/ or config schema edits
make test-domain-artifacts # after artifact / eval CLI edits
```

## Preflight / benchmarks

```bash
make preflight-sanity
make preflight-learn-proof   # GPU/time; check terminals first
make preflight-calibrate
```

**Composable gate (Phase 2):**

```bash
uv run ow benchmark gate --list
uv run ow benchmark gate beat_noop --dry-run
uv run ow benchmark gate beat_noop --out /tmp/beat_noop.json
uv run ow benchmark gate beat_random --dry-run
uv run ow benchmark gate beat_random --out /tmp/beat_random.json
```

Gate recipes: `conf/benchmark/gates/*.yaml` (`beat_noop`, `beat_random`; `curriculum_staged` deferred). Train overrides still live in `src/jax/preflight.py` — YAML is metadata for agents. Full ladder remains `ow benchmark learn-proof`.

Thresholds: `docs/benchmarks/preflight-calibration.json` (never invent gate numbers).

## Eval & promotion

```bash
uv run ow eval status --run outputs/campaigns/<c>/runs/<id>
uv run ow eval status --run outputs/campaigns/<c>/runs/<id> --watch
uv run ow eval status --run outputs/campaigns/<c>/runs/<id> --watch --idle-exit-seconds 30
uv run ow eval jobs cancel --run outputs/campaigns/<c>/runs/<id> --all-queued --dry-run
uv run ow eval jobs cancel --run outputs/campaigns/<c>/runs/<id> --job-id <uuid>
uv run ow eval worker --run outputs/campaigns/<c>/runs/<id> --verbose
uv run ow eval tournament --checkpoint outputs/.../jax_ckpt_last.pkl --baselines noop
uv run ow train ... artifacts=hybrid_promotion   # async docker + tournament worker
```

### Hybrid promotion poll contract

1. Train with `artifacts=hybrid_promotion`; note `run_dir` from `orbit_train_start`.
2. Poll: `uv run ow eval status --run <run_dir> --watch --poll-seconds 5`.
3. Queue idle when `jobs` has no `queued`/`running` entries; check `promoted_manifest` and worker logs under `queue/`.
4. Cancel mistaken queue entries: `ow eval jobs cancel --run <run_dir> --all-queued --dry-run` first, then without `--dry-run`.
5. Worker processing: `ow eval worker --run <run_dir> --verbose` (or rely on autostart + `queue/worker.stderr.log`).

### Promotion rollback (operator)

```bash
uv run ow promote show --campaign <c>
uv run ow promote history --campaign <c> --limit 10
uv run ow promote demote --campaign <c> --dry-run
uv run ow promote demote --campaign <c>
uv run ow promote demote --campaign <c> --to-previous
```

Clears `promoted/current_best/manifest.json` and campaign `current_best_*` fields; appends an audit row to `indexes/promoted.jsonl`. `--to-previous` restores the prior indexed promotion when its checkpoint still exists.

## Discovery

```bash
uv run ow --help
uv run ow train --help
uv run ow eval --help
uv run ow promote --help
uv run ow benchmark --help
make help
```

## Config vs code (quick boundary)

| Change via Hydra YAML / CLI only | Requires `src/` edit |
|----------------------------------|----------------------|
| Training hyperparams, opponent mix, curriculum stages | New opponent family / heuristic |
| `task=shield_*`, reward weights | New shield mode or feature schema |
| `artifacts=hybrid_promotion`, tournament baselines | PPO / rollout / env mechanics |
| Preflight threshold JSON (after calibrate) | Preflight gate recipe tuples in Python (Phase 2 YAML metadata in `conf/benchmark/gates/`) |

## Copy-paste agent prompts

**Short train smoke after CLI change**

> Run `uv run ow train training=smoke training.total_updates=5 curriculum=off task=shield_off` and confirm `orbit_train_start` / `orbit_train_complete` lines and `logs/*_jax.jsonl` under the run dir.

**Inspect hybrid promotion queue**

> Run `uv run ow eval status --run <run_dir> --watch --poll-seconds 5` and summarize queued/running `checkpoint_eval` / `tournament` jobs; if worker autostarted, read `queue/worker.stderr.log`.

**Cancel stale artifact jobs**

> Run `uv run ow eval jobs cancel --run <run_dir> --all-queued --dry-run`, confirm targets, then rerun without `--dry-run`.

**Rollback mistaken promotion**

> Run `uv run ow promote show --campaign <c>`, then `uv run ow promote demote --campaign <c> --dry-run` and confirm JSON action before applying without `--dry-run`.

**Preflight Gates 2–4**

> Run `uv run ow benchmark gate beat_noop --dry-run` (or `beat_random`) to verify overrides, then `make preflight-learn-proof` only if no other GPU job is active; compare report to `docs/benchmarks/preflight-calibration.json` thresholds.
