# Agent capabilities (Orbit Wars)

Task-oriented guide for coding agents. Canonical policy: `AGENTS.md`. Architecture: `docs/ONBOARDING.md`.

## Session context

```bash
make agent-context          # JSON: preflight thresholds, roadmap, recent runs index
uv run ow train print_resolved_config=true
```

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

Thresholds: `docs/benchmarks/preflight-calibration.json` (never invent gate numbers).

## Eval & promotion

```bash
uv run ow eval status --run outputs/campaigns/<c>/runs/<id>
uv run ow eval worker --run outputs/campaigns/<c>/runs/<id> --verbose
uv run ow eval tournament --checkpoint outputs/.../jax_ckpt_last.pkl --baselines noop
uv run ow train ... artifacts=hybrid_promotion   # async docker + tournament worker
```

## Discovery

```bash
uv run ow --help
uv run ow train --help
uv run ow eval --help
uv run ow benchmark --help
make help
```

## Config vs code (quick boundary)

| Change via Hydra YAML / CLI only | Requires `src/` edit |
|----------------------------------|----------------------|
| Training hyperparams, opponent mix, curriculum stages | New opponent family / heuristic |
| `task=shield_*`, reward weights | New shield mode or feature schema |
| `artifacts=hybrid_promotion`, tournament baselines | PPO / rollout / env mechanics |
| Preflight threshold JSON (after calibrate) | Preflight gate recipe tuples in Python (Phase 2: YAML) |

## Copy-paste agent prompts

**Short train smoke after CLI change**

> Run `uv run ow train training=smoke training.total_updates=5 curriculum=off task=shield_off` and confirm `orbit_train_start` / `orbit_train_complete` lines and `logs/*_jax.jsonl` under the run dir.

**Inspect hybrid promotion queue**

> Run `uv run ow eval status --run <run_dir>` and summarize queued/running `checkpoint_eval` / `tournament` jobs; if worker autostarted, read `queue/worker.stderr.log`.

**Preflight Gates 2–4**

> Run `make preflight-learn-proof` only if no other GPU job is active; compare report to `docs/benchmarks/preflight-calibration.json` thresholds.
