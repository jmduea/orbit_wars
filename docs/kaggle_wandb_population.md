# Kaggle W&B Population MVP

This runbook covers the first W&B-native Kaggle population workflow.

## Shape

W&B is the population control plane:

- W&B sweep config assigns candidate parameters.
- W&B runs hold candidate metrics and calibration metadata.
- W&B checkpoint artifacts are the canonical handoff surface.
- The W&B API key is read inside Kaggle from a Kaggle Secret named
  `WANDB_API_KEY` by default. The launcher must not package a raw API key in
  `worker-env.json`.

Kaggle is the hosted worker backend:

- The local launcher renders a Kaggle script kernel package.
- The launcher requests accelerators in ordered fallback.
- The worker verifies GPU-backed JAX before real training.
- Kaggle outputs are diagnostic backup, not the source of truth.

## Files

- Launcher: `scripts/kaggle_wandb_population.py`
- Worker: `scripts/kaggle_worker_entry.py`
- Sweep: `conf/sweeps/wandb/kaggle_population_mvp.yaml`
- Orchestration package: `src/orchestration/`

## Prepare A Worker Package

Run preflight before rendering or launching:

```bash
uv run python scripts/kaggle_wandb_population.py preflight \
  --kernel-id <kaggle-user>/orbit-wars-wandb-population \
  --project orbit_wars \
  --entity <wandb-entity>
```

Preflight checks Kaggle CLI/auth, W&B import/API access, sweep YAML parsing,
placeholder kernel IDs, and package directory writability. It does not launch a
kernel or train.

```bash
uv run python scripts/kaggle_wandb_population.py prepare \
  --kernel-id <kaggle-user>/orbit-wars-wandb-population \
  --sweep-id <wandb-sweep-id>
```

The package is written under `outputs/kaggle_population/kernel/` by default. It
includes `src/`, `conf/`, the Kaggle worker, `scripts/benchmark_jax_rl.py`,
`pyproject.toml`, and a `package-summary.json` with top-level package contents
and non-secret generated environment values. GPU packages rewrite the local
CUDA 13 JAX dependency to plain `jax`; the worker bootstrap then installs the
version-aligned CUDA 12 stack expected by Kaggle GPU images. Before launching,
add a Kaggle Secret named `WANDB_API_KEY` to the worker notebook or set
`ORBIT_WARS_KAGGLE_WANDB_SECRET_NAME` locally to use a different secret name.

## Dry-Run Launch

```bash
uv run python scripts/kaggle_wandb_population.py launch \
  --dry-run \
  --kernel-id <kaggle-user>/orbit-wars-wandb-population \
  --sweep-id <wandb-sweep-id> \
  --accelerator NvidiaTeslaT4
```

This prints the Kaggle CLI command without contacting Kaggle.

## Create A W&B Sweep And Launch

```bash
uv run python scripts/kaggle_wandb_population.py launch \
  --create-sweep \
  --project orbit_wars \
  --entity <wandb-entity> \
  --kernel-id <kaggle-user>/orbit-wars-wandb-population
```

The launcher uses `conf/sweeps/wandb/kaggle_population_mvp.yaml`, adds
population tags, renders the worker package, and pushes it with the first
accepted accelerator.

Launch attempts are appended to `outputs/kaggle_population/launches.jsonl`.
The ledger is diagnostic-only: W&B sweep assignment remains the source of truth.

## Status And Output Sync

```bash
uv run python scripts/kaggle_wandb_population.py status <kaggle-user>/<kernel-slug>

uv run python scripts/kaggle_wandb_population.py sync-output \
  <kaggle-user>/<kernel-slug> \
  --output-dir outputs/kaggle_population/synced \
  --force
```

`status` prints normalized state plus the raw Kaggle response. `sync-output`
records the downloaded path and command result in the launch ledger.

## Shortlist

```bash
uv run python scripts/kaggle_wandb_population.py shortlist \
  --project orbit_wars \
  --entity <wandb-entity> \
  --sweep-id <wandb-sweep-id> \
  --output outputs/kaggle_population/shortlist.json
```

The shortlist ranks finished runs with checkpoint artifacts ahead of partial or
diagnostic runs, then considers `episode_reward_mean`, `samples_per_sec`, and
`ppo_samples_per_sec`.

For the latest checkpoint candidate for a sweep:

```bash
uv run python scripts/kaggle_wandb_population.py latest-checkpoint \
  --project orbit_wars \
  --entity <wandb-entity> \
  --sweep-id <wandb-sweep-id>
```

Pass `--run-id <wandb-run-id-or-name>` to narrow the query to one candidate.
Checkpoint output includes artifact name, version, and aliases when W&B exposes
them.

## Tiny Live Smoke

Use the smallest manual validation sequence:

```bash
uv run python scripts/kaggle_wandb_population.py preflight \
  --kernel-id <kaggle-user>/orbit-wars-wandb-population \
  --project orbit_wars \
  --entity <wandb-entity>

uv run python scripts/kaggle_wandb_population.py launch \
  --dry-run \
  --kernel-id <kaggle-user>/orbit-wars-wandb-population \
  --sweep-id <wandb-sweep-id> \
  --accelerator NvidiaTeslaT4

uv run python scripts/kaggle_wandb_population.py launch \
  --create-sweep \
  --project orbit_wars \
  --entity <wandb-entity> \
  --kernel-id <kaggle-user>/orbit-wars-wandb-population \
  --accelerator NvidiaTeslaT4

uv run python scripts/kaggle_wandb_population.py status <kaggle-user>/<kernel-slug>
uv run python scripts/kaggle_wandb_population.py sync-output <kaggle-user>/<kernel-slug> --force
uv run python scripts/kaggle_wandb_population.py shortlist --project orbit_wars --entity <wandb-entity> --sweep-id <wandb-sweep-id>
```

Stop conditions:

- JAX reports no GPU backend.
- `uv sync` fails in Kaggle.
- All calibration variants fail and `ORBIT_WARS_KAGGLE_ALLOW_CALIBRATION_FALLBACK=1` is not set.
- No W&B sweep run appears.
- No checkpoint artifact appears with the `latest` alias.

## Worker Behavior

Inside Kaggle, the worker:

1. Loads packaged environment values from `worker-env.json`.
2. Installs `uv` if needed, then runs `uv sync`.
3. Verifies JAX sees a GPU unless `ORBIT_WARS_KAGGLE_ALLOW_CPU=1`.
4. Runs one W&B sweep assignment.
5. Estimates throughput settings from observed GPU memory and candidate shape.
6. Benchmarks a bounded calibration grid using `scripts/benchmark_jax_rl.py`.
7. Runs `uv run python -m src.train` with the selected calibration overrides.
8. Logs checkpoint artifacts through W&B with `latest` and `update-<N>` aliases.

Useful worker environment overrides:

- `ORBIT_WARS_KAGGLE_CALIBRATION_WARMUP`
- `ORBIT_WARS_KAGGLE_CALIBRATION_UPDATES`
- `ORBIT_WARS_KAGGLE_CALIBRATION_MAX_VARIANTS`
- `ORBIT_WARS_KAGGLE_CALIBRATION_TIMEOUT_SECONDS`
- `ORBIT_WARS_KAGGLE_ALLOW_CALIBRATION_FALLBACK`
- `ORBIT_WARS_KAGGLE_ALLOW_CPU`

Real training keeps CPU fallback disabled unless `ORBIT_WARS_KAGGLE_ALLOW_CPU=1`
is explicitly set. `worker-summary.json` is always written and includes
diagnostics, calibration settings/results, selected overrides, final command,
and exit code.

## Notes

- Do not use this as proof of tournament readiness. This is the Population MVP:
  launch workers, calibrate, train candidates, upload checkpoints, shortlist.
- Scripted-nearest gates, tournaments, action-distribution diversity, and
  self-play pool mutation are follow-up work.
