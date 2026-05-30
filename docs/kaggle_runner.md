# Kaggle Runner

Preferred operator entry: **`uv run ow train kaggle`**. The legacy script name
`scripts/kaggle_wandb_population.py` is deprecated (forwards to `kaggle_runner`).

## Shape

Kaggle is the hosted worker backend for remote training:

- The local launcher renders a Kaggle script kernel package.
- Standalone mode (`ow train kaggle`) skips W&B on Kaggle; config comes from sweep
  YAML fixed parameters plus Hydra `--override` tokens.
- The worker verifies GPU-backed JAX before real training.
- Kaggle outputs are diagnostic backup; post-hoc W&B sync is future work.

## Files

- CLI: `uv run ow train kaggle` → `src/cli/train_hosts.py` → `src/cli/kaggle_runner.py`
- Orchestration: `src/orchestration/kaggle_runner.py`
- Script shim: `scripts/kaggle_runner.py`
- Worker: `scripts/kaggle_worker_entry.py`
- Sweep: `conf/sweeps/wandb/kaggle_runner_mvp.yaml`

## Operator commands

```bash
# Default: standalone launch on P100 with full run-type
uv run ow train kaggle format=mix_2p_4p_16env training.total_updates=500

# Benchmark throughput grid
uv run ow train kaggle --run-type benchmark --accelerator NvidiaTeslaP100 format=mix_2p_4p_16env

# Preflight / prepare
uv run ow train kaggle preflight
uv run ow train kaggle prepare

# Status and sync
uv run ow train kaggle status <kaggle-user>/orbit-wars-kaggle-runner
uv run ow train kaggle sync <kaggle-user>/orbit-wars-kaggle-runner
```

Legacy script equivalents:

```bash
uv run python scripts/kaggle_runner.py preflight --no-wandb
uv run python scripts/kaggle_runner.py launch --no-wandb --run-type smoke
uv run python scripts/kaggle_runner.py sync-output <user>/<slug> --force
```

Packages and ledgers default to `outputs/kaggle_runner/` (`kernel/`, `launches.jsonl`, `synced/`).

## Worker behavior

Inside Kaggle, the worker:

1. Loads packaged environment values from `worker-env.json`.
2. Installs `uv` if needed, then runs `uv sync --no-dev`.
3. Verifies JAX sees a GPU unless `ORBIT_WARS_KAGGLE_ALLOW_CPU=1`.
4. In standalone mode, resolves training config from sweep YAML + overrides.
5. Benchmarks a bounded calibration grid when run-type requires it.
6. Runs `uv run python -m src.train` with selected overrides.
7. Writes `worker-summary.json` with diagnostics and exit code.

Useful worker environment overrides:

- `ORBIT_WARS_KAGGLE_RUN_TYPE` — `full`, `smoke`, or `benchmark`
- `ORBIT_WARS_KAGGLE_TRUST_BASE_JAX` — default `1` on Kaggle
- `ORBIT_WARS_KAGGLE_CALIBRATION_*` — calibration grid bounds
- `ORBIT_WARS_KAGGLE_ALLOW_CALIBRATION_FALLBACK`
- `ORBIT_WARS_KAGGLE_ALLOW_CPU`

## Competition submissions

To package a checkpoint as `submission.tar.gz` and validate before upload, see
[kaggle_submission.md](kaggle_submission.md).

## Notes

- `ow train kaggle` always uses standalone mode (`--no-wandb`). W&B sweep creation
  (`--create-sweep`) is not exposed through `ow`; use `scripts/kaggle_runner.py`
  directly if you need W&B-on-Kaggle (legacy path).
- Do not use this as proof of tournament readiness; this is remote training +
  calibration MVP.
