# Kaggle Runner

Preferred operator entry: **`uv run ow train kaggle`**. Lower-level module: **`uv run python -m src.cli.kaggle_runner`**.

## Status on main

Remote Kaggle training is **partially wired**: CLI and orchestration live under `src/cli/kaggle_runner.py` and `src/orchestration/kaggle_runner.py`, but the packaged worker entry and default sweep YAML referenced by orchestration are **not present in the tree** (worker entry script and conf/sweeps/wandb/kaggle_runner_mvp.yaml). Treat launch/sync as experimental until those land.

For GPU long runs today, use **[colab_runner.md](colab_runner.md)** (`ow train colab`).

## Shape

Kaggle is the hosted worker backend for remote training:

- The local launcher renders a Kaggle script kernel package under `outputs/kaggle_runner/kernel/`.
- Standalone mode (`ow train kaggle`) skips W&B on Kaggle; Hydra overrides pass through from the CLI.
- Kaggle outputs sync to `outputs/kaggle_runner/synced/` for diagnostics.

## Operator commands

```bash
# Default launch (standalone / no W&B on worker)
uv run ow train kaggle format=mix_2p_4p_16env training.total_updates=500

# Benchmark throughput grid
uv run ow train kaggle --run-type benchmark --accelerator NvidiaTeslaP100 format=mix_2p_4p_16env

# Preflight / prepare
uv run ow train kaggle preflight
uv run ow train kaggle prepare

# Status and sync
uv run ow train kaggle status <kaggle-user>/orbit-wars-kaggle-runner
uv run ow train kaggle sync-output <kaggle-user>/orbit-wars-kaggle-runner --force
```

Module equivalents:

```bash
uv run python -m src.cli.kaggle_runner preflight --no-wandb
uv run python -m src.cli.kaggle_runner launch --no-wandb --run-type smoke
uv run python -m src.cli.kaggle_runner sync-output <user>/<slug> --force
```

Packages and ledgers default to `outputs/kaggle_runner/` (`kernel/`, `launches.jsonl`, `synced/`).

## Competition submissions

To package a checkpoint as `submission.tar.gz` and validate before upload, see
[COMPETITION_SUBMISSION.md](competition/COMPETITION_SUBMISSION.md).

## Related

- Submit-valid tournament ladder: `docs/solutions/architecture-patterns/gate5-unified-tournament-submit-valid-funnel.md`
- Colab long-run host (recommended): [colab_runner.md](colab_runner.md)
