# Running Orbit Wars experiments with Hydra

This guide is the canonical reference for launching, sweeping, and resuming Orbit Wars training with Hydra.

## 1) Hydra basics for this repo

Training entrypoint:

```bash
uv run python -m src.train experiment=attention_training
```

How config composition works:

- `conf/config.yaml` is the root config.
- `experiment=<name>` selects a preset from `conf/experiment/<name>.yaml`.
- Additional CLI overrides patch specific keys.

Examples:

```bash
uv run python -m src.train experiment=attention_training
uv run python -m src.train experiment=attention_training env.player_count=4
uv run python -m src.train experiment=attention_training ppo.total_updates=5000
```

### Override patterns

- Existing key override: `key=value`
- Nested override: `section.key=value`
- Optional append-only override for missing keys: `+new_key=value`

```bash
uv run python -m src.train experiment=attention_training +notes=hydra_test
```

If Hydra reports that a key is not in the struct/schema, either:
1. Use the correct existing key name, or
2. Intentionally add with `+...` when dynamic keys are supported for your workflow.

## 2) Multirun usage and output layout

Use `-m` to sweep one or more values:

```bash
uv run python -m src.train -m \
  experiment=attention_training \
  env.player_count=2,4 \
  ppo.total_updates=1000,2000
```

This runs one job per Cartesian-product combination.

Hydra outputs:

- Single run: `outputs/<YYYY-MM-DD>/<HH-MM-SS>/`
- Multirun: `multirun/<YYYY-MM-DD>/<HH-MM-SS>/<job_id>/`
- Each job includes `.hydra/` metadata (`config.yaml`, `hydra.yaml`, `overrides.yaml`).

Training artifacts/checkpoints still follow the project artifact paths (for example `/artifacts/...`) based on run configuration.

## 3) Resume behavior with checkpoints

Use `resume_checkpoint=<path>` in Hydra form:

```bash
uv run python -m src.train \
  experiment=attention_training \
  resume_checkpoint=/artifacts/attention_training/orbit_wars_ppo_attention_training/ckpt_000050.pt
```

```bash
uv run python -m src.train \
  experiment=jax_training \
  resume_checkpoint=/artifacts/jax_training/orbit_wars_ppo_jax_training/jax_ckpt_000050.pkl
```

Behavior details:

- `ppo.total_updates` is the absolute final update target.
- Resuming from update `N` continues at `N+1`.
- Keep architecture- and backend-compatible configs when resuming.

## 4) Backend notes (JAX-only)

- Setup: `env_backend=jax`, `rl_backend=jax`
- Checkpoints: `jax_ckpt_last.pkl`, `jax_ckpt_*.pkl`
- Use JAX-compatible experiment presets (for example `jax_training`, `jax_mixed_2p_4p_training`, `jax_entity_transformer_*`, `attention_training`, `full_training`).
- Do not change shape-defining settings between save and resume/eval unless intentionally starting a fresh run.

## 5) Migration: old `--config` commands → Hydra commands

For the complete migration matrix, timeline, and troubleshooting, see [`docs/hydra_migration.md`](hydra_migration.md).

Quick examples:

```bash
uv run python -m src.train experiment=attention_training
```

```bash
uv run python -m src.train experiment=jax_training resume_checkpoint=/path/to/jax_ckpt_000050.pkl
```

## Canonical experiment authoring policy

- Canonical experiment editing and sweeps happen only in `conf/` (`conf/config.yaml`, `conf/experiment/*.yaml`, and config groups).
- `configs/` has been removed; use Hydra experiment selection from `conf/experiment/` for all authoring and execution.
