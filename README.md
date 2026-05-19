# orbit_wars

Orbit Wars now uses **Hydra-first training commands**. The canonical entrypoint is:

```bash
uv run python -m src.train experiment=attention_training
```

For complete experiment operations (sweeps, resumes, logs/checkpoints, evaluation), see [`docs/experiments.md`](docs/experiments.md). For Hydra migration guidance, see [`docs/hydra_migration.md`](docs/hydra_migration.md).

## Hydra basics for this repo

- Base config root: `conf/config.yaml`.
- Experiment presets live under `conf/experiment/*.yaml` and are selected with `experiment=<name>`.
- Compose + override from the CLI:

```bash
uv run python -m src.train experiment=attention_training
uv run python -m src.train experiment=full_training env.player_count=4
uv run python -m src.train experiment=jax_training ppo.total_updates=2000
```

### Adding keys with `+`

Use `+key=value` only when the target key is intentionally absent from the config schema and you need to append it dynamically:

```bash
uv run python -m src.train experiment=attention_training +tags='["ablation","hydra"]'
```

If a key already exists, use normal assignment (`key=value`) instead of `+key=value`.

## Quick run examples

```bash
uv run python -m src.train experiment=full_training
uv run python -m src.train experiment=attention_training
uv run python -m src.train experiment=attention_shaped_reward
uv run python -m src.train experiment=attention_self_play_pool
uv run python -m src.train experiment=jax_training
uv run python -m src.train experiment=jax_self_play_shaped_reward
```

## Resume behavior with checkpoints

Resume with `resume_checkpoint=<path>` (Hydra override form):

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

`ppo.total_updates` is interpreted as the **final target update number**. If you resume from update 50 and set `ppo.total_updates=2000`, training continues at update 51 and stops after update 2000.

## Backend notes (Torch vs JAX)

- Torch path: typically `env_backend=kaggle`, `rl_backend=torch`, checkpoints `ckpt_*.pt` / `ckpt_last.pt`.
- JAX path: `env_backend=jax`, `rl_backend=jax`, checkpoints `jax_ckpt_*.pkl` / `jax_ckpt_last.pkl`.
- Keep backend-specific experiment presets when resuming checkpoints (Torch checkpoint with Torch preset, JAX checkpoint with JAX preset).

## Multirun basics

Hydra multirun (`-m`) launches one job per override combination:

```bash
uv run python -m src.train -m \
  experiment=attention_training \
  env.player_count=2,4 \
  ppo.total_updates=1000,2000
```

Hydra writes multirun job outputs under `multirun/<date>/<time>/<job_id>/` (including `.hydra/` metadata per job), while training artifacts/checkpoints still go to the configured artifact paths.

## Hydra experiment selection (forward-safe)

Use Hydra overrides directly in all scripts and automation:

- `uv run python -m src.train` (defaults from base config)
- `uv run python -m src.train experiment=attention_training`
- `uv run python -m src.train experiment=jax_training resume_checkpoint=/path/to/jax_ckpt_000050.pkl`

