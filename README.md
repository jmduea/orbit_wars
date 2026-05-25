# Orbit Wars

Orbit Wars is a Python 3.12 reinforcement-learning project managed with `uv` and launched through Hydra.

The canonical training entrypoint composes responsibility-based config groups from `conf/`:

```bash
uv run python -m src.train
```

Print the resolved config without training:

```bash
uv run python -m src.train print_resolved_config=true
```

## Config Groups

Configuration is organized by responsibility:

| Group | Responsibility |
| --- | --- |
| `model` | Policy architecture and capacity. |
| `task` | Environment shape, player count, candidate count, feature history, and trajectory shield shape. |
| `reward` | Reward shaping and terminal reward behavior. |
| `training` | PPO budget, optimizer, rollout, batching, and reseeding controls. |
| `format` | Player-count mix and rollout group topology. |
| `opponents` | Opponent source policy, self-play enablement, static mixture, and snapshot pool mechanics. |
| `curriculum` | Stage progression and stage-local opponent-family weights. |
| `telemetry` | Metric groups and W&B logging metadata. |
| `artifacts` | Checkpoints, retention, replay generation, and artifact pipeline behavior. |
| `output` | Campaign/run layout, manifests, retention class, and local cache paths. |

Examples:

```bash
uv run python -m src.train model=attention training.total_updates=1000
uv run python -m src.train task.candidate_count=16 reward.reward_production_delta=0.01
uv run python -m src.train format=mix_2p_4p_16env opponents=self_play_curriculum
```

Overrides outside these responsibility groups are rejected by Hydra instead of being normalized at runtime.

## Sweeps

Hydra multirun varies group choices or coherent field axes:

```bash
uv run python -m src.train -m \
  model=attention,entity_transformer_700k \
  training.total_updates=250,500 \
  task.candidate_count=8,16
```

W&B sweep templates live in `conf/sweeps/wandb/` and are split by campaign intent: capacity, budget, reward, task complexity, curriculum, and throughput.

## Outputs

New training runs use a campaign-oriented output layout:

```text
outputs/campaigns/<campaign>/runs/<run_id>/
```

Set `output.campaign=<slug>` to group related runs by experimental question. Each run envelope keeps Hydra's `.hydra/` snapshot, `manifest.json`, logs, checkpoints, queue state, and evaluation artifacts together. W&B generated files are routed into the run envelope, while W&B artifact/data caches live under `outputs/cache/`.

Existing top-level `outputs/YYYY-MM-DD/`, `wandb/`, and `artifacts/` data may still exist from legacy runs, but they are not the canonical layout for new runs.

## Resume

Use `resume_checkpoint=<path>` while keeping architecture- and shape-defining config compatible with the checkpoint:

```bash
uv run python -m src.train resume_checkpoint=/path/to/jax_ckpt_last.pkl
```

## Development

Install dependencies:

```bash
uv sync --group dev
```

Run tests:

```bash
make test-fast    # CPU-only daily loop (safe on WSL2)
make test-jax     # serial JAX subset when editing JAX code
make test         # full suite incl. slow tests; serial only — never use pytest -n
```

Do **not** use `pytest-xdist` / `-n auto`: parallel JAX/CUDA workers have crashed WSL2.

For config work, start with:

```bash
make test-domain-config
```
