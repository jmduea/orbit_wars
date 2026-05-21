# orbit_wars

Orbit Wars now uses **Hydra-first training commands**. The canonical entrypoint is:

```bash
uv run python -m src.train experiment=attention_training
```

For complete experiment operations (sweeps, resumes, logs/checkpoints, evaluation), see [`docs/experiments.md`](docs/experiments.md). For Hydra migration guidance, see [`docs/hydra_migration.md`](docs/hydra_migration.md).
For a goal-oriented knob map (capacity vs budget vs curriculum vs environment difficulty) plus sweep templates, see the **Experiment Tuning Playbook** section in [`docs/experiments.md#7-experiment-tuning-playbook-what-to-change-for-each-goal`](docs/experiments.md#7-experiment-tuning-playbook-what-to-change-for-each-goal).

## Environment setup (Codex/CI-safe)

Before running tests or scripts, sync the declared dependencies (including dev tooling):

```bash
uv sync --group dev
```

Then run tests through uv so the same locked environment is used everywhere:

```bash
uv run --group dev pytest
```

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
   resume_checkpoint=/artifacts/attention_training/orbit_wars_ppo_attention_training/jax_ckpt_000050.pkl
```

`ppo.total_updates` is interpreted as the **final target update number**. If you resume from update 50 and set `ppo.total_updates=2000`, training continues at update 51 and stops after update 2000.

## Backend Notes

- Training uses the JAX environment, JAX policy, and JAX PPO implementation.
- Checkpoints are `jax_ckpt_*.pkl` / `jax_ckpt_last.pkl`.
- Checkpoint serialization runs through the async artifact pipeline by default; final and latest checkpoints are flushed before a successful training exit.
- Keep architecture- and shape-compatible presets when resuming checkpoints.

## Async artifact jobs

Training writes Docker-backed replay/evaluation jobs under each run's `artifact_jobs/` directory instead of rendering them inline. With the default `artifact_pipeline.replay_backend=docker`, the worker packages the checkpoint as a Kaggle submission and runs evaluation inside `gcr.io/kaggle-images/python-simulations`. Process queued jobs with:

```bash
uv run python scripts/run_artifact_worker.py artifacts/<run>/artifact_jobs --once
```

Docker-rendered replay HTML files are written under the completed job output directory, for example `artifact_jobs/docker_u000100_<job_id>/replays/replay_u000100_2p.html`, and the completed job JSON records them in `replay_html_paths`.

If a worker exits while a job is marked `running`, recover it explicitly:

```bash
uv run python scripts/run_artifact_worker.py artifacts/<run>/artifact_jobs --once --recover-running
```

Set `artifact_pipeline.replay_backend=local` for local HTML replay rendering, `artifact_pipeline.replay_async=false` to restore inline local replay generation, or `artifact_pipeline.enabled=false` to use synchronous checkpoint writes. Additional Docker validation jobs are off by default; enable them with `artifact_pipeline.docker_validation_async=true`.

## Kaggle submission validation

Validate a trained checkpoint against Kaggle's simulation Docker image before uploading:

```bash
uv run python scripts/validate_kaggle_docker_submission.py \
   --checkpoint artifacts/<run>/jax_ckpt_last.pkl \
   --player-count both
```

The command builds a Kaggle-style `submission.tar.gz` with root `main.py`, exports a stripped inference artifact instead of shipping the raw training checkpoint, and runs the exact tarball inside `gcr.io/kaggle-images/python-simulations`. Passing output reports dependency versions, package path, cold import time, first-action latency, and seeded 2-player/4-player self-play results.

Use `--skip-docker` to build and inspect the package without launching Docker. Failures exit non-zero and identify the phase, such as `dependency_probe_failed`, `package_layout_failed`, `submission_import_failed`, `artifact_load_failed`, `first_action_failed`, `timeout_failed`, `invalid_action_failed`, `episode_failed_2p`, or `episode_failed_4p`.

For a quicker container smoke test, keep the same packaging path but bound the local self-play length:

```bash
uv run python scripts/validate_kaggle_docker_submission.py \
   --checkpoint artifacts/<run>/jax_ckpt_last.pkl \
   --player-count 2 \
   --episode-steps 20
```

The default remains `--episode-steps 500`, matching the competition configuration.

## Multirun basics

Hydra multirun (`-m`) launches one job per override combination:

```bash
uv run python -m src.train -m \
  experiment=attention_training \
  env.player_count=2,4 \
  ppo.total_updates=1000,2000
```

Hydra writes multirun job outputs under `multirun/<date>/<time>/<job_id>/` (including `.hydra/` metadata per job), while training artifacts/checkpoints still go to the configured artifact paths.

## Memory-first tuning order (reduce OOM risk)

When you hit GPU/TPU OOM, tune these in order so you reduce activation memory before changing task difficulty:

1. **Enable gradient checkpointing first**  
   `ppo.enable_gradient_checkpointing=true` trades compute for lower peak memory during policy forward/backward.
2. **Lower rollout microbatch envs (if using rollout collectors that honor it)**  
   `ppo.rollout_microbatch_envs=<N>` keeps per-step rollout memory bounded by splitting env batches.
3. **Lower PPO update chunk rows**  
   Set `ppo.update_chunk_rows_min` smaller (for example `4096` or `2048`) so update-time policy apply uses smaller chunks.
4. **Cap PPO update chunk rows**  
   Set `ppo.update_chunk_rows_max` (for example `8192`) to prevent oversized chunks when `ppo.minibatch_size` is large.
5. **Only then reduce global workload**  
   Decrease `ppo.num_envs`, `ppo.rollout_steps`, or model size.

Safety checks enforced at startup:

- `ppo.update_chunk_rows_min > 0`
- `ppo.update_chunk_rows_max > 0` when set
- `ppo.update_chunk_rows_max >= ppo.update_chunk_rows_min` when both set
- `ppo.rollout_microbatch_envs > 0` when set

## Hydra experiment selection (forward-safe)

Use Hydra overrides directly in all scripts and automation:

- `uv run python -m src.train` (defaults from base config)
- `uv run python -m src.train experiment=attention_training`
- `uv run python -m src.train experiment=jax_training resume_checkpoint=/path/to/jax_ckpt_000050.pkl`

## Canonical experiment authoring policy

- Canonical experiment editing and sweeps happen only in `conf/` (`conf/config.yaml`, `conf/experiment/*.yaml`, and config groups).
- `configs/` has been removed; use Hydra experiment selection from `conf/experiment/` for all authoring and execution.
