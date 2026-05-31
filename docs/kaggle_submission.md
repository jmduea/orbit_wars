# Kaggle submission packaging

Package a trained JAX checkpoint as `submission.tar.gz` for the Orbit Wars competition, validate it locally in Kaggle's simulation Docker image, then upload via `ow eval submit` or the Kaggle CLI.

## Requirements

- Trained checkpoint: `jax_ckpt_last.pkl` or `jax_ckpt_XXXXXX.pkl` under a campaign run directory.
- Docker (required for **local validation**): Docker Desktop with WSL integration, or a Linux host with the `docker` CLI.
- Image: `gcr.io/kaggle-images/python-simulations` (pulled automatically on first run).
- WSL hosts without a CUDA JAX wheel: set `ORBIT_WARS_ALLOW_CPU_JAX_ON_NVIDIA=1` for training and packaging; the artifact worker inherits this when spawned from `ow train`.

There is no local no-Docker submission validation path. Packaging alone only checks tarball layout (required files, safe paths). Import probes, Kaggle `exec()` fidelity, and episode self-play run inside Docker.

## Package and validate

From the repository root:

```bash
uv run ow eval package \
  --checkpoint outputs/campaigns/<campaign>/runs/<run_id>/checkpoints/jax_ckpt_last.pkl \
  --output-dir /tmp/kaggle_submit \
  --validate-docker
```

On success:

- `package_path=/tmp/kaggle_submit/submission.tar.gz` is printed.
- With `--validate-docker`, `stdout` ends with JSON `"ok": true` and 2p/4p episode results.
- Replays: `/tmp/kaggle_submit/replays/replay_u*.html`

Validation inside Docker runs two import paths on the same tarball:

1. **Kaggle-fidelity** — `exec()` of `main.py` with no `__file__` (matches competition loader).
2. **Episode self-play** — `importlib` load plus seeded 2p/4p games.

### Packaging only (no validation)

To build `submission.tar.gz` without Docker (layout checks only — **not** competition compatibility):

```bash
uv run ow eval package \
  --checkpoint outputs/campaigns/<campaign>/runs/<run_id>/checkpoints/jax_ckpt_last.pkl \
  --output-dir /tmp/kaggle_submit
```

Use this to inspect tarball contents before running Docker validation or uploading.

### Artifact pipeline output

When `artifacts.artifact_pipeline.docker_validation_async` or replay `backend=docker` is enabled, tarballs are written under:

```text
outputs/campaigns/<campaign>/runs/<run_id>/evaluations/<job_id>/docker_validation/submission.tar.gz
```

### Hybrid promotion (`artifacts=hybrid_promotion`)

Strict promotion runs asynchronously during training when scalar metrics improve the campaign best:

1. Training queues `checkpoint_eval` (not separate docker + tournament jobs).
2. Artifact worker validates the submission tarball in Kaggle Docker, then runs tournament gates.
3. Promoted manifest updates only when Docker passes **and** tournament gates pass.

Layout per eligible checkpoint:

```text
evaluations/checkpoint_eval_u<update>_<job_id>/
  manifest.json              # validation_ok, tournament_id, promoted
  docker_manifest.json
  docker_validation/
    submission.tar.gz
    replays/replay_u*.html
  tournament/
    progress.json
    leaderboard.json
    matches/*.json
```

Manual re-run of a queued job:

```bash
uv run ow eval worker --run outputs/campaigns/<campaign>/runs/<run_id>
```

Add `--retry-failed` to pick up failed jobs, or `--watch` to poll until idle. Low-level script equivalent: `scripts/run_artifact_worker.py`.

CLI tournament eval (no Docker gate): `uv run ow eval tournament --checkpoint ... --campaign ...`

See `docs/architecture/tournament-eval.md` for formats, baselines, and gate thresholds.

## Upload to Kaggle

Requires the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) on `PATH` with credentials configured (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`).

Package and submit from the repository root:

```bash
uv run ow eval submit \
  --checkpoint outputs/campaigns/<campaign>/runs/<run_id>/checkpoints/jax_ckpt_last.pkl \
  -m "update description"
```

Add `--validate-docker` to run local Kaggle Docker validation before upload. Use `--dry-run` to print the `kaggle competitions submit` command without uploading. To upload an existing tarball: `--package /path/to/submission.tar.gz`.

Default competition slug: `orbit-wars` (override with `--competition`).

After upload, wait for the validation episode on Kaggle. A passing run should not show `Invalid raw Python` or `NameError: name '__file__' is not defined`.

## Failure phases

Docker validation only (`ow eval package --validate-docker`, artifact worker, or `ow eval submit --validate-docker`):

| Phase | Typical cause |
|-------|----------------|
| `checkpoint_missing` / `checkpoint_schema_failed` | Bad checkpoint path or unsupported architecture |
| `package_layout_failed` | Unsafe tar paths or missing root `main.py` |
| `submission_import_failed` | Import error (including Kaggle `exec` probe) |
| `setup_failed` | JIT warmup did not finish at import time |
| `first_action_failed` / `episode_failed_*` | Runtime or action-format error during play |
| `docker_unavailable` | Docker daemon not reachable (common on WSL without Desktop) |

Packaging without `--validate-docker` can fail on checkpoint export or tarball layout only.

## Tests

```bash
make test-domain-artifacts
```

Covers packager layout, Kaggle `exec` import probe helpers, and template invariants (no `__file__` in generated `main.py`).
