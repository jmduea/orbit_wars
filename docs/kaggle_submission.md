# Kaggle submission packaging

Package a trained JAX checkpoint as `submission.tar.gz` for the Orbit Wars competition, validate it locally in Kaggle's simulation Docker image, then upload the tarball via the competition **Submit** UI.

## Requirements

- Trained checkpoint: `jax_ckpt_last.pkl` or `jax_ckpt_XXXXXX.pkl` under a campaign run directory.
- Docker (for full validation): Docker Desktop with WSL integration, or a Linux host with the `docker` CLI.
- Image: `gcr.io/kaggle-images/python-simulations` (pulled automatically on first run).
- WSL hosts without a CUDA JAX wheel: set `ORBIT_WARS_ALLOW_CPU_JAX_ON_NVIDIA=1` for training and packaging; the artifact worker inherits this when spawned from `ow train`.

## Package and validate

From the repository root:

```bash
uv run python scripts/validate_kaggle_docker_submission.py \
  --checkpoint outputs/campaigns/<campaign>/runs/<run_id>/jax_ckpt_last.pkl \
  --output-dir /tmp/kaggle_submit \
  --player-count both
```

On success:

- `package_path=/tmp/kaggle_submit/submission.tar.gz` is printed.
- `stdout` ends with JSON `"ok": true` and 2p/4p episode results.
- Replays (if enabled): `/tmp/kaggle_submit/replays/replay_u*.html`

Validation runs two import paths on the same tarball:

1. **Kaggle-fidelity** — `exec()` of `main.py` with no `__file__` (matches competition loader).
2. **Episode self-play** — `importlib` load plus seeded 2p/4p games inside Docker.

### Artifact pipeline output

When `artifacts.artifact_pipeline.docker_validation_async` or replay `backend=docker` is enabled, tarballs are written under:

```text
outputs/campaigns/<campaign>/runs/<run_id>/evaluations/<job_id>/docker_validation/submission.tar.gz
```

## Upload to Kaggle

1. Open the Orbit Wars competition **Submit** page.
2. Upload `submission.tar.gz` (not the raw `.pkl` checkpoint).
3. Wait for the validation episode. A passing run should not show `Invalid raw Python` or `NameError: name '__file__' is not defined`.

## Package-only (no Docker)

```bash
uv run python scripts/validate_kaggle_docker_submission.py \
  --checkpoint <path/to/jax_ckpt_last.pkl> \
  --output-dir /tmp/kaggle_submit \
  --skip-docker
```

Use this to inspect tarball layout; it does not prove competition compatibility.

## Failure phases

| Phase | Typical cause |
|-------|----------------|
| `checkpoint_missing` / `checkpoint_schema_failed` | Bad checkpoint path or unsupported architecture |
| `package_layout_failed` | Unsafe tar paths or missing root `main.py` |
| `submission_import_failed` | Import error (including Kaggle `exec` probe) |
| `setup_failed` | JIT warmup did not finish at import time |
| `first_action_failed` / `episode_failed_*` | Runtime or action-format error during play |
| `docker_unavailable` | Docker daemon not reachable (common on WSL without Desktop) |

## Tests

```bash
make test-domain-artifacts
```

Covers packager layout, Kaggle `exec` import probe helpers, and template invariants (no `__file__` in generated `main.py`).
