# Ralplan: Kaggle Docker Submission Validation

## Decision

Implement JAX-in-Docker submission validation first. Package a trained JAX checkpoint agent as a Kaggle-valid `submission.tar.gz`, validate that exact tarball inside `gcr.io/kaggle-images/python-simulations`, and reserve a portable NumPy runtime only for concrete Docker evidence that JAX/Flax is untenable.

## Decision Drivers

- Validate the same artifact structure Kaggle accepts: root `main.py` with `agent(obs)`.
- Preserve trained-policy behavior by using the existing JAX/Flax model path before reimplementing inference.
- Avoid fragile training/runtime coupling by stripping checkpoints to inference-only artifacts.
- Catch leaderboard-like failures locally: dependencies, import, artifact load, first action latency, action format, and 2p/4p episode execution.

## Guardrails

- The packaged submission must ship a stripped inference artifact, not the raw `jax_ckpt_*.pkl` training checkpoint.
- The stripped artifact must contain only inference params, plain-data config/model fields, feature metadata, and version/architecture metadata.
- The runtime package must avoid `src.config`, Hydra, WandB, Optax, training state, and replay helpers.
- The Docker validator must untar and import the exact generated package in isolation, with no repo path available for imports.
- The Docker validator must measure cold import time and first `agent(obs)` latency, then report max/first action latency.
- Pointer-policy decoding must handle all `model.max_moves_k` steps and must not copy the single-step replay helper behavior.
- Validation must run explicit 2-player and 4-player seeded self-play episodes unless the user chooses a narrower player-count option later.

## Implementation Steps

1. Add a Kaggle Docker dependency probe for `gcr.io/kaggle-images/python-simulations`.
   - Probe `jax`, `flax`, `numpy`, and `kaggle_environments` imports.
   - Print versions, import timing, available devices, and whether `make("orbit_wars")` succeeds.
   - Classify failures as `docker_unavailable`, `image_pull_failed`, `dependency_probe_failed`, or `environment_probe_failed`.

2. Add a checkpoint export phase.
   - Accept an explicit `--checkpoint <path>`.
   - Load the trusted local training checkpoint with repo dependencies available.
   - Write an inference-only artifact containing params, model/env config scalars, feature metadata, checkpoint update, and export format version.
   - Reject unsupported architectures or missing metadata with clear errors.

3. Add a Kaggle submission package builder.
   - Emit `submission.tar.gz` with `main.py` at root.
   - Include the stripped inference artifact and minimal runtime-safe modules/templates.
   - Inspect the tarball for package layout and parent-path traversal issues before Docker validation.

4. Implement runtime-safe JAX inference and pointer decoding.
   - Load the stripped artifact.
   - Build the JAX/Flax policy from plain config fields.
   - Encode observations without importing training entrypoints or Hydra config composition.
   - Decode pointer outputs across `(source, max_moves_k)` steps: select target, select ship bucket for that step and target, skip no-op/invalid/zero bucket, allocate ships from remaining source ships, and emit `[from_planet_id, direction_angle, num_ships]` moves.

5. Validate the exact tarball in Docker.
   - Mount only the generated tarball and a scratch/output directory.
   - Extract inside the container, import root `main.py`, verify `agent(obs)` exists, load the stripped artifact, and call `agent(obs)` on a real Orbit Wars observation.
   - Run seeded 2-player and 4-player self-play episodes using copies of the packaged agent.
   - Exit non-zero with clear failure phases: `package_layout_failed`, `submission_import_failed`, `artifact_load_failed`, `first_action_failed`, `timeout_failed`, `invalid_action_failed`, `episode_failed_2p`, `episode_failed_4p`.

6. Add focused tests and docs.
   - Test package layout and path safety.
   - Test stripped artifact schema and unsupported-checkpoint errors.
   - Test runtime import boundaries so training-only modules are not pulled into submission runtime.
   - Test pointer `max_moves_k` decoding with mocked outputs, including invalid/no-op/zero-bucket and later-step-only cases.
   - Document the command, expected outputs, Docker requirement, and failure categories.

## Acceptance Criteria

- A documented command accepts an explicit checkpoint and produces the exact `submission.tar.gz` it validates.
- Validation uses `gcr.io/kaggle-images/python-simulations` and reports measured dependency viability.
- The tarball has root `main.py` exposing `agent(obs)`.
- The package loads a stripped inference artifact, never the raw training checkpoint.
- The runtime avoids training-only imports unless a measured and documented exception is explicitly accepted later.
- Pointer policies correctly decode all `model.max_moves_k` steps into legal Orbit Wars action lists.
- Docker validation passes seeded 2-player and 4-player self-play with copies of the exact packaged submission.
- Passing output reports package path, Docker image, checkpoint path, dependency probe, cold import time, first/max action latency, and final episode rewards/statuses.
- Failing output exits non-zero and names the failing phase clearly.

## Fallback Rule

Do not start a portable NumPy runtime unless Docker validation shows the JAX/Flax path fails concrete constraints such as missing dependencies, incompatible runtime behavior, package size, cold-start timeout, or repeated act-time budget failures.

## Verification Commands

```bash
rtk uv run --group dev pytest tests/test_kaggle_submission_packager.py tests/test_kaggle_submission_runtime.py
rtk uv run python scripts/validate_kaggle_docker_submission.py --checkpoint artifacts/<run>/jax_ckpt_last.pkl --player-count both
```
