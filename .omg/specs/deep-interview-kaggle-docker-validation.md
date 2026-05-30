# Deep Interview Spec: Kaggle Docker Submission Validation

## Goal

Build a repeatable validation workflow for Orbit Wars Kaggle submissions that packages a trained JAX checkpoint agent and runs it in Kaggle's simulation Docker environment, matching the leaderboard execution environment as closely as practical.

Minimum output: one repeatable local command that takes an explicit JAX checkpoint path, builds or stages a Kaggle-valid submission package, runs it inside `gcr.io/kaggle-images/python-simulations`, and reports clear pass/fail evidence.

Bonus output: a local leaderboard-style harness that can evaluate self-play checkpoints and third-party submission agents across seeded matches.

## Constraints

- Submission format must follow Orbit Wars rules: `main.py` at package root with an `agent(obs)` function.
- Multi-file submissions should be bundled as `submission.tar.gz` with `main.py` at the root and any helper files/model weights included beside it.
- Docker testing should use Kaggle's published simulation image: `gcr.io/kaggle-images/python-simulations`.
- Validation command must accept an explicit checkpoint path, for example `--checkpoint artifacts/<run>/jax_ckpt_last.pkl`.
- The first production path targets the trained JAX checkpoint agent, not only a scripted baseline.
- Failures must be logged clearly enough to distinguish import errors, missing dependencies, action-format/runtime errors, timeout failures, and match execution failures.

## Non-Goals

- Submitting to Kaggle automatically is not required for the first workflow.
- Auto-discovering latest checkpoints is not required initially.
- A full rating-system clone of Kaggle's leaderboard is not required initially.
- Rewriting the model into a portable non-JAX inference runtime is not required unless Docker validation proves current dependencies are untenable.

## Evidence Gathered

- Kaggle Docker README says the simulation image is hosted at `gcr.io/kaggle-images/python-simulations` and runs the `kaggle-environments` CLI.
- Kaggle Orbit Wars overview says submissions must expose a root `main.py` with an `agent` function; multi-file agents can be submitted as a `tar.gz` containing `main.py`, helpers, and model weights.
- Orbit Wars action format is `[[from_planet_id, direction_angle, num_ships], ...]`; returning `[]` is valid.
- Kaggle validation runs an episode against copies of the submitted agent before accepting the submission.
- Kaggle environment configuration includes a 500-step episode limit and 1-second act timeout.
- Kaggle `kaggle-environments` dependencies include JAX and NumPy, making a JAX checkpoint agent plausible in the Docker image.
- Repo exploration found training checkpoints saved as `jax_ckpt_XXXXXX.pkl` and `jax_ckpt_last.pkl`, plus existing replay/evaluation code in `src/replay.py` but no current Kaggle submission packager or Docker validation command.

## Acceptance Criteria

1. A developer can run a single documented command with an explicit checkpoint path to package and validate the trained JAX agent in Kaggle Docker.
2. The generated package is Kaggle-valid: `main.py` exists at the root and exposes `agent(obs)`.
3. The Docker validation runs a self-play episode using copies of the packaged agent, matching Kaggle's initial validation behavior.
4. The command exits non-zero on import, dependency, timeout, invalid action, or episode execution failure.
5. Failure output clearly identifies the failing phase and preserves enough logs for debugging.
6. Passing output reports the package path, Docker image used, checkpoint path, episode status, and final rewards/statuses.
7. The implementation leaves a clear extension point for a local harness that runs multiple seeded matches against random, scripted baseline, checkpoint, and third-party agents.

## Assumptions Exposed And Resolved

- Assumption: the work should validate a real trained JAX checkpoint package, not only a baseline agent. Resolved: target the JAX checkpoint package first.
- Assumption: Docker should be the Kaggle simulation image, not a custom repo image. Resolved: use `gcr.io/kaggle-images/python-simulations`.
- Assumption: runtime dependency viability should be guessed up front. Revised: use Kaggle's actual pyproject and Dockerfile as evidence, then let Docker validation expose concrete failures.
- Assumption: checkpoint selection should be automatic. Resolved: use an explicit checkpoint path argument for the initial command.

## Ontology

- Submission Package: Kaggle-valid root containing `main.py` and optional helper/model files.
- Agent Entrypoint: `agent(obs)` function in `main.py` returning Orbit Wars move lists.
- Checkpoint Artifact: `jax_ckpt_*.pkl` file containing trained JAX policy params and feature metadata.
- Docker Validation: Local execution in `gcr.io/kaggle-images/python-simulations` using `kaggle_environments.make("orbit_wars")`.
- Self-Play Episode: Kaggle-style validation episode where the packaged agent plays against copies of itself.
- Local Harness: Future seeded multi-match runner for comparing checkpoints and third-party agents.

## Interview Transcript

1. Asked for primary output. User clarified: validate chosen agent submission packaging works through a repeatable test command at minimum; local leaderboard harness is a strong bonus.
2. Asked which package to target first. User selected packaged trained JAX checkpoint agent.
3. Asked what counts as passing validation. User clarified: meet runtime and dependency constraints, clearly log failure reasons, and complete self-play validation without errors.
4. Challenged dependency assumptions. User provided Kaggle `kaggle-environments` pyproject and Dockerfile as the source of truth for runtime constraints.
5. Asked how to select the model artifact. User selected explicit checkpoint path argument.

## Ambiguity Score

- Initial: 100%
- After repo and Docker context: 78%
- After output clarification: 58%
- After target package selection: 42%
- After success criteria clarification: 31%
- After dependency evidence correction: 24%
- After checkpoint source selection: 18%

## Recommended Execution Path

Proceed with consensus refinement only if implementation scope needs architecture debate. Otherwise execute directly with an implementation pass focused on a packager script, Docker validation command, and focused tests/docs.
