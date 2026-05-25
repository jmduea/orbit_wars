setup:
	uv sync --group dev

test:
	uv run --group dev pytest

# CPU-only daily loop: no JAX imports, serial execution (safe on WSL2 + NVIDIA).
test-fast:
	uv run --group dev pytest -m "not slow and not jax"

# JAX tests outside the slow tier; always serial to avoid multi-process CUDA init.
test-jax:
	uv run --group dev pytest -m "jax and not slow"

test-full: test

test-domain-config:
	uv run --group dev pytest tests/test_config_consolidation.py tests/test_telemetry.py tests/test_metric_registry.py tests/test_run_paths.py -m "not slow and not jax"

test-domain-features:
	uv run --group dev pytest tests/test_features.py tests/test_feature_history.py tests/test_feature_registry.py tests/test_normalization.py -m "not slow and not jax"

test-domain-jax-env:
	uv run --group dev pytest tests/test_jax_env.py -m "jax and not slow"

test-domain-policy:
	uv run --group dev pytest tests/test_jax_policy.py tests/test_jax_ppo.py tests/test_trajectory_shield.py -m "jax and not slow"

test-domain-artifacts:
	uv run --group dev pytest tests/test_artifact_pipeline.py tests/test_replay.py tests/test_kaggle_submission_packager.py -m "not slow and not jax"

test-domain-curriculum:
	uv run --group dev pytest tests/test_curriculum.py tests/test_jax_train_timing.py -m "not slow and not jax"
