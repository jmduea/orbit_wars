setup:
	uv sync --group dev

test:
	uv run --group dev pytest

# CPU-only daily loop: no JAX imports, serial execution (safe on WSL2 + NVIDIA).
test-fast:
	uv run --group dev pytest -m "not slow and not jax"

# Lightweight JAX checks (metric math, action builders) — no rollout/training smokes.
test-jax:
	uv run --group dev pytest -m "jax and not slow"

test-full: test

test-domain-config:
	uv run --group dev pytest tests/test_config_consolidation.py tests/test_telemetry.py tests/test_metric_registry.py tests/test_run_paths.py -m "not slow and not jax"

test-domain-features:
	uv run --group dev pytest tests/test_feature_registry.py tests/test_feature_encoding_golden.py tests/test_intercept_edge_features.py -m "not slow and not jax"

test-domain-policy:
	uv run --group dev pytest tests/test_jax_policy_encoder.py tests/test_jax_ppo.py tests/test_ppo_update.py tests/test_trajectory_shield.py -m "jax and not slow"

test-domain-artifacts:
	uv run --group dev pytest tests/test_artifact_pipeline.py tests/test_replay.py tests/test_kaggle_submission_packager.py tests/test_checkpoint_compat.py -m "not slow and not jax"

test-domain-curriculum:
	uv run --group dev pytest tests/test_curriculum.py tests/test_jax_train_timing.py -m "not slow and not jax"
