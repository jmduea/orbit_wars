# Shared pytest environment: CPU JAX + persistent XLA compile cache (see tests/conftest.py).
export JAX_COMPILATION_CACHE_DIR ?= $(HOME)/.cache/orbit-wars-jax-compile
export ORBIT_WARS_PYTEST_JAX_CACHE ?= 1

PYTEST := uv run --group dev pytest
PYTEST_CPU := JAX_PLATFORMS=cpu $(PYTEST)
# CPU-only parallel workers (explicit targets only; never on GPU pytest or slow tier).
PYTEST_XDIST ?= -n 4 --dist loadscope
XDIST_ENV := ORBIT_WARS_PYTEST_XDIST=1 JAX_PLATFORMS=cpu

setup:
	uv sync --group dev

# Default dev loop: CPU-only, no slow/JAX-compile smokes (safe on WSL2 + NVIDIA).
test: test-fast

test-fast:
	$(PYTEST_CPU) -m "not slow and not jax and not sweep"

# Parallel CPU fast tier (~25-40s on 4 cores). Requires pytest-xdist in dev group.
test-fast-parallel:
	$(XDIST_ENV) $(PYTEST) -m "not slow and not jax and not sweep" $(PYTEST_XDIST)

# Lightweight JAX checks (metric math, action builders) — no rollout/training smokes.
test-jax:
	$(PYTEST_CPU) -m "jax and not slow and not sweep"

test-jax-parallel:
	$(XDIST_ENV) $(PYTEST) -m "jax and not slow and not sweep" $(PYTEST_XDIST)

# PERF1 gate: factorized sampler K=5 within 10% of main baseline (isolated process; not under pytest).
test-launch-hygiene-throughput:
	env -u JAX_COMPILATION_CACHE_DIR ORBIT_WARS_PYTEST_JAX_CACHE=0 \
		uv run ow benchmark factorized-sampler \
		--max-moves-k 5 --batch-size 32 --warmup 5 --repeats 20 --assert-max-ms 3.22

# PERF2 tier-2 gate: admission-shaped e2e throughput vs learning-first baseline (GPU).
# Tier-1 pass (above) does not imply tier-2 pass. Throughput may fail on hygiene branch.
test-launch-hygiene-e2e-throughput:
	env -u JAX_COMPILATION_CACHE_DIR ORBIT_WARS_PYTEST_JAX_CACHE=0 \
		uv run ow benchmark training \
		--preset admission --label launch_hygiene_e2e_gate \
		--updates 20 --warmup 2 --detailed-timing \
		--out /tmp/launch_hygiene_e2e_gate.json \
		--baseline docs/benchmarks/launch-hygiene-e2e-baseline-learning-first.json \
		--assert-within-pct 10

# Kaggle-relevant JAX env parity (mechanics + 4p); runs in test-jax tier, not slow.
test-kaggle-parity:
	$(PYTEST_CPU) tests/test_jax_env_parity.py tests/test_jax_env.py tests/test_jax_env_dispatch.py tests/test_map_pool_reset.py -m "jax and not slow"

# Fast + jax tiers in parallel (~75s wall vs ~140s serial).
test-daily:
	@$(MAKE) test-fast & p1=$$!; $(MAKE) test-jax & p2=$$!; \
	wait $$p1; r1=$$?; wait $$p2; r2=$$?; test $$r1 -eq 0 -a $$r2 -eq 0

test-daily-parallel:
	@$(MAKE) test-fast-parallel & p1=$$!; $(MAKE) test-jax-parallel & p2=$$!; \
	wait $$p1; r1=$$?; wait $$p2; r2=$$?; test $$r1 -eq 0 -a $$r2 -eq 0

# JAX compile/training smokes + bounded sweep sample; serial; module warmup in conftest.
test-slow:
	$(PYTEST_CPU) -m "slow and not sweep"

# Full W&B sweep Cartesian grid (~3+ min); weekly / explicit pre-release only.
test-sweep:
	$(PYTEST_CPU) -m sweep

# Pre-merge: daily tiers + slow (excludes full sweep grid).
test-premerge: test-daily test-slow

# Everything including full sweep grid (single pytest process).
test-premerge-all: test-daily test-slow test-sweep

test-full:
	$(PYTEST_CPU)

test-domain-config:
	$(PYTEST_CPU) tests/test_config_consolidation.py tests/test_telemetry.py tests/test_metric_registry.py tests/test_run_paths.py -m "not slow and not jax and not sweep"

test-domain-features:
	$(PYTEST_CPU) tests/test_feature_registry.py tests/test_feature_encoding_golden.py tests/test_intercept_edge_features.py -m "not slow and not jax"

test-domain-jax-env:
	$(PYTEST_CPU) tests/test_jax_env_parity.py tests/test_jax_env.py tests/test_jax_env_dispatch.py -m "jax and not slow"

test-domain-policy:
	$(PYTEST_CPU) tests/test_jax_policy_encoder.py tests/test_jax_ppo.py tests/test_ppo_update.py tests/test_trajectory_shield.py -m "jax and not slow"

test-domain-artifacts:
	$(PYTEST_CPU) tests/test_artifact_pipeline.py tests/test_promotion.py tests/test_replay.py tests/test_tournament.py tests/test_kaggle_submission_packager.py tests/test_checkpoint_compat.py -m "not slow and not jax"

test-domain-curriculum:
	$(PYTEST_CPU) tests/test_curriculum.py tests/test_jax_train_timing.py -m "not slow and not jax"

preflight-sanity:
	uv run ow benchmark sanity --out outputs/preflight/sanity_repro.json

preflight-learn-proof:
	uv run ow benchmark learn-proof --through beat_random --out outputs/preflight/learn_proof_report.json

preflight-calibrate:
	uv run ow benchmark calibrate --analyze-only --analyze-campaigns

