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

.PHONY: help agent-context

help:
	@echo "Orbit Wars Makefile targets"
	@echo ""
	@echo "Tests (default: make test = test-fast):"
	@echo "  test, test-fast          CPU fast tier"
	@echo "  test-jax                 Lightweight JAX tier"
	@echo "  test-daily               test-fast + test-jax (parallel)"
	@echo "  test-premerge            test-daily + test-slow"
	@echo "  test-sweep               Full W&B sweep grid (pre-release)"
	@echo "  test-fast-parallel       CPU xdist (ORBIT_WARS_PYTEST_XDIST=1)"
	@echo "  test-jax-trace-hygiene   tier-A static rg gate + jit import/smoke tests"
	@echo "  test-domain-{config,features,jax-env,policy,artifacts,curriculum}"
	@echo "  test-cov-fast            fast tier + HTML coverage (htmlcov/)"
	@echo "  test-cov-report          fast tier + coverage.xml artifact"
	@echo ""
	@echo "Preflight (GPU for learn-proof; see docs/operator-runbook.md):"
	@echo "  preflight-sanity, preflight-learn-proof, preflight-calibrate"
	@echo "  gate-admission            learning + throughput (one JSON)"
	@echo "  sweep-ppo-admission       3-seed admission sweep (ADMISSION_SEEDS, REPO_ROOT)"
	@echo ""
	@echo "Launch hygiene throughput (tier-1 CPU-safe; tier-2 GPU):"
	@echo "  test-launch-hygiene-throughput      tier-1 sampler microbench"
	@echo "  test-launch-hygiene-e2e-throughput  tier-2 production-path e2e gate"
	@echo ""
	@echo "Agents:"
	@echo "  agent-context            JSON session context for coding agents"
	@echo "                           RESOLVED=smoke embeds truncated resolved config"
	@echo "  See docs/AGENT_CAPABILITIES.md and AGENTS.md"

agent-context:
	uv run python scripts/agent_context.py $(if $(filter smoke,$(RESOLVED)),--resolved smoke,)

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

test-jax-trace-hygiene:
	./scripts/jax_trace_hygiene.sh
	$(PYTEST_CPU) tests/test_jax_trace_hygiene.py -m "jax and not slow and not sweep"

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
	$(PYTEST_CPU) tests/test_artifact_pipeline.py tests/test_promotion.py tests/test_replay.py tests/test_tournament.py tests/test_kaggle_submission_packager.py tests/test_checkpoint_compat.py tests/test_eval_validate_invariant.py -m "not slow and not jax"

test-domain-curriculum:
	$(PYTEST_CPU) tests/test_curriculum.py tests/test_jax_train_timing.py -m "not slow and not jax"

# Line coverage on fast tier (serial CPU; supplements behavioral gates, not a substitute).
COV_FAST := $(PYTEST_CPU) -m "not slow and not jax and not sweep" --cov=src --cov-report=term-missing:skip-covered

test-cov-fast:
	$(COV_FAST) --cov-report=html:htmlcov

test-cov-report:
	$(COV_FAST) --cov-report=xml:coverage.xml

preflight-sanity:
	uv run ow benchmark sanity --out outputs/preflight/sanity_repro.json

preflight-learn-proof:
	# Gates 2–5 ladder: beat_noop → beat_random (+ optional tournament-proof via learn-proof).
	# Dry-run first: uv run ow benchmark gate run beat_noop --dry-run
	# Thresholds: docs/benchmarks/preflight-calibration.json (see docs/operator-runbook.md)
	uv run ow benchmark learn-proof --through beat_random --out outputs/preflight/learn_proof_report.json

preflight-calibrate:
	uv run ow benchmark calibrate --analyze-only --analyze-campaigns

# Cherry-pick admission: beat_noop learning + throughput extract in one gate JSON.
# Map-pool picks: ADMISSION_BASELINE=docs/benchmarks/launch-hygiene-e2e-baseline-map-pool.json \
#   make gate-admission REPO_ROOT=../orbit_wars-integration
ADMISSION_BASELINE ?= docs/benchmarks/launch-hygiene-e2e-baseline-learning-first.json
ADMISSION_OUT ?= outputs/benchmarks/admission/gate.json
ADMISSION_SEEDS ?= 42 43 44

gate-admission:
	@mkdir -p $(dir $(ADMISSION_OUT))
	uv run ow benchmark gate run admission \
	  --out $(ADMISSION_OUT) \
	  --throughput-baseline $(ADMISSION_BASELINE) \
	  $(if $(REPO_ROOT),--repo-root $(REPO_ROOT),)

# Simple 3-run PPO admission sweep; pass REPO_ROOT for anchor worktree gates.
sweep-ppo-admission:
	@set -e; \
	mkdir -p outputs/benchmarks/admission; \
	for seed in $(ADMISSION_SEEDS); do \
	  echo "=== admission seed=$$seed ==="; \
	  uv run ow benchmark gate run admission \
	    --out outputs/benchmarks/admission/seed_$${seed}.json \
	    --train-overrides seed=$$seed \
	    --throughput-baseline $(ADMISSION_BASELINE) \
	    $(if $(REPO_ROOT),--repo-root $(REPO_ROOT),); \
	done

