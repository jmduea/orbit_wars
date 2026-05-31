# Ralplan: Test Suite Speed and Dev-Loop Efficiency

Linked spec: `.omg/specs/deep-interview-test-efficiency.md` (`test-efficiency`)

## RALPLAN-DR Summary

### Principles

1. **Opt-out slow, not opt-in fast** — mark only expensive/integration tests `@pytest.mark.slow`; default dev loop runs `-m "not slow"`.
2. **Marker expressions over file lists** — domain Makefile targets use markers because files like `test_curriculum.py` mix fast unit tests and slow JAX rollouts.
3. **Measure before optimizing** — sweep split + tier wiring first; JAX fixtures and xdist only where evidence shows remaining budget pressure.
4. **JAX safety** — serial execution only; xdist rejected after WSL2 crash during verification.
5. **Full tier unchanged in coverage** — all 155 items remain in `make test-full`; no deleted tests.

### Decision Drivers

1. Fast tier must complete in **<60s** on reference dev machine (currently 220s for config tier due to one test).
2. Developers need **reliable domain commands** and **IDE marker filtering** without running the 11–15 min full suite locally.
3. **No CI** in this effort — local Makefile/docs are the contract.

### Viable Options (consensus-selected: **Option B**)

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A: Markers + sweep split only** | `-m "not slow"`, Makefile targets, defer fixtures/xdist | Lowest risk, fastest to ship | May miss <60s without xdist; JAX domains still slow |
| **B: Tiered markers + sweep split + module JAX fixtures + xdist on fast** | Full spec scope with architect/critic guardrails | Meets all acceptance criteria | More conftest/Makefile complexity |
| **C: CPU-only fast tier** | Fast = pure Python only; all JAX in slow | Predictable <60s | JAX edits get no fast feedback |

**Chosen: Option B** with opt-out `slow` (architect adjustment) and phased fixture rollout (critic adjustment).

---

## ADR: Test Tier Architecture

**Decision:** Introduce pytest markers with opt-out `slow`, auto-applied domain markers, Makefile tier targets, bounded sweep smoke + full slow sweep, module-scoped JAX fixtures for homogeneous test groups, and `pytest-xdist` limited to `-m "not slow"`.

**Drivers:** 205s sweep test in routine runs; 679s full suite minus sweep; mixed fast/slow tests in single files.

**Alternatives considered:**
- Mark all 155 tests `fast` explicitly → rejected (high drift, parametrization pain).
- Single session-scoped JAX fixture → rejected (architecture/shape conflicts across tests).
- xdist on JAX domains → rejected (GPU OOM/flakes on NVIDIA WSL2).

**Consequences:**
- `make test` stays the full suite (unchanged from today).
- Developers run `make test-fast` for daily edits; `make test` before sharing/merging.
- New sweep smoke test adds +1 collected item (156 total) unless full test is renamed in place.

---

## Implementation Plan

### Phase 1 — Markers and sweep split (highest leverage)

**Files:** `pyproject.toml`, `tests/conftest.py`, `tests/test_config_consolidation.py`, all slow test files.

1. Register markers in `pyproject.toml`:
   - `slow`, `jax`
   - Domain: `config`, `features`, `jax_env`, `policy`, `artifacts`, `curriculum`
   - Enable `--strict-markers`

2. Add `tests/conftest.py` collection hook to auto-apply domain markers from file path patterns (reduce manual tagging).

3. Explicitly mark `@pytest.mark.slow` on:
   - `test_wandb_sweep_campaign_samples_compose_full` (renamed from current full combinatorial)
   - `test_training_loop_logs_curriculum_events_on_same_update`
   - JAX rollout/PPO integration tests in `test_curriculum.py` (lines ~165+)
   - Parametrized 4-arch smoke and other >20s tests in `test_jax_ppo.py`
   - Entire `test_jax_env_parity.py` (parity is slow-tier guardrail)

4. Split sweep test in `test_config_consolidation.py`:
   - **`test_wandb_sweep_yaml_smoke_compose`** (fast): first value per parameter per YAML file (~13 composes)
   - **`test_wandb_sweep_campaign_samples_compose_full`** (`slow`): existing Cartesian product (~2,302 composes)

5. Mark JAX-using tests with `@pytest.mark.jax` for IDE filtering.

**Projected fast tier:** ~70–80 tests, ~17–25s serial (measured excluding sweep + slow JAX).

### Phase 2 — Makefile and docs

**Files:** `Makefile`, `AGENTS.md`, `docs/ONBOARDING.md`, optionally `.vscode/settings.json`.

```makefile
test:
	uv run --group dev pytest

test-fast:
	uv run --group dev pytest -m "not slow and not jax"

test-jax:
	uv run --group dev pytest -m "jax and not slow"

test-full: test

test-domain-config:
	uv run --group dev pytest -m "config and not slow"

test-domain-features:
	uv run --group dev pytest -m "features"

test-domain-jax-env:
	uv run --group dev pytest -m "jax_env"

test-domain-policy:
	uv run --group dev pytest -m "policy"

test-domain-artifacts:
	uv run --group dev pytest -m "artifacts"

test-domain-curriculum:
	uv run --group dev pytest -m "curriculum and not slow"
```

Document xdist policy: **only `test-fast` uses `-n auto`**; JAX domain targets run serially.

Fix stale `AGENTS.md` references to removed `test_env.py` / `test_evaluate.py`.

Optional IDE snippet in ONBOARDING:
`"python.testing.pytestArgs": ["-m", "not slow"]`

### Phase 3 — Module-scoped JAX fixtures (if Phase 1 exceeds 60s with xdist)

**Files:** `tests/conftest.py`, optionally `tests/test_jax_ppo.py`, `tests/test_curriculum.py`.

- Add `small_train_config(**overrides)` helper with canonical tiny dims.
- Module-scoped fixtures keyed by `(architecture, player_count, num_envs)` for homogeneous groups.
- Use indirect parametrization for 4-arch smoke so each architecture pays init once per module.
- Do **not** share mutable `train_state` across tests that mutate params.
- Collect before/after `--durations=10` evidence on `test_jax_ppo.py` and `test_curriculum.py`.

### Phase 4 — pytest-xdist

**Files:** `pyproject.toml` (add `pytest-xdist` to dev group).

- Verify `make test-fast` passes twice consecutively with `-n auto --dist loadscope`.
- Document NVIDIA/WSL2 caveats and `ORBIT_WARS_ALLOW_CPU_JAX_ON_NVIDIA` interaction.

---

## Test Classification (summary)

| Tier | Rule | Approx count |
|------|------|--------------|
| Fast | `-m "not slow"` | ~75–85 items |
| Slow | `@pytest.mark.slow` | ~70–80 items |
| Full | no filter | ≥155 items (156 if sweep split adds smoke) |

| Domain marker | Primary files |
|---------------|---------------|
| `config` | `test_config_consolidation.py`, `test_telemetry.py`, `test_metric_registry.py`, `test_run_paths.py`, fast curriculum config tests |
| `features` | `test_features.py`, `test_feature_history.py`, `test_feature_registry.py`, `test_normalization.py` |
| `jax_env` | `test_jax_env.py`, `test_jax_env_parity.py` |
| `policy` | `test_jax_policy.py`, `test_jax_ppo.py`, `test_trajectory_shield.py` |
| `artifacts` | `test_artifact_pipeline.py`, `test_replay.py`, `test_kaggle_submission_packager.py` |
| `curriculum` | `test_curriculum.py`, `test_jax_train_timing.py` |

---

## Verification Protocol

```bash
# Inventory
uv run --group dev pytest --collect-only -q

# Fast tier budget (acceptance #1)
/usr/bin/time -f '%e sec' make test-fast

# Full tier (acceptance #2)
make test-full

# Markers (acceptance #3)
uv run --group dev pytest --markers

# Sweep not in fast tier (acceptance #5)
uv run --group dev pytest -m "not slow" -k "sweep_campaign_samples_compose_full" --collect-only -q  # expect 0

# xdist stability (acceptance #7)
make test-fast && make test-fast

# JAX fixture evidence (acceptance #6, if Phase 3 executed)
uv run --group dev pytest tests/test_jax_ppo.py --durations=10 -q
```

Record wall-clock, collect counts, and top-10 durations in completion evidence.

---

## Architect Review

**Verdict:** Conditionally approved with adjustments incorporated above (opt-out slow, marker expressions, module fixtures, xdist on fast only).

## Critic Review

**Verdict:** Revise → addressed via classification matrix, sweep contract, xdist policy, IDE/docs tasks, and verification protocol in this plan.

## Resolved Decision

**`make test` semantics:** User chose **keep `make test` = full suite**; add **`make test-fast`** for the daily <60s loop.

---

## Workflow Gates

- [ ] Phase 1 complete + fast tier measured
- [ ] Phase 2 docs/Makefile aligned
- [ ] Phase 3 fixtures (if needed)
- [ ] Phase 4 xdist verified
- [ ] Full tier green with evidence recorded
