# Test Suite Speed and Dev-Loop Efficiency Spec

Generated: 2026-05-24
Workflow: deep-interview
Final ambiguity: 13%

## Goal

Speed up the day-to-day Python test feedback loop without reducing coverage or correctness. Developers should have a **fast tier under 60 seconds** for routine edits, **reliable domain-scoped commands** when working in a subsystem, and **pytest markers/fixtures** that make IDE "run related tests" workflows trustworthy. The full suite remains the pre-share/pre-merge correctness gate.

## Context

Brownfield repo with **155 pytest items** across 18 files under `tests/`. No CI today. Minimal pytest config in `pyproject.toml` (`testpaths` only); no markers, no xdist, no shared JAX warmup fixtures.

**Measured baseline (2026-05-24, this workstation):**

| Scope | Duration | Notes |
|-------|----------|-------|
| Config/feature candidate tier (59 tests) | **220s** | Dominated by `test_wandb_sweep_campaign_samples_compose` (~205s, ~2,300 Hydra composes) |
| Full suite minus sweep test (154 tests) | **679s (~11 min)** | JAX rollout/PPO/curriculum tests dominate |
| Sweep test alone | **~205s** | Single function in `test_config_consolidation.py` |

Top slow tests excluding sweep: `test_training_loop_logs_curriculum_events_on_same_update` (~115s), curriculum rollout tests (~20–43s each), JAX PPO smoke/rollout tests (~18–27s each, multiplied across 4 architectures in parametrized smoke).

Existing docs (`AGENTS.md`, `docs/ONBOARDING.md`) list domain file groupings but provide no executable fast/slow split or Makefile targets.

## Requirements

### 1. Pytest markers and tier definitions

Introduce explicit markers, registered in `pyproject.toml`:

| Marker | Purpose |
|--------|---------|
| `fast` | Safe for daily edits; target **<60s aggregate** on a typical dev machine |
| `slow` | Integration/heavy tests deferred from fast tier |
| `jax` | Requires JAX import/JIT (may overlap with `slow`) |
| Domain markers (optional but recommended) | `config`, `features`, `jax_env`, `policy`, `artifacts`, `curriculum` — align with ONBOARDING groupings |

Every test must carry appropriate markers. Default `pytest` invocation (no args beyond `testpaths`) should run **fast-only** or document clearly that `make test` = fast tier and `make test-full` = everything.

### 2. Executable tier commands

Add Makefile targets (or thin scripts) with stable names:

- `make test` or `make test-fast` — fast tier only
- `make test-domain-<name>` — documented domain subsets (config, env, policy, artifacts, …)
- `make test-full` — all tests including `slow`

Update `AGENTS.md` and `docs/ONBOARDING.md` so tier commands match implementation (remove stale references to deleted test files).

### 3. Split or shrink known slow tests

**In scope:**

- **`test_wandb_sweep_campaign_samples_compose`**: stop full Cartesian product over all sweep YAML grids in the default tier. Replace with bounded sampling, per-file smoke composes, or move to `slow` with a reduced default assertion set. Preserve intent: sweep YAMLs remain composable.
- **`test_training_loop_logs_curriculum_events_on_same_update`**: mark `slow`; reduce work where possible (fewer updates, slimmer config) without losing the curriculum-event assertion.
- Review other tests >20s for fixture reuse or slimmer configs.

### 4. Shared JAX warmup/session fixtures

Add session-scoped (or module-scoped) fixtures in `tests/conftest.py` (or domain conftest files) to amortize:

- JAX policy build + `init_train_state`
- Common small test configs reused across `test_jax_ppo.py`, `test_jax_policy.py`, `test_curriculum.py`

Goal: cut repeated JIT compilation across tests in the same session without hiding per-test isolation bugs.

### 5. Serial execution only (WSL2 safety)

Do **not** use `pytest-xdist` or `-n auto`. Parallel workers that each import JAX/CUDA exhausted GPU memory and crashed WSL2 during implementation. `tests/conftest.py` rejects xdist at startup.

- `make test-fast`: CPU-only daily loop (`-m "not slow and not jax"`), serial
- `make test-jax`: JAX tests outside slow tier, serial
- `make test`: full suite, serial

### 6. IDE integration

Ensure markers are registered so VS Code/Cursor pytest discovery can filter by marker expressions. Document recommended `.vscode` settings snippets if helpful (e.g. fast-tier args).

## Non-Goals

- **No CI/GitHub Actions** in this effort (explicitly out of scope for now).
- No reduction in total test count or removal of parity/JAX correctness coverage from the **full** tier.
- No skipping slow tests silently in full tier.
- MCP server Node tests (`mcp-server/`) remain a separate package; optional doc cross-link only.

## Acceptance Criteria

1. **Fast tier completes in <60s** on the reference dev machine used for baseline (document command and measured time in PR/commit notes).
2. **Full tier retains all 155 tests** (or equivalent coverage if tests are split, not deleted) and passes.
3. Markers are registered; `pytest --markers` lists tiers and domains.
4. Makefile (or script) targets exist for fast, domain, and full tiers; docs match commands.
5. `test_wandb_sweep_campaign_samples_compose` is no longer in the fast tier and no longer dominates routine runs (~205s → deferred or bounded).
6. Shared JAX fixtures demonstrably reduce repeated-init overhead (note before/after slowest-test timings in evidence).
7. xdist is **not used**; conftest blocks parallel workers; docs warn about WSL2 + CUDA.

## Assumptions Exposed & Resolved

| Assumption | Resolution |
|------------|------------|
| Pain is local dev wait time | **Confirmed** — primary pain is local `pytest` during coding |
| Also need targeted-run clarity | **Confirmed** — domain commands + IDE markers, not just faster full suite |
| Fast tier <60s for daily edits | **Confirmed** — full suite only before sharing/merging |
| CI is part of the fix | **Rejected** — out of scope for this effort |
| Markers, split slow tests, JAX fixtures, xdist acceptable | **Confirmed** |
| Coverage/correctness non-negotiable | **Confirmed** — full tier unchanged in intent |

## Ontology

- **Fast tier**: Marker-selected tests completing in <60s; default dev loop.
- **Slow tier**: Integration/JAX-heavy/deferred tests; required before share/merge.
- **Domain tier**: Subsystem-scoped pytest selection aligned with code ownership (config, env, policy, …).
- **JAX warmup fixture**: Session/module fixture amortizing policy/train-state initialization across tests.
- **Sweep compose test**: Combinatorial Hydra validation in `test_config_consolidation.py`; primary fast-tier blocker.

## Interview Transcript

1. **Primary pain (Goal):** A — local dev feedback loop (long `pytest` waits) **+** D — better targeted runs / clearer tooling.
2. **Success criteria:** A — fast tier <60s for daily edits, full suite before sharing **+** B — domain-scoped commands **+** D — IDE-first markers/fixtures for "run related tests".
3. **In-scope changes:** markers/tiers, split/shrink slow tests, shared JAX fixtures, pytest-xdist. **Not selected:** CI.
