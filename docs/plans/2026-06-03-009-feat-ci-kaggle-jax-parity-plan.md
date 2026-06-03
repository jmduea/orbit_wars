---
date: 2026-06-03
topic: ci-kaggle-jax-parity
status: completed
origin: LFG request; AGENTS.md `make test-kaggle-parity`
---

# Plan: CI enforces Kaggle/JAX env parity

## Summary

Add a GitHub Actions job that runs `make test-kaggle-parity` on pull requests and pushes to the default branch so JAX env mechanics cannot drift from the Kaggle reference without failing CI.

## Problem Frame

Parity is verified locally via `make test-kaggle-parity` (`tests/test_jax_env_parity.py` plus related jax-env tests). The repo has only a label-sync workflow today; regressions can merge without running parity on a clean Linux runner.

## Requirements

| ID | Requirement |
|----|-------------|
| R1 | Workflow triggers on `pull_request` and on `push` to `main` / `master` |
| R2 | Job runs `uv sync --group dev` then `make test-kaggle-parity` on `ubuntu-latest` |
| R3 | CPU JAX only (`JAX_PLATFORMS=cpu` — already set by Makefile `PYTEST_CPU`) |
| R4 | No invented thresholds; pass/fail is pytest exit code |
| R5 | Document the check in PR body / onboarding pointer if missing |

## Key Technical Decisions

**KTD1 — Dedicated workflow file.** New `.github/workflows/kaggle-jax-parity.yml` mirrors minimal pattern from `sync-labels.yml` (permissions, ubuntu-latest).

**KTD2 — Python 3.12 + uv.** Use `actions/setup-python@v5` with `3.12` and `astral-sh/setup-uv@v5`; match repo `requires-python`.

**KTD3 — Ship with comet branch when open.** Land CI on the active feature branch (`feat/jax-comet-subsystem`, PR #188) so one merge enables enforcement and closes comet parity work; no separate CI-only branch unless review demands split.

## Implementation Units

### U1. GitHub Actions workflow

**Files:** `.github/workflows/kaggle-jax-parity.yml` (new)

**Approach:** Single job `kaggle-parity`; checkout → setup Python 3.12 → setup uv → `uv sync --group dev` → `make test-kaggle-parity`.

**Test scenarios:**
- Workflow YAML is valid (push triggers run)
- Local: `make test-kaggle-parity` passes before push

**Requirements:** R1–R4

### U2. Operator visibility (optional, minimal)

**Files:** `docs/ONBOARDING.md` or `README.md` — one line noting CI runs parity on PRs if not already stated.

**Requirements:** R5

**Test expectation:** none — documentation only.

## Scope Boundaries

### In scope

- CI job for `make test-kaggle-parity` only (not full `test-premerge` or GPU tiers).

### Deferred to Follow-Up Work

- Broader CI matrix (`test-daily`, slow tier, launch-hygiene GPU gates).

### Out of scope

- Changing parity test assertions or thresholds
- Merging comet implementation (separate PR #188 content; same merge if already on branch)

## Verification

- Local: `make test-kaggle-parity`
- After push: GitHub Actions check green on PR

## Assumptions

- `kaggle-environments` installs on Ubuntu CI without extra secrets.
- First CI run may pay JAX compile cost; job timeout ≥ 15 minutes if needed.
