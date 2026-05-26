# Plan: Kaggle W&B Population Training

**Source spec:** `.omg/specs/deep-interview-kaggle-wandb-population.md`  
**Slug:** `kaggle-wandb-population`  
**Workflow:** ralplan  
**Status:** planned (consensus iter-1)  
**Date:** 2026-05-26

---

## Executive Summary

Build the Population MVP as a W&B-native orchestration flow. W&B sweeps are the first-class population controller: they own candidate assignment, run identity, metrics, and checkpoint artifact lineage. The repo supplies a one-command launcher that creates or targets a W&B sweep, packages a Kaggle worker kernel, pushes/runs that kernel through the Kaggle CLI, polls status, syncs diagnostic outputs, and queries W&B for a candidate shortlist.

The MVP should avoid a custom repo-owned population manifest unless W&B sweeps cannot express the needed candidate space or resume semantics. A later hybrid manifest remains an explicit fallback.

---

## RALPLAN-DR Summary

### Principles

1. **W&B is the population control plane** — candidate configs, run identity, metrics, and checkpoint artifacts should be visible and queryable in W&B first.
2. **Kaggle is an execution backend** — local code should handle packaging, accelerator fallback, status polling, and output sync, not become a second experiment database.
3. **Keep training entrypoints stable** — workers should still launch via `uv run python -m src.train ...` with normal Hydra overrides.
4. **Calibrate before long runs** — spend a bounded GPU budget to find stable high-throughput settings before committing longer worker time.
5. **Make the MVP testable without Kaggle** — split pure sweep/render/status/shortlist logic from live Kaggle CLI and W&B API calls.

### Decision Drivers

1. **User selected Native W&B first** — W&B sweeps should be attempted before introducing a custom population manifest.
2. **Checkpoint handoff is artifact-driven** — W&B checkpoint artifacts are canonical; Kaggle outputs are diagnostics.
3. **Kaggle accelerator behavior is operationally uncertain** — the launcher needs ordered fallback and in-kernel verification.
4. **Throughput calibration is part of correctness** — the MVP is not just "it runs"; it must avoid obviously underutilized GPU settings.
5. **Tournament plumbing is intentionally deferred** — the plan should stop at W&B-backed shortlist generation, leaving scripted-nearest/tournament promotion as follow-up work.

---

## Viable Options Considered

| Option | Description | Pros | Cons | Verdict |
|---|---|---|---|---|
| Native W&B Sweep First | W&B sweep config defines candidate population; Kaggle kernels run worker jobs that consume sweep assignments. | Best matches user preference; W&B UI/API remains source of truth; clean comparison across candidates. | Some Kaggle lifecycle state lives outside W&B unless mirrored as run metadata. | **Chosen** |
| Local Population Manifest First | Repo-owned YAML expands candidate configs, then launcher creates W&B runs. | Easiest to unit test and customize; good Kaggle control. | Creates a parallel orchestration layer earlier than needed. | Rejected for MVP |
| Hybrid Manifest + W&B Runs | Repo manifest owns candidates and Kaggle lifecycle; W&B owns results/artifacts. | Strong fallback if sweeps are too restrictive. | More moving parts; risks diluting W&B-first intent. | Deferred fallback |

---

## ADR

### Decision

Implement the Population MVP around W&B sweeps first. Add a thin local Kaggle launcher/worker toolkit that can:

- create or use a W&B sweep from repo-provided sweep YAML,
- package a Kaggle script kernel,
- request preferred accelerators with ordered fallback,
- run a calibration stage inside the worker,
- launch training through the existing Hydra entrypoint,
- upload checkpoints as W&B artifacts,
- poll Kaggle status and download outputs for diagnostics,
- query W&B after completion to produce a promotion shortlist.

### Alternatives

- A custom repo population manifest was considered but deferred.
- A full tournament/promotion controller was considered out of MVP scope.
- Continue-run mode remains a later mode, not the first implementation track.

### Why This Decision

The user wants W&B-first orchestration as long as the one-command launch ergonomics remain. Native sweeps best preserve W&B as the visible experiment system while the repo code handles the Kaggle-specific mechanics that W&B does not cover directly.

### Consequences

- The first config source should be a W&B sweep YAML, not a new population YAML.
- The launcher needs W&B API integration for sweep creation/status/shortlisting.
- Kaggle kernel status must be mirrored into W&B run metadata and local logs so operator state is not lost.
- If W&B sweeps cannot express candidate resume/promotion cleanly, the later hybrid manifest should be introduced deliberately.

---

## Architecture

### New Modules And Scripts

Add a new orchestration package under `src/orchestration/`:

- `src/orchestration/kaggle_cli.py` — typed wrapper around `kaggle kernels push/status/files/output`; no training logic.
- `src/orchestration/wandb_sweeps.py` — sweep creation/resolution, run query, artifact query, shortlist data access.
- `src/orchestration/population.py` — pure data model for candidate, calibration result, worker status, shortlist row.
- `src/orchestration/throughput.py` — hardware-aware parameter estimates and bounded calibration sweep generation.
- `src/orchestration/kernel_package.py` — render a Kaggle worker directory from templates.

Add scripts:

- `scripts/kaggle_wandb_population.py` — local one-command launcher and status/sync/shortlist CLI.
- `scripts/kaggle_worker_entry.py` — script executed inside Kaggle kernel.

Add templates/config:

- `conf/sweeps/wandb/kaggle_population_mvp.yaml` — first W&B population sweep.
- `conf/orchestration/kaggle_population.yaml` or `conf/kaggle/population.yaml` — non-population operational defaults only: accelerator preference, kernel timeout, calibration budget, output sync policy.
- `templates/kaggle/kernel-metadata.json` or generated equivalent.

### W&B-First Boundary

W&B owns:

- sweep ID,
- candidate parameter assignment,
- run identity,
- run group/tags,
- metrics,
- checkpoint artifacts,
- calibration metadata,
- shortlist query inputs.

Local launcher owns:

- Kaggle kernel packaging,
- accelerator fallback attempts,
- status polling,
- output download,
- local diagnostic ledger.

Local diagnostic state must be useful but non-canonical. A failed local ledger should not make W&B runs/checkpoints unusable.

### Kaggle Worker Flow

Each worker kernel should:

1. Print Python, CUDA, JAX, GPU, and memory diagnostics.
2. Install/sync project dependencies.
3. Start or attach to W&B sweep/agent context.
4. Receive candidate overrides from W&B.
5. Estimate throughput parameters from observed accelerator and model shape.
6. Run bounded calibration trials with short `training.total_updates`.
7. Select stable high-throughput overrides.
8. Run the longer candidate training command via `uv run python -m src.train ...`.
9. Upload checkpoints to W&B artifacts.
10. Emit a final worker summary JSON into Kaggle outputs.

### Checkpoint Artifact Flow

Current `TelemetryLogger.log_checkpoint()` already emits W&B checkpoint artifacts when `telemetry.wandb.log_artifacts=true`. The MVP should harden this path:

- set `telemetry.wandb.log_artifacts=true` for Kaggle workers,
- include candidate ID, sweep ID, accelerator, update, and run ID in artifact aliases/metadata,
- mark latest usable checkpoint with a stable alias such as `latest`, `candidate-{id}-latest`, or W&B artifact aliases if available,
- keep existing local checkpoint retention to avoid excessive Kaggle disk usage.

If W&B artifact aliasing is awkward with the current logger, add a minimal telemetry method for artifact metadata/aliases rather than rewriting the telemetry layer.

---

## Implementation Plan

### Phase 0 — Dry-Run Contracts

Goal: make the orchestration shape testable without network or Kaggle.

Tasks:

- Define dataclasses for `AcceleratorPreference`, `KaggleKernelRef`, `CalibrationPlan`, `CalibrationResult`, `PopulationCandidate`, and `ShortlistRow`.
- Add command rendering for `uv run python -m src.train` with Hydra overrides.
- Add pure generation of W&B sweep YAML from an existing repo sweep plus Kaggle-specific tags/group.
- Add unit tests for command rendering, accelerator fallback ordering, and shortlist ranking.

Verification:

- `make test-domain-config`
- targeted tests for new orchestration pure logic, using `uv run --group dev pytest ... -m "not slow and not jax"`

### Phase 1 — Local Launcher Skeleton

Goal: one command can prepare and dry-run a Kaggle W&B population campaign.

Tasks:

- Implement `scripts/kaggle_wandb_population.py` subcommands:
  - `prepare`
  - `launch --dry-run`
  - `status`
  - `sync-output`
  - `shortlist`
- Implement Kaggle CLI wrapper with subprocess calls and parsed status output.
- Render a Kaggle kernel work directory with metadata and worker entry script.
- Support ordered accelerator list, timeout, kernel slug prefix, and W&B sweep ID.
- Keep all live calls behind injectable command runners for tests.

Verification:

- Unit tests with fake Kaggle CLI output.
- `make test-domain-config` if Hydra config is touched.

### Phase 2 — W&B Sweep Integration

Goal: W&B is the real population source of truth.

Tasks:

- Add W&B API helpers to create or resolve a sweep.
- Use `conf/sweeps/wandb/kaggle_population_mvp.yaml` as the initial population definition.
- Ensure candidate runs receive consistent tags: `kaggle`, `population`, campaign slug, accelerator request, and model family.
- Query W&B runs for metrics and checkpoint artifact presence.
- Implement shortlist ranking by cheap filter:
  - completed status,
  - checkpoint artifact present,
  - stable calibration,
  - `episode_reward_mean`,
  - `samples_per_sec` / `ppo_samples_per_sec`.

Verification:

- Unit tests mock W&B API objects.
- No live W&B calls in default tests.

### Phase 3 — Kaggle Worker MVP

Goal: a pushed Kaggle kernel can run a W&B-assigned candidate.

Tasks:

- Implement `scripts/kaggle_worker_entry.py`.
- In-kernel diagnostics:
  - `nvidia-smi` when present,
  - `jax.devices()`,
  - GPU memory when discoverable,
  - installed package versions.
- Fail fast if JAX is not GPU-backed for non-validation runs.
- Install/sync dependencies using `uv` or a fallback pip path if Kaggle lacks `uv`.
- Start W&B agent or equivalent W&B sweep execution.
- Apply candidate overrides to the Hydra training command.
- Disable Kaggle-hostile optional work by default:
  - Docker validation off,
  - replay generation off unless explicitly requested,
  - checkpoint cadence tuned for artifact upload.

Verification:

- Local `--dry-run-worker` command renders and prints the exact Kaggle command.
- Tiny local worker smoke with CPU allowed only under explicit validation flag.

### Phase 4 — Throughput Calibration

Goal: avoid long runs with bad GPU utilization.

Tasks:

- Estimate initial settings from:
  - observed GPU memory,
  - model hidden size/layers/heads,
  - player count mix,
  - `trajectory_shield_horizon`,
  - feature history/horizon,
  - current default `rollout_steps`, `num_envs`, `minibatch_size`, `rollout_microbatch_envs`, and update chunk bounds.
- Generate a bounded calibration sweep around those estimates.
- Run short calibration jobs before long candidate training.
- Record calibration metrics in W&B:
  - selected overrides,
  - rejected overrides,
  - OOM/failure reasons,
  - `samples_per_sec`,
  - `ppo_samples_per_sec`,
  - update seconds.
- Select the best stable setting and launch the longer candidate segment.

Verification:

- Pure estimator tests for several accelerator memory tiers.
- Fake calibration result tests for selection logic.
- Manual live Kaggle smoke before claiming end-to-end acceptance.

### Phase 5 — Live Population MVP

Goal: prove the complete MVP with a small Kaggle run.

Tasks:

- Run one tiny validation kernel.
- Run a small population campaign with multiple W&B candidate runs.
- Confirm W&B run configs/metrics/checkpoint artifacts.
- Sync Kaggle outputs locally.
- Generate promotion shortlist from W&B.
- Document exact launch/status/sync/shortlist commands in `docs/`.

Verification:

- `make test-fast` before live run.
- Live Kaggle evidence:
  - kernel IDs,
  - W&B sweep ID,
  - W&B run IDs,
  - checkpoint artifact names,
  - shortlist output path.

---

## Critic Review

### Iteration 1 Findings

Status: **approved with constraints**.

Concerns addressed in this plan:

- **Risk: W&B sweep agents may not map cleanly to Kaggle kernel lifecycle.** Mitigation: keep a local Kaggle launcher responsible for kernel lifecycle, but W&B remains the population source of truth.
- **Risk: checkpoint artifacts are currently minimal.** Mitigation: harden existing telemetry artifact logging with metadata/aliases only as needed.
- **Risk: live Kaggle tests are not suitable for normal CI.** Mitigation: all core rendering/ranking/fallback logic must be unit-tested with fake CLIs/APIs; live Kaggle is a manual acceptance step.
- **Risk: calibration could become a second research project.** Mitigation: bounded estimator-plus-sweep only; cross-accelerator calibration deferred.
- **Risk: repo-local manifest sneaks back in.** Mitigation: local diagnostic ledger is allowed, but candidate assignment and result lineage stay in W&B for MVP.

---

## Test Strategy

Default repo verification must follow agent testing rules: use `make test-fast` or domain targets, not slow/JAX compile tiers unless explicitly requested.

Fast tests:

- Orchestration dataclass validation.
- Kaggle CLI command rendering.
- Kaggle status parsing.
- Accelerator fallback selection.
- Kernel package rendering.
- W&B sweep config generation.
- W&B API query adaptation with fake objects.
- Shortlist ranking and filtering.
- Throughput estimate and calibration selection pure logic.

Domain checks:

- `make test-domain-config` after adding config groups or sweep YAML.
- `make test-domain-artifacts` if checkpoint artifact metadata/aliases touch artifact pipeline behavior.
- `make test-fast` before any live Kaggle run.

Manual/live checks:

- Kaggle CLI auth check.
- W&B auth check.
- `launch --dry-run`.
- Tiny Kaggle GPU smoke.
- Small population campaign.
- W&B artifact and shortlist inspection.

Do not run full slow/JAX suites as routine iteration.

---

## Acceptance Checklist

- [ ] `scripts/kaggle_wandb_population.py launch --dry-run` renders a valid Kaggle worker package.
- [ ] Launcher can create/resolve a W&B sweep from `conf/sweeps/wandb/kaggle_population_mvp.yaml`.
- [ ] Launcher can request accelerator IDs in ordered fallback.
- [ ] Worker verifies GPU-backed JAX before real training.
- [ ] Worker runs bounded calibration and logs selected parameters.
- [ ] Worker launches `uv run python -m src.train` with W&B candidate overrides.
- [ ] Checkpoints upload as W&B artifacts.
- [ ] Local `status` can report Kaggle kernel state.
- [ ] Local `sync-output` can download Kaggle diagnostics.
- [ ] Local `shortlist` queries W&B and emits ranked candidates.
- [ ] A small live Kaggle population run proves multiple candidate runs and checkpoint artifacts.

---

## Follow-Up Work After MVP

- Continue-run mode from latest W&B checkpoint artifact.
- Scripted-nearest evaluation gate.
- Candidate-vs-baseline and candidate-vs-current-best tournaments.
- Candidate-vs-candidate tournament matrix.
- Action-distribution summaries and diversity-aware promotion.
- Self-play pool mutation from promoted artifacts.
- Hybrid repo manifest if W&B sweeps cannot represent population/promotion needs cleanly.
- Cross-accelerator calibration if ordered fallback leaves performance uncertainty.
