# Plan: Kaggle W&B MVP Hardening

**Source plan:** `.omg/plans/ralplan-kaggle-wandb-population.md`  
**Source spec:** `.omg/specs/deep-interview-kaggle-wandb-population.md`  
**Slug:** `kaggle-wandb-mvp-hardening`  
**Workflow:** ralplan  
**Status:** planned (consensus iter-1)  
**Date:** 2026-05-26

---

## Executive Summary

Finish the Kaggle/W&B Population MVP by hardening the current implementation rather than expanding scope. The target is not tournament readiness; it is a trustworthy launch path for W&B-native Kaggle workers:

1. local command can create/resolve a W&B sweep and package a Kaggle worker,
2. Kaggle worker can bootstrap dependencies, verify GPU-backed JAX, calibrate throughput, and run one W&B-assigned candidate,
3. checkpoints land as W&B artifacts with usable aliases/metadata,
4. status/sync/shortlist commands are operational,
5. live validation is a tiny manual smoke, not a broad local test run.

Because background training runs are active and broad tests can crash WSL2, verification for this hardening phase must avoid `make test-fast`, full pytest, JAX rollout/training suites, and any routine `make test-jax`.

---

## RALPLAN-DR Summary

### Principles

1. **Do not compete with training runs** — prefer inspection, py-compile, and dry-run commands over broad tests.
2. **Harden only MVP-critical seams** — auth, packaging, W&B sweep/run continuity, calibration, artifact handoff, and operator commands.
3. **Live smoke beats local over-testing** — the real acceptance is one tiny Kaggle GPU worker, not a heavy local suite.
4. **W&B remains canonical** — avoid introducing a repo population manifest during hardening.
5. **Fail loudly before spending GPU budget** — missing auth, CPU JAX, bad sweep ID, or package defects should stop early.

### Decision Drivers

1. The current implementation has the right module shape but still needs operational guardrails.
2. Worker calibration now exists, but it needs safe budgets, readable summaries, and failure behavior.
3. Checkpoint artifact logging now has aliases, but continuation discovery should be made explicit.
4. Kaggle and W&B auth failures should be preflighted locally before launching a kernel.
5. Verification must be sparse because the host is running training and broad tests are unsafe.

---

## Viable Options Considered

| Option | Description | Pros | Cons | Verdict |
|---|---|---|---|---|
| Live-first hardening | Immediately attempt a Kaggle smoke and patch issues reactively. | Fastest evidence; finds real API/runtime gaps. | Risk of wasting GPU time on obvious local/auth defects. | Rejected as first step |
| Offline-first hardening | Add preflight/dry-run guardrails, summaries, and continuation helpers before live smoke. | Minimizes wasted GPU time; matches test-sparing constraint. | Slightly slower to real evidence. | **Chosen** |
| Full local QA first | Run broad suites before live work. | Strong local confidence. | Conflicts with WSL2/training-run constraint. | Rejected |

---

## ADR

### Decision

Finish the MVP with an offline-first hardening pass, followed by a tiny live Kaggle smoke. The implementation should remain W&B-sweep-native and should not introduce a custom population manifest.

### Why

The user wants the Kaggle work finished but also explicitly warned that broad tests can crash WSL2 while background training runs are active. The responsible path is to harden operator-facing seams with cheap checks, then use the smallest possible live Kaggle run as the acceptance proof.

### Consequences

- No broad local test gate is required for this phase.
- `py_compile`, dry-run command rendering, and file inspection are acceptable verification.
- Live Kaggle/W&B smoke becomes the primary final proof.
- Any discovered need for a repo manifest remains a follow-up, not part of this hardening plan.

---

## Remaining Work

### H1 — Local Preflight Command

Add `scripts/kaggle_wandb_population.py preflight`.

It should check:

- Kaggle CLI exists.
- `kaggle kernels list --mine` or another low-cost auth check succeeds.
- W&B import succeeds.
- W&B auth/API can resolve the configured project/entity or report a clear auth error.
- sweep YAML exists and parses.
- kernel ID is not the placeholder `replace-me/...`.
- package render location is writable.
- optional: warn if `KAGGLE_USERNAME` is missing.

Preflight must not launch kernels or run training.

### H2 — Worker Package Completeness

Harden `render_kernel_package()` so the package includes every file the worker needs:

- `src/`
- `conf/`
- `scripts/kaggle_worker_entry.py`
- `scripts/benchmark_jax_rl.py`
- `pyproject.toml`
- `uv.lock`
- README/run metadata if useful

It should exclude:

- `outputs/`
- `wandb/`
- `.venv/`
- `__pycache__/`
- `.git/`
- `.omg/`
- `.understand-anything/`

Add a package summary JSON listing included top-level files and generated env values, excluding secrets.

### H3 — Worker Safety And Calibration Controls

Make worker calibration explicitly bounded and operator-visible:

- Add env/config knobs for calibration warmup, updates, max variants, and timeout.
- Add default short calibration budget.
- Add per-variant timeout handling.
- Always write `worker-summary.json` with diagnostics, calibration results, selected overrides, final command, and exit code.
- If all calibration variants fail, stop by default unless `ORBIT_WARS_KAGGLE_ALLOW_CALIBRATION_FALLBACK=1`.
- Keep CPU fallback disabled for real training.

### H4 — W&B Run And Artifact Continuity

Make artifact handoff inspectable:

- Ensure worker training subprocess attaches to the W&B agent run via `WANDB_RUN_ID` and `WANDB_RESUME=allow`.
- Ensure checkpoint artifacts include update/run metadata and `latest` alias.
- Add a shortlist field for latest checkpoint artifact name/version/aliases when discoverable.
- Add a helper command or documented query for "latest checkpoint for sweep/candidate".

### H5 — Launcher Status And Ledger

Keep local state diagnostic-only:

- Write `outputs/kaggle_population/launches.jsonl` for each launch attempt.
- Include timestamp, package dir, kernel ID, accelerator attempted, return code, stdout/stderr tail, sweep ID.
- `status` should print normalized state and raw Kaggle response.
- `sync-output` should record downloaded path in the ledger.

This ledger must not become candidate assignment truth.

### H6 — Tiny Live Smoke Runbook

Update docs with a minimal live sequence:

1. `preflight`
2. `launch --dry-run`
3. `launch --create-sweep --accelerator <one cheap/preferred GPU>`
4. `status`
5. `sync-output`
6. `shortlist`

Include a "stop conditions" section:

- no GPU JAX,
- dependency sync fails,
- calibration all variants fail,
- no checkpoint artifact,
- W&B agent run not visible.

---

## Implementation Order

1. Add `preflight` and package completeness fixes.
2. Add worker summary/failure controls around calibration.
3. Add launch ledger and status/sync ledger updates.
4. Improve shortlist artifact details.
5. Update docs with live smoke steps and stop conditions.
6. Run only syntax/dry-run verification locally.
7. Optionally perform one tiny live Kaggle smoke when the user is ready.

---

## Critic Review

Status: **approved with constraints**.

Required constraints:

- Do not run broad pytest, `make test-fast`, `make test`, or `make test-jax` during hardening while training jobs are active.
- Do not perform live Kaggle launch without explicit user readiness if credentials/quota risk is unclear.
- Do not introduce a new population manifest.
- Do not make Kaggle outputs canonical.
- Do not proceed to tournament architecture in this phase.

Residual risks:

- Kaggle image Python/CUDA/JAX compatibility may still fail live.
- W&B sweep agent behavior inside Kaggle may need one reactive patch after smoke.
- Kaggle accelerator IDs may be accepted by CLI but provision differently at runtime.

---

## Minimal Verification Plan

Allowed local checks:

- `uv run python -m py_compile` on touched Python files.
- `launch --dry-run` with a `/tmp` package directory.
- `preflight` only if it does not launch kernels or train.
- Manual inspection of generated package contents.

Avoid until user explicitly approves:

- `make test-fast`
- `make test`
- `make test-jax`
- broad `pytest`
- local JAX rollout/training smoke

Manual live acceptance, when approved:

- One Kaggle worker reaches GPU-backed JAX diagnostics.
- One W&B sweep run appears.
- Calibration results appear in W&B and `worker-summary.json`.
- Training starts with selected overrides.
- At least one checkpoint artifact appears with `latest` alias.
- `shortlist` emits that candidate.

---

## Completion Criteria

The hardening phase is complete when:

- `preflight` catches missing local prerequisites.
- dry-run launch renders a valid Kaggle command and package.
- worker package includes all required repo files.
- worker summary captures diagnostics/calibration/final command/exit code.
- calibration failure behavior is explicit and bounded.
- W&B checkpoint artifact continuity is discoverable.
- launcher ledger records launch attempts and syncs.
- docs describe the tiny live smoke path.
- no broad tests were run as part of hardening.
