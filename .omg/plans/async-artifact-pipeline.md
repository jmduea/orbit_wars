# Ralplan: Async Artifact Pipeline

## Decision

Implement a trainer-owned async artifact pipeline with latest/final checkpoint priority.

Checkpoint `device_get` and pickle work moves to a bounded in-process worker. Replay and Docker validation become optional committed-checkpoint jobs that can be consumed by a separate worker. Telemetry remains single-owner in the trainer lane.

## Decision Drivers

- Training should not block on every intermediate checkpoint, replay render, Docker validation, or W&B artifact upload.
- `jax_ckpt_last.pkl` and the final checkpoint are correctness artifacts and must be resume-safe.
- Intermediate numbered checkpoints may be skipped under queue pressure, but only visibly and never for final/latest safety.
- Retention must not delete checkpoints referenced by pending/running replay or validation work.
- Worker failures must be drained and surfaced by the trainer, never silently lost.

## Core Invariants

### Queue And Backpressure

- Checkpoint jobs live in a bounded queue, defaulting to a small size such as `1`.
- Queue overflow may drop or coalesce only non-final intermediate numbered checkpoints.
- Latest and final checkpoint requests are never dropped.
- At most one pending latest payload exists; a newer latest payload may replace an older pending latest payload before commit.
- Final checkpoint admission bypasses normal intermediate skipping: the trainer drops/coalesces older intermediate work if needed, submits or performs the final save, waits for commit, and only then claims successful shutdown.
- Every skipped checkpoint/replay/validation job emits a status event with a reason such as `queue_pressure`, `shutdown_timeout`, or `worker_failed`.

### Worker Result Propagation

- Every worker item must complete as exactly one result: `committed`, `failed`, `skipped`, or `coalesced`.
- Checkpoint worker results flow through a trainer-drained `CheckpointResult` channel.
- The trainer drains results after enqueue attempts, periodically during training, before retention, before telemetry finish, and during shutdown.
- Latest/final checkpoint failures raise or fail the run according to config, with `fail_training_on_checkpoint_error=true` by default.
- Any worker exception marks the artifact pipeline unhealthy. Once unhealthy, the trainer stops accepting intermediate checkpoint jobs and uses fail-fast or synchronous final/latest handling according to config.
- Training must not report successful completion until the final checkpoint committed result has been observed.

### Atomic Checkpoint Commit

- Write temp files in the target directory.
- Flush and `os.fsync()` each checkpoint file.
- Atomically replace the final path.
- `fsync()` the containing directory after each replace.
- For combined update jobs, commit the numbered checkpoint first, then `jax_ckpt_last.pkl`.
- A checkpoint job is `committed` only after file fsync, atomic replace, directory fsync, and successful result publication.
- Ledger/event append happens after commit. If ledger append fails, the checkpoint remains committed but the trainer must surface the ledger failure.
- If numbered commit succeeds but latest commit fails, latest is considered stale; the trainer reports latest lag/failure and treats latest/final failure according to config.

### Retention Protection

- Retention runs only after checkpoint commit results have been drained.
- Retention accepts explicit protected paths and active worker leases.
- Protected paths include `jax_ckpt_last.pkl`, final checkpoint, pending queue targets, in-progress temp/final paths, replay/Docker checkpoint dependencies, and configured keep-policy paths.
- Retention never unlinks temp files with active leases, symlinks, out-of-run-dir files, latest, final, or pending/running dependency paths.
- Replay workers select earlier checkpoints from committed files only. If an earlier checkpoint is missing due to retention, replay records fallback/skip rather than using a partial path.

### Telemetry Ownership

- Worker threads/processes never call W&B directly.
- Workers emit structured artifact events into a trainer-owned drain lane and durable ledger where enabled.
- Only the trainer lane calls `TelemetryLogger` methods.
- Trainer drains artifact events periodically, after checkpoint enqueue/result drain, after retention, after replay/Docker status updates, and before `telemetry.finish()`.
- Worker failure events are visible even when W&B is disabled.

### Shutdown Policy

- Normal completion: enqueue or synchronously perform final checkpoint, wait for final/latest commit up to timeout, drain all worker results/events, finish telemetry, then return.
- `KeyboardInterrupt`: request best-effort latest checkpoint for the last completed update, wait up to interrupt timeout, drain/report events, finish telemetry, then preserve interrupted exit semantics.
- Training exception: request best-effort latest checkpoint if state is available, wait up to exception timeout, drain/report events, finish telemetry, then re-raise the original exception. Worker exceptions must be attached/reported without masking the original failure.
- Worker exception: mark pipeline unhealthy, stop accepting intermediate jobs, report error, and fail shutdown if final checkpoint cannot be preserved.
- Timeout: preserve already committed latest/final state, emit timeout event, and do not claim success for any uncommitted final checkpoint.

## Architecture

- `src/artifact_pipeline.py`: checkpoint job/result types, bounded queue admission, coalescing/skipping policy, worker lifecycle, atomic writer, ledger/event helpers.
- `src/jax_train.py`: enqueue checkpoint jobs, drain results/events, run retention after committed results, schedule replay/Docker jobs, and own final flush in `try/finally`.
- `src/checkpoint_retention.py`: accept protected paths/updates so pending/running dependencies cannot be pruned.
- `src/telemetry.py`: remain trainer-owned; optionally add small helpers for structured artifact events.
- `scripts/run_artifact_worker.py`: optional external worker for replay/Docker validation jobs from committed checkpoint paths.
- `scripts/validate_kaggle_docker_submission.py`: remains the Docker validation implementation used by external jobs.

## Config Additions

Add `ArtifactPipelineConfig` to `TrainConfig` and regenerate `default_cfg.yaml`.

Suggested fields:

```yaml
artifact_pipeline:
  enabled: true
  checkpoint_queue_size: 1
  checkpoint_timeout_seconds: 300.0
  final_flush_timeout_seconds: 900.0
  interrupt_flush_timeout_seconds: 60.0
  exception_flush_timeout_seconds: 60.0
  latest_lag_warning_updates: 1
  coalesce_intermediate_checkpoints: true
  replay_async: true
  docker_validation_async: false
  ledger_enabled: true
  queue_dir: artifact_jobs
  fail_training_on_checkpoint_error: true
  fail_training_on_optional_job_error: false
```

Validation rules:

- Queue sizes and timeouts must be positive.
- Final/latest checkpoints cannot be coalesced or skipped.
- Docker validation jobs require committed checkpoint paths and unique per-job output directories.
- If the durable ledger is disabled, optional replay/Docker crash recovery is explicitly disabled and logged.

## Implementation Steps

1. Add artifact pipeline primitives and tests.
   - Bounded checkpoint queue.
   - Latest/final priority.
   - Intermediate skip/coalesce events.
   - `CheckpointResult` drain channel.
   - Pipeline unhealthy state and worker error propagation.

2. Split checkpoint saving into payload creation and atomic commit.
   - Preserve existing checkpoint payload schema for `load_jax_checkpoint` compatibility.
   - Worker performs `jax.device_get` and pickle.
   - Atomic writer implements temp file, file fsync, replace, directory fsync, numbered-before-latest.

3. Wire the pipeline into JAX training.
   - Replace inline checkpoint/retention/replay block with enqueue + result drain + retention scheduling + optional job scheduling.
   - Wrap training in `try/finally` for flush and telemetry finish.
   - Ensure final checkpoint commit is observed before successful completion.

4. Make retention dependency-aware.
   - Add protected paths/updates to `prune_checkpoints`.
   - Protect pending/running replay/Docker dependencies and final/latest paths.

5. Move replay and Docker validation to committed-checkpoint jobs.
   - Preserve `replay.every_n_checkpoints` eligibility semantics.
   - If a checkpoint is skipped under pressure, replay/validation for that checkpoint is skipped with a status event.
   - External worker consumes committed checkpoint paths and writes replay/validation artifacts/status.

6. Add config/schema/default updates.
   - Update `src/conf_schema.py`.
   - Regenerate `default_cfg.yaml`.
   - Add Hydra config group or base config entries as appropriate.
   - Add config validation for capacities/timeouts.

7. Add docs and focused verification.
   - Document async checkpoint safety, optional replay/Docker worker, skipped-intermediate semantics, and final flush behavior.

## Acceptance Criteria

- Hot training path no longer runs replay rendering or Docker validation inline.
- Intermediate checkpoint serialization can happen in the worker, while latest/final checkpoint durability is guaranteed at flush.
- Final checkpoint is never skipped, even when the queue is full.
- Worker failures are observed by the trainer and cannot silently degrade into missing artifacts.
- `jax_ckpt_last.pkl` and final numbered checkpoint are atomically committed and loadable after successful flush.
- Retention cannot delete checkpoints referenced by pending/running artifact jobs.
- Telemetry calls remain single-owner and happen only from the trainer lane.
- Replay/Docker jobs operate only on committed checkpoint paths.
- Existing checkpoint payloads remain load-compatible.
- Config defaults and generated default config remain in sync.

## Required Tests

- Queue overflow drops intermediate but not latest/final.
- Final checkpoint is admitted and committed when queue is full.
- Worker pickle/device_get failure propagates through `CheckpointResult` and fails latest/final flush.
- Atomic writer leaves no corrupt latest on injected failures.
- Ledger/result is published only after commit.
- Retention respects protected paths and active leases.
- Telemetry logs only committed artifacts from trainer lane.
- Replay is skipped when its checkpoint was skipped or missing, with a visible status event.
- Normal completion, `KeyboardInterrupt`, training exception, worker exception, and flush timeout shutdown paths.
- Config/default generation and invalid config validation.

## Verification Commands

```bash
rtk uv run --group dev pytest tests/test_artifact_pipeline.py tests/test_checkpoint_retention.py
rtk uv run --group dev pytest tests/test_jax_ppo.py tests/test_replay.py tests/test_kaggle_submission_packager.py
rtk uv run python scripts/generate_default_cfg.py --check
rtk uv run --group dev pytest
```

## Deferred Work

- SQLite job database is deferred until filesystem ledger/queryability becomes insufficient.
- Mandatory Docker validation during training is deferred; Docker validation remains an optional external worker job.
- Cross-process replay/Docker worker leasing can start simple, but must not weaken retention protection for claimed jobs.
