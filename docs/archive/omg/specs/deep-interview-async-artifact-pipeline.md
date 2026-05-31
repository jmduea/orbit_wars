# Deep Interview Spec: Async Artifact Pipeline

## Goal

Decouple training progress from blocking artifact work by introducing an async artifact pipeline for checkpointing, checkpoint retention, telemetry artifact logging, replay generation, and Kaggle Docker validation/replay inspection.

The training loop should stop performing heavy checkpoint/replay work inline during update checkpoints. Instead, it should enqueue or dispatch artifact work so PPO training can continue, while still guaranteeing that normal shutdown flushes pending work and leaves a usable latest checkpoint.

## Current Blocking Path

Today `src/jax_train.py` performs all checkpoint-period work inline:

1. `save_jax_checkpoint(...)` serializes `jax_ckpt_last.pkl` and `jax_ckpt_XXXXXX.pkl` synchronously.
2. `prune_checkpoints(...)` runs immediately afterward.
3. `telemetry.log_checkpoint(...)` runs in the same branch.
4. `maybe_write_jax_checkpoint_replay(...)` generates replay episodes and HTML/JSON metadata synchronously.
5. Replay metadata is logged to telemetry synchronously when present.

The new Kaggle Docker validator already provides a separate package/validation path that can run checkpoint-derived validation and bounded self-play outside the training loop.

## Required Behavior

- Checkpoint persistence must become async/background from the training loop's perspective.
- Replay generation must stop blocking PPO training updates.
- Docker validation/replay generation should be handled by a separate optional worker process rather than directly by the trainer.
- Normal training shutdown must flush all pending artifact jobs before exit.
- The latest checkpoint must not be lost; resume safety has priority over throughput.
- If artifact work falls behind, stale replay/inspection jobs may be skipped or coalesced, but latest checkpoint safety must be preserved.
- Queue/storage design is intentionally deferred to ralplan. Candidate designs include filesystem job files plus JSONL events, SQLite, or a hybrid in-process plus durable ledger.

## Non-Goals

- Do not make Docker validation mandatory for every checkpoint during active training.
- Do not block training on replay rendering or Docker validation except during explicit final flush/shutdown behavior.
- Do not require a distributed job system or external service unless ralplan proves local filesystem/process coordination is inadequate.
- Do not compromise checkpoint compatibility or resume behavior to gain async throughput.

## Acceptance Criteria

1. Training checkpoint intervals no longer block on full checkpoint serialization, retention, replay generation, or Docker validation in the hot update path.
2. A normal training exit waits for pending required artifact work and reports failures clearly.
3. The latest checkpoint is guaranteed to exist in a resume-safe state after checkpoint flush completes.
4. Replay and Docker validation work can be skipped, coalesced, or retried when stale or resource constrained without corrupting checkpoint state.
5. Artifact work has visible status/logging so users can tell whether checkpoint, retention, replay, telemetry, and Docker validation jobs succeeded, failed, skipped, or are pending.
6. The optional worker can consume checkpoint-derived jobs independently of the training process.
7. Existing replay cadence semantics, such as `replay.every_n_checkpoints`, remain configurable or have a documented migration path.
8. Tests cover async flush behavior, failure reporting, latest-checkpoint integrity, replay-job skipping/coalescing, and worker/job handoff semantics.

## Assumptions Exposed And Resolved

- Assumption: only checkpoint serialization needs async behavior. Resolved: target a full artifact pipeline, including retention, telemetry artifacts, replay generation, and validation handoff.
- Assumption: background work may be best-effort. Resolved: normal shutdown should flush all pending jobs, with latest checkpoint protected above all else.
- Assumption: every replay/validation job must be preserved. Resolved: latest checkpoint must be preserved; stale replays and non-essential validation work may be dropped under pressure.
- Assumption: Docker validation should run inside the active training process. Resolved: Docker validation should be consumed by a separate optional worker process.
- Assumption: queue durability should be decided in the interview. Resolved: defer queue storage design to ralplan consensus.

## Ontology

- Artifact Job: A unit of work derived from a training update, such as checkpoint write, retention pass, replay render, telemetry artifact log, or Docker validation.
- Required Job: Work that must complete for resume safety, primarily latest checkpoint persistence.
- Optional Job: Work useful for inspection/validation, such as replay rendering or Docker smoke validation, that can be skipped/coalesced if stale.
- Artifact Worker: A background in-process worker or separate process that consumes artifact jobs.
- Durable Queue: The mechanism by which training records jobs for in-process or external workers; storage choice is deferred to ralplan.
- Final Flush: The shutdown phase where training waits for required pending work and reports optional-work status.

## Interview Transcript

1. Asked for the primary output. User selected a full async artifact pipeline.
2. Asked how training should handle pending jobs at shutdown/failure. User selected flushing all pending jobs before exit.
3. Asked about backpressure. User selected always keeping latest checkpoint and dropping stale replays.
4. Challenged Docker validation during active training. User selected a separate optional worker process.
5. Asked about queue durability. User was open to suggestions, with checkpoint durability as the non-negotiable and replay/validation skipping acceptable under constraints.
6. Proposed filesystem jobs plus JSONL events. User chose to defer the queue choice to ralplan.

## Ambiguity Score

- Initial: 100%
- After codebase context: 72%
- After full-pipeline target: 55%
- After shutdown/flush policy: 42%
- After backpressure policy: 32%
- After Docker worker separation: 24%
- After queue-decision deferral: 18%

## Recommended Next Step

Run ralplan consensus on the queue/worker architecture, especially the boundary between in-process async checkpointing and external replay/Docker validation workers. The key decision is how to represent durable artifact jobs while preserving checkpoint integrity and keeping the trainer implementation small.
