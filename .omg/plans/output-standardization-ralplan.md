# RALPLAN: Output Standardization

## Decision

Implement a Hydra-first campaign output layout for new Orbit Wars runs.

Hydra's `runtime.output_dir` must be the immutable run envelope. The application should derive child paths from that envelope rather than creating a second nested checkpoint run directory.

Campaigns are the primary human browsing surface. A campaign is a named experimental question or comparison frame, not only a formal W&B or Hydra sweep. Model compatibility family is durable metadata and index data, not the default physical root.

## Accepted Defaults

- Ad hoc campaign: `scratch`
- Run ID format: timestamp + seed + Hydra job number when present + short random suffix
- Current-best metric: `episode_reward_mean`
- W&B artifact mirroring: local-first; remote artifact/registry promotion can be added later
- Legacy data: leave existing `outputs/`, `artifacts/`, and `wandb/` data in place

## Principles

1. Hydra owns the physical run envelope and `.hydra/` stays inside that envelope.
2. Every run and derived job has a manifest with provenance and retention metadata.
3. Campaigns organize experiment questions; model family remains queryable metadata.
4. Queue state and result artifacts are separate concepts with separate directories.
5. Retention keeps compact essentials by default and preserves promoted outputs explicitly.

## Decision Drivers

1. Avoid breaking checkpoint save/load and explicit `resume_checkpoint` behavior.
2. Remove fragmentation across top-level `outputs/`, `artifacts/`, and `wandb/` for new runs.
3. Make replay, Docker validation, and packaging outputs traceable without inspecting opaque job folders.
4. Keep config ownership schema-first and Hydra-compatible.
5. Add enough smoke tests to prove real output layout, not just unit path composition.

## Architecture

### Path Contract

Target physical shape:

```text
outputs/
  campaigns/
    <campaign_slug>/
      campaign_manifest.json
      runs/
        <run_id>/
          .hydra/
          manifest.json
          logs/
          checkpoints/
          queue/
            optional_jobs/
          evaluations/
            <job_id>/
              manifest.json
              replay/
              docker_validation/
              logs/
          cache/
            wandb/
      promoted/
        current_best/
          manifest.json
          checkpoint/
          submission_package/
  indexes/
    runs.jsonl
    campaigns.jsonl
    promoted.jsonl
  cache/
    wandb-artifacts/
    wandb-data/
```

Single-run Hydra directory:

```text
outputs/campaigns/${output.campaign}/runs/${output.run_id}
```

Multirun directory:

```text
outputs/campaigns/${output.campaign}/runs/${output.run_id}
```

Hydra job numbers may be included in `output.run_id`, but the final run envelope must be the same place where Hydra writes `.hydra/`.

### Pre-Hydra Run Identity

Run ID must exist before Hydra creates `runtime.output_dir`.

Implementation must not generate the canonical run ID only inside `resolve_run_paths()`, because that is too late. Use a pre-Hydra resolver/bootstrap mechanism or Hydra-compatible resolver registration so `output.run_id` is available to `hydra.run.dir`, `hydra.sweep.dir`, and `hydra.sweep.subdir`.

The run ID should be stable for one process invocation and collision-resistant enough for multiruns.

### Run Context

Replace tuple-style path return values with a `RunContext` dataclass rooted at `hydra.runtime.output_dir` when Hydra is initialized.

Required fields:

- `run_id`
- `campaign_slug`
- `run_dir`
- `manifest_path`
- `campaign_dir`
- `campaign_manifest_path`
- `logs_dir`
- `log_path`
- `checkpoints_dir`
- `queue_dir`
- `evaluations_dir`
- `wandb_dir`
- `wandb_artifact_dir`
- `wandb_data_dir`
- `indexes_dir`
- `retention_class`
- `model_compatibility_family`

For Hydra runs, `run_context.run_dir == Path(HydraConfig.get().runtime.output_dir)`. Child paths are derived from it.

For non-Hydra fallback, use the same shape under the configured output root so tests and direct calls remain deterministic.

### Manifest Schema

Run manifests should be written atomically and include at least:

- run ID
- campaign slug
- run name
- job type
- model compatibility family
- seed
- W&B project/entity/group/tags/run ID when available
- Hydra output directory
- resolved config path
- Hydra overrides path
- command or CLI override summary when available
- git commit and dirty flag/hash when available
- output paths
- produced artifacts
- retention class
- parent/source run or checkpoint for derived jobs
- created timestamp

Campaign manifests and JSONL indexes should be append/update safe and tolerate repeated runs.

### W&B Routing

Set W&B local paths before `wandb.init()`:

- generated run files: run-local `cache/wandb/` or equivalent `wandb_dir`
- downloaded artifacts: unified output cache `outputs/cache/wandb-artifacts/`
- upload staging data: unified output cache `outputs/cache/wandb-data/`

Use local-first W&B artifact policy in this pass. Preserve project/entity/group/tags and add campaign/run/model metadata.

### Artifact Jobs And Evaluations

Queue state remains in `queue/optional_jobs/`.

Replay, Docker validation, and packaging outputs must not be siblings of queue JSON files. Job JSON should record the result directory and result manifest path. Result artifacts should live under either:

- run-local `runs/<run_id>/evaluations/<job_id>/` for jobs derived from one run/checkpoint
- campaign-level `campaigns/<campaign>/evaluations/<evaluation_id>/` for cross-run evaluations

Docker validation should write logs, replay HTML files, and package outputs under the result directory. Raw bulky outputs should receive a retention class and can be pruned unless promoted.

### Retention And Promotion

Keep compact essentials for every run:

- `.hydra/` config snapshots
- run manifest
- metrics/events logs
- latest and/or best checkpoint using `episode_reward_mean` initially
- enough metadata to reproduce replay/evaluation/package generation

Promoted campaign output should track the current best candidate locally under `promoted/current_best/` with a manifest and competition package metadata. Remote W&B artifact/registry promotion is deferred.

## Implementation Order

1. Add schema/config fields for output root, campaign slug, run ID, retention class, W&B cache paths, queue path, evaluation path, and promotion policy.
2. Register the pre-Hydra run ID mechanism and configure Hydra run/sweep templates so Hydra creates the final run envelope directly.
3. Introduce `RunContext` in `src/run_paths.py` and update path resolution so Hydra runs use `hydra.runtime.output_dir` as `run_dir`.
4. Add atomic manifest and index write helpers.
5. Integrate `RunContext` into `src/jax_train.py` for checkpoints, logs, queue paths, manifests, retention, and resume-safe behavior.
6. Route W&B local generated files and caches before `wandb.init()` in telemetry setup.
7. Split optional job queue state from result output paths in `src/artifact_pipeline.py`, `src/jax_train.py`, and `scripts/run_artifact_worker.py`.
8. Add local promoted current-best metadata and package path support.
9. Update docs to describe the new output contract and mark old top-level output folders as legacy for existing data.
10. Add and run focused tests, then Hydra smoke tests.

## Compatibility Rules

- Do not move existing local run directories during implementation.
- Do not require old checkpoints to contain new manifest/output fields.
- Explicit `resume_checkpoint=/path/to/old_checkpoint.pkl` must remain valid if the checkpoint is otherwise compatible.
- Keep old CLI defaults only where needed for manual compatibility, but training/worker code must pass canonical output paths.
- Do not reintroduce top-level `artifacts/` as a canonical training output root for new Hydra runs.

## Test Plan

- Config/schema tests for output fields, default campaign `scratch`, invalid slugs, relative path validation, W&B path settings, and no legacy flat aliases.
- `RunContext` tests for Hydra and non-Hydra path resolution.
- Manifest tests for atomic writes and required run/campaign/job fields.
- Telemetry tests proving W&B local path routing happens before fake `wandb.init()`.
- Artifact pipeline and worker tests proving queue JSON and Docker/replay result artifacts are separated.
- Checkpoint retention tests proving protected/latest/best checkpoint behavior remains intact.
- Resume compatibility test using an old explicit checkpoint path.
- Hydra smoke test: `uv run python -m src.train print_resolved_config=true` shows the new output fields and Hydra directory template.
- Hydra training smoke test: one-update training creates `.hydra/`, `manifest.json`, `logs/`, and `checkpoints/` in the same run envelope.
- Hydra multirun smoke test with two seeds creates distinct envelopes under `outputs/campaigns/scratch/runs/`.

## Verification Commands

```bash
rtk uv run --group dev pytest tests/test_config_consolidation.py tests/test_artifact_pipeline.py tests/test_kaggle_submission_packager.py tests/test_telemetry.py
rtk uv run python -m src.train print_resolved_config=true
rtk uv run python -m src.train training.total_updates=1 artifacts.checkpoint_every=1 telemetry.wandb.enabled=false
rtk uv run python -m src.train -m seed=1,2 training.total_updates=1 artifacts.checkpoint_every=1 telemetry.wandb.enabled=false
```

## ADR

Decision: Hydra-first campaign layout with immutable run envelopes and manifests.

Drivers: Preserve Hydra snapshots, reduce output fragmentation, improve artifact provenance, bound disk growth, and keep resume compatibility safe.

Alternatives considered:

- Model-family physical root plus campaign indexes: good for checkpoint compatibility browsing, weaker for experimental questions and mixed-family comparisons.
- App-owned layout independent of Hydra: more direct Python control, but fights Hydra and risks duplicate output roots.
- Manifest-only over current layout: fast, but insufficient for the fragmentation and retention goals.

Why chosen: It aligns with Hydra and W&B best practices, the accepted campaign definition, and the desired compact/provenance-first retention policy.

Consequences: Requires careful pre-Hydra identity setup and real Hydra smoke tests, but avoids building a parallel output system.
