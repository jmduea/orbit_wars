# Deep Interview Spec: Output Standardization

## Goal

Standardize Orbit Wars training, sweep, evaluation, replay, Docker validation, W&B, and artifact-worker outputs under one coherent output system that is easy to browse, bounded in disk growth, and traceable without deep forensic dives.

The canonical concept is a campaign: a named experimental question or comparison frame. A campaign can be a formal Hydra/W&B sweep, an ad hoc batch of related runs, a baseline stabilization effort, a capacity comparison, or a submission-candidate evaluation. It may contain one run or many runs, and may compare multiple model compatibility families when that is part of the question.

## Current Context

- Hydra currently creates default date/time directories under `outputs/YYYY-MM-DD/HH-MM-SS`.
- `src/run_paths.py` detects `hydra.runtime.output_dir` and places checkpoints under that run output directory's `checkpoints/` subtree.
- `conf/artifacts/default.yaml` still defines `artifacts.save_dir`, `artifact_pipeline.queue_dir`, and `replay.output_dir`, but `artifacts.save_dir` is effectively bypassed for Hydra runs.
- `artifact_jobs/` currently mixes queued job JSON files with Docker validation output folders named like `docker_u000100_<job_id>/`, making provenance hard to inspect from the folder tree alone.
- W&B local generated files currently live in top-level `wandb/` by default unless directed elsewhere.
- The current local footprint is approximately `1.5G` in `outputs/`, `52M` in `wandb/`, and almost nothing in top-level `artifacts/`; the main growth pressure is run-local outputs/checkpoints/replays/Docker products.

## Research Findings

- Hydra is intended to create a per-run output directory and store `.hydra/config.yaml`, `.hydra/hydra.yaml`, and `.hydra/overrides.yaml` there. `hydra.run.dir`, `hydra.sweep.dir`, and `hydra.sweep.subdir` are the correct controls for customizing physical layout.
- W&B local generated files can be redirected with `WANDB_DIR`; downloaded artifacts and upload staging are controlled separately via `WANDB_ARTIFACT_DIR` and `WANDB_DATA_DIR`.
- W&B run groups and job types are suitable for campaign/job categorization; run IDs should be treated as durable identities, while human-readable names are not unique.
- Common experiment-tracking practice favors immutable per-execution directories plus manifests/indexes over moving or renaming artifacts after the fact.
- Bulky intermediate artifacts should use retention classes. Promoted outputs, best checkpoints, submission packages, and reproducibility metadata should survive; dense replay dumps and transient Docker validation folders should not accumulate by default.

## Target Shape

Use one top-level output root, likely `outputs/`, with campaign-centric organization for user-facing experiment work and explicit manifests for every run or derived job.

Candidate layout:

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
          wandb/
      evaluations/
        <evaluation_id>/
          manifest.json
          summaries/
          packages/
      promoted/
        current_best/
          manifest.json
          submission_package/
  indexes/
    runs.jsonl
    campaigns.jsonl
    promoted.jsonl
  cache/
    wandb-artifacts/
    wandb-data/
```

The exact path names may change during planning, but the structural rules should remain:

- Every physical execution directory has a manifest.
- Campaigns are the primary human browsing surface.
- Model compatibility family is stored as manifest metadata, W&B tags/config, and optionally registry lineage, rather than being the default physical root.
- Physical run directories are immutable once created.
- Campaign-level evaluation outputs summarize and promote results across runs.
- W&B generated files for a run should live under that run envelope when practical; W&B caches/downloaded artifacts should live under a controlled cache area, not top-level `wandb/`.

## Retention Policy

Default retention should keep compact essentials for every run:

- resolved Hydra config and overrides
- manifest metadata
- metrics/log JSONL
- latest checkpoint and/or best checkpoint according to configured metric
- enough provenance to reproduce replay/evaluation/package generation

Bulky intermediate products should be pruned or treated as short-lived unless promoted:

- intermediate checkpoints beyond retention policy
- dense replay dumps
- raw Docker validation work directories
- transient worker logs beyond failure diagnostics

Campaign-level promoted outputs should keep the current best run with full reproducibility metadata and an easy-to-submit competition package or reusable evaluation package.

## Acceptance Criteria

1. New Hydra training runs write under the unified output root with a campaign/run structure rather than the default date/time-only layout.
2. New runs include a manifest that records run ID, campaign, model compatibility family, W&B run ID, command/overrides, git identity, resolved config paths, output paths, and retention class.
3. W&B local generated files no longer default to top-level `wandb/`; they are run-local or explicitly configured under the unified output root/cache.
4. `artifacts/`, `outputs/`, and `wandb/` fragmentation is removed for new runs. Top-level `artifacts/` should not be a second canonical output location.
5. Replay and Docker validation outputs no longer appear as untraceable sibling folders in `artifact_jobs/`; queue state and result artifacts are separated or represented by job/evaluation manifests.
6. Campaign-level evaluation/promoted outputs keep the current best package and the metadata needed to reproduce it.
7. Default retention prevents unbounded disk growth while preserving compact run provenance.
8. Existing local data can be left in place for a fresh start, but an optional read-only index or cleanup/migration helper is acceptable if low-risk.
9. Checkpoint compatibility and resume behavior remain safe for existing checkpoints.
10. Tests cover path resolution, config validation, manifest creation, retention behavior, and artifact job output layout.

## Non-Goals

- Do not require full migration of all existing `outputs/`, `artifacts/`, and `wandb/` data as a prerequisite.
- Do not make model compatibility family the only physical browsing dimension.
- Do not keep all intermediate Docker/replay artifacts indefinitely by default.
- Do not disable Hydra `.hydra/` snapshots just to reduce visual clutter.
- Do not rely on W&B run names as durable identities.

## Assumptions Exposed And Resolved

- Initial leaning was model-family-first storage because model architecture affects checkpoint compatibility.
- Research and follow-up clarified that campaigns are the primary experimental question/comparison frame, while model family can be metadata and registry lineage.
- Existing data can be treated as legacy unless a feasible low-risk indexing or cleanup script is planned.
- The core issue with `artifact_jobs/` is not only folder naming; it is the lack of an obvious result model tying job, source checkpoint, produced files, and retention class together.

## Ontology

- Output root: one canonical local root for generated run/campaign/evaluation outputs.
- Campaign: named experimental question or comparison frame.
- Run: immutable training execution with Hydra config snapshot and manifest.
- Evaluation job: derived execution that evaluates, validates, replays, or packages a source run/checkpoint.
- Promoted output: campaign-level current best package/checkpoint/replay bundle intended for submission or further evaluation.
- Model compatibility family: model/checkpoint shape lineage such as attention or GNN pointer; important metadata but not necessarily the physical root.
- Retention class: policy label controlling whether outputs are kept, compacted, cached, or pruned.

## Interview Transcript

1. Asked top-level grouping. Answer: leaning toward model compatibility family but saw value in experiment/sweep campaign.
2. Asked where mixed-family sweeps should live. Answer: split by model family, linked by campaign manifest.
3. Asked migration scope. Answer: fresh start acceptable unless full migration is feasible; disk growth is a major motivation.
4. Asked retention policy. Answer: keep compact essentials for every run.
5. Asked replay/Docker output model. Answer: campaign-level evaluation outputs are appropriate, with current best retained and packaged for competition/subsequent evals.
6. Challenged model-family physical root after research. Answer: needed more explanation of campaign definition.
7. Defined campaign as named experimental question/comparison frame. Answer: accepted.

## Resolved Planning Defaults

- Use a Hydra-first campaign layout where Hydra's runtime output directory is the immutable run envelope.
- Default ad hoc campaign: `scratch`.
- Default run ID format: timestamp, seed, Hydra job number when present, and a short random suffix.
- Default current-best metric: `episode_reward_mean`.
- Default W&B artifact policy: local-first; remote W&B Artifact/Registry promotion is deferred.
- Existing local output data remains in place as legacy data; no full migration is required for the first implementation.
- Persisted execution plan: `.omg/plans/output-standardization-ralplan.md`.
