# Kaggle W&B Population Training

## Status

Approved.

## Goal

Build a W&B-first Kaggle orchestration workflow for Orbit Wars training. Kaggle provides hosted GPU worker time; W&B is the control plane for run identity, configs, metrics, and checkpoint artifacts; local repo tooling provides one-command launch, status tracking, output sync, and promotion shortlist generation.

The first milestone is a Population MVP that proves long-running Kaggle GPU jobs can be launched with enough confidence to justify later tournament architecture and self-play promotion plumbing.

## User Intent

The preferred operator experience is W&B-first orchestration with the same practical convenience as a one-command launcher. A local command should launch or refresh the Kaggle workers, but W&B should be the system of record for training runs, metrics, checkpoint artifacts, and candidate comparison.

The workflow should support two long-term modes:

- Continue-run mode: keep training one logical policy from the latest W&B checkpoint artifact.
- Population mode: train distinct candidate configs, model variants, and opponent mixes as separate W&B runs, then shortlist promising checkpoints for promotion and later tournaments.

Population mode is the first deliverable.

## MVP Scope

The Population MVP must:

- Provide one local command to start a population campaign.
- Push and launch one or more Kaggle kernels as W&B-tracked training workers.
- Give each worker a distinct candidate config/model/opponent setup.
- Request preferred Kaggle accelerators through the Kaggle CLI when selectable.
- Use ordered accelerator fallback if stronger accelerator classes are unavailable.
- Verify inside the kernel that JAX sees and uses the expected GPU backend.
- Detect observed hardware details and record them to W&B.
- Estimate ballpark training parameters from model/config shape and observed hardware.
- Run a bounded throughput calibration sweep before committing to longer runs.
- Launch longer candidate training only after calibration finds stable settings.
- Upload checkpoints to W&B artifacts as the canonical handoff path.
- Download/sync Kaggle outputs locally as diagnostic backup.
- Query W&B after worker completion and produce a promotion shortlist.

## Non-Goals For MVP

- Full tournament automation.
- Automatic self-play pool mutation.
- Perfect 30-hour quota optimization.
- Kaggle outputs as canonical checkpoint storage.
- Cross-accelerator-class calibration unless ordered fallback proves insufficient.
- Complete population-based training controller with automated mutation/evolution.

## Accelerator Policy

The launcher should prefer stronger Kaggle GPU accelerator classes when the API allows selecting them. The first implementation should use ordered fallback:

1. Try accelerator IDs in a configured preference order.
2. Launch on the first accepted accelerator.
3. Validate actual in-kernel hardware before spending meaningful training budget.
4. Stop or downgrade only according to explicit fallback policy.

The kernel must record at least:

- Requested accelerator ID.
- Observed GPU name.
- Available GPU memory when discoverable.
- JAX devices and backend.
- CUDA/JAX compatibility status.
- Calibration throughput and stability metrics.

CPU fallback is allowed only for tiny validation, never for real training budget.

## Throughput Calibration

The MVP should spend a small amount of GPU time finding stable high-throughput training parameters before launching long jobs. This is intentional: the system should avoid spending a 30-hour free-compute budget with poor GPU utilization, memory stalls, unstable batch shapes, or excessive host/device overhead.

Calibration should:

- Determine available accelerator and approximate memory budget.
- Estimate initial values for `training.num_envs`, `training.rollout_steps`, `training.minibatch_size`, `training.update_chunk_rows_min`, `training.update_chunk_rows_max`, and `training.rollout_microbatch_envs`.
- Run a bounded sweep around the estimate.
- Prefer settings that maximize stable `samples_per_sec` and `ppo_samples_per_sec` without OOM, recompilation churn, or obvious memory thrashing.
- Persist calibration results as W&B/config metadata for reuse by heavier model families, including high feature horizon and trajectory shield horizon variants.

## W&B Artifact Contract

W&B artifacts are canonical for checkpoint handoff.

Each worker should:

- Log config, candidate identity, accelerator request, observed hardware, calibration settings, and training metrics.
- Upload checkpoint artifacts at configured cadence.
- Mark the latest usable checkpoint clearly enough that a later worker can resume.
- Keep Kaggle output files available for local diagnostic sync, but not rely on them for normal continuation.

Local orchestration should:

- Resolve the latest/best checkpoint artifact for continue-run mode.
- Query candidate runs and checkpoint artifacts for population mode.
- Emit a local campaign record with Kaggle kernel IDs, W&B run IDs, status, and output-sync paths.

## Population Shortlist

For the MVP, promotion is a shortlist rather than a final tournament decision.

The shortlist should use W&B metrics as a cheap filter. Candidate runs should be ranked or flagged by metrics such as:

- `episode_reward_mean`.
- Win-rate or evaluation metrics already logged by training if available.
- `samples_per_sec` and `ppo_samples_per_sec` for throughput sanity.
- Checkpoint availability and training stability.

The full pipeline should later add:

- High win-rate gates against `scripted_nearest`.
- Candidate-vs-baseline and candidate-vs-current-best tournaments.
- Candidate-vs-candidate tournaments for promising models.
- Action-distribution summaries to identify behaviorally different candidates.
- Diversity-aware promotion for sufficiently strong but meaningfully different policies.
- Promotion into self-play opponent pools or future training populations.

## Acceptance Criteria

Population MVP is accepted when:

- A single local command can launch multiple Kaggle-backed W&B candidate workers.
- W&B shows separate candidate runs with distinct configs, metrics, hardware metadata, and checkpoint artifacts.
- The kernel verifies GPU-backed JAX before meaningful training.
- A bounded calibration sweep runs and records stable high-throughput parameters.
- Longer worker runs use calibration-selected settings.
- The local launcher can poll Kaggle status and sync Kaggle output artifacts for diagnostics.
- The local tooling can query W&B and produce a promotion shortlist from completed candidate runs.
- The result is trustworthy enough to justify implementing tournament and promotion plumbing next.

## Open Questions For Planning

- Exact CLI surface for campaign creation, population config definition, and status/sync commands.
- Whether to implement W&B sweep integration directly or use a custom W&B campaign abstraction with explicit candidate configs.
- Checkpoint artifact naming scheme and retention policy.
- Candidate config source format: Hydra multirun-style overrides, W&B sweep YAML, or a dedicated population YAML.
- Minimum calibration duration and safe throughput floor.
- First accelerator preference order.
- How to represent action-distribution summaries for later diversity-aware promotion.

## Interview Transcript Summary

- User prefers W&B-first orchestration as long as the one-command launcher convenience remains.
- W&B artifacts should be checkpoint handoff unless Kaggle output sync proves clearly easier; Kaggle outputs remain diagnostic backup.
- Population mode and continue-run mode should both exist long-term, with population mode first.
- Population mode should enable testing new self-play opponents and sufficiently different model versions.
- Promising candidates should eventually be promoted through tournaments and self-play pools.
- Promotion should favor high win rates against `scripted_nearest`, W&B cheap filtering, tournament validation, and action-distribution diversity.
- The first milestone should be Population MVP as proof of life, then a full end-to-end tournament/promotion pipeline.
- The MVP should not merely avoid using GPU time; it should spend a small calibration budget to find high stable throughput before long runs.
- Accelerator policy should prefer stronger selectable Kaggle GPU classes using ordered fallback; cross-class calibration can wait until needed.
