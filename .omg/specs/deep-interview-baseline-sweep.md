# Deep Interview Spec: Workstation-Friendly Baseline Sweep

## Goal

Design a W&B sweep workflow that finds a sensible default training baseline for Orbit Wars. The baseline should be performant enough to serve as a credible comparison point, fast enough to iterate on, stable enough across seeds to support confident later sweeps, and light enough that the workstation remains usable for normal foreground activity such as watching video on another monitor.

## Recommended Direction

Use a two-stage sweep:

1. Comfort and throughput filter: quickly reject configs that are too slow, too resource-heavy, or likely to disturb workstation use.
2. Performance and stability validation: run the best comfortable configs longer and repeat the finalists across seeds.

Use one locked default baseline after validation, plus a small set of sentinel checks when sweeping major future axes. This avoids a full interaction sweep up front while still catching cases where the baseline choice masks important interactions.

## Constraints

- Prefer a Pareto-balanced baseline over a single-metric winner.
- Stability and reproducibility matter nearly as much as raw performance.
- The baseline must stay inside a conservative workstation-load envelope.
- The sweep should use both a conservative config ceiling and recorded utilization guardrails.
- Avoid a full interaction sweep unless it can be bounded to a reasonable runtime.
- Keep the result useful for later hyperparameter sweeps without gimping the search space.

## Non-Goals

- Do not prove every interaction between every model, task, reward, curriculum, and PPO parameter.
- Do not optimize solely for maximum `samples_per_sec`.
- Do not optimize solely for maximum early `overall_win_rate` if the config is unstable or uncomfortable to run.
- Do not choose a baseline that requires unattended/heavy-machine conditions for normal development comparisons.

## Candidate Sweep Shape

### Stage 1: Comfort And Throughput Filter

Start from the existing throughput sweep shape and keep the envelope conservative:

- `model=attention` initially.
- Cap `training.num_envs` at the comfortable range, likely `8` and `16` first.
- Include smaller and medium `training.rollout_steps`, likely `64`, `128`, and possibly `250` only if comfortable.
- Keep `training.minibatch_size` values that fit without memory pressure, likely `256`, `512`, and maybe `1024` if utilization is acceptable.
- Consider setting `training.rollout_microbatch_envs` for configs that otherwise spike load.
- Keep `training.enable_gradient_checkpointing=true` unless profiling shows it costs too much wall time.
- Reduce optional telemetry groups and avoid expensive artifact behavior during screening if it affects interactivity.

Primary filter metrics:

- `samples_per_sec`
- `env_steps_per_sec`
- `rollout_seconds`
- `ppo_seconds`
- `update_seconds`
- observed or logged workstation utilization guardrails
- manual comfort status: video playback remains smooth

### Stage 2: Performance And Stability Validation

Take the top comfortable Stage 1 configs and validate them with a modest budget:

- Use longer runs than Stage 1, such as the current default `training.total_updates=500` or another agreed midpoint.
- Repeat finalists across multiple seeds; start with 3 seeds unless runtime suggests 5 is affordable.
- Rank by Pareto balance across performance, throughput, and stability.

Primary performance metrics:

- `overall_win_rate`
- `episode_reward_mean`
- `win_rate_2p`
- `first_place_rate_4p`
- policy-health metrics such as `approx_kl` and entropy

Stability metrics:

- variance across seeds
- absence of collapsed or pathological runs
- consistent timing profile between seeds

## Sentinel Checks

After choosing one default baseline, keep a small sentinel set for future sweeps:

- A lightweight model-capacity sentinel, such as `attention` versus one entity-transformer size.
- A task-complexity sentinel, such as default `candidate_count` versus one higher value.
- A curriculum/opponent sentinel if future work changes opponent profiles.

These sentinels should be rerun when sweeping major axes so later experiments can detect baseline interactions without paying for a full interaction matrix.

## Acceptance Criteria

- A proposed baseline config is selected from a two-stage sweep, not from a single short run.
- The selected baseline is Pareto-competitive on performance and throughput among comfortable configs.
- The selected baseline has repeated-seed evidence, preferably at least 3 seeds.
- The selected baseline stays under the workstation comfort envelope: conservative parallelism plus utilization guardrails, with no observed video skipping during active-use runs.
- The chosen baseline can be expressed as Hydra overrides or committed config values for repeatable future comparisons.
- A small sentinel-check list is documented for later sweeps that might interact with the baseline.

## Assumptions Exposed And Resolved

- A single locked baseline is useful, but only if paired with sentinel checks for major future sweep axes.
- A full interaction sweep is likely too expensive for the first pass.
- Workstation comfort is a first-class constraint, not an after-the-fact preference.
- Stability across seeds is required before declaring a default baseline.

## Open Details For Planning

- Exact utilization thresholds for CPU, GPU, RAM, and thermals.
- Exact Stage 1 and Stage 2 run budgets.
- Whether to implement utilization logging directly, document a manual guardrail procedure, or both.
- Whether the first deliverable should be new W&B sweep YAML files, documentation, helper scripts, or all three.

## Interview Transcript

- User requested a sweep to find sensible default baselines balancing promising performance, training throughput, and generalization across later parameter sweeps.
- User added a workstation comfort constraint: training should not be so taxing that video playback on another monitor skips frames.
- User selected a Pareto-balanced and stability/reproducibility-oriented success rule.
- User selected conservative config ceilings plus utilization guardrails for workstation load.
- User clarified that the goal is to lock in solid performant parameters for later sweeps without gimping hyperparameter search.
- User selected a two-stage budget.
- User was open to baseline-interaction safeguards, preferring to avoid a full interaction sweep unless it is reasonably bounded.
