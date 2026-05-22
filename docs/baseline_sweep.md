# Workstation-Friendly Baseline Sweep

This workflow finds a default Orbit Wars training baseline that is useful for later comparisons without making active workstation use miserable. It favors Pareto-balanced configs: credible performance, good throughput, stable repeated seeds, and smooth foreground use.

## Sweep Files

- `conf/sweeps/wandb/baseline_stage1_comfort.yaml` screens conservative rollout shapes for throughput and workstation comfort.
- `conf/sweeps/wandb/baseline_stage2_stability.yaml` validates a chosen Stage 1 finalist across fixed seeds.
- `conf/sweeps/wandb/baseline_sentinels.yaml` runs a small interaction smoke check after a baseline is chosen.

## Stage 1: Comfort Filter

Stage 1 is intentionally cheap. It controls load with the `format` group because mixed rollout configs define their own rollout-group environment counts. Do not assume `training.num_envs` changes active parallelism when `format.rollout_groups` is populated.

The default Stage 1 template starts with `format=mix_2p_4p_8env`, `training.total_updates=3`, and rollout lengths `16` and `32`. A one-update smoke with `rollout_steps=64` took several minutes on the current workstation, so larger Stage 1 settings should be treated as follow-up probes rather than the first comfort screen.

Launch the sweep:

```bash
wandb sweep conf/sweeps/wandb/baseline_stage1_comfort.yaml
```

Then run an agent from the returned sweep ID:

```bash
wandb agent <entity>/<project>/<sweep_id>
```

Reject a Stage 1 run if any of these occur during the run:

- Video playback on the second monitor visibly skips frames.
- The desktop becomes noticeably laggy for normal foreground use.
- System RAM exceeds 85% for more than a brief spike.
- Swap usage grows during the run.
- GPU utilization or temperature remains high enough to disturb foreground use or trigger thermal throttling.
- `samples_per_sec`, `env_steps_per_sec`, `rollout_seconds`, `ppo_seconds`, or `update_seconds` show an obvious pathological outlier.

Record the comfort result in the W&B run notes or tags before promoting a finalist. Use a simple status such as `comfort=pass`, `comfort=borderline`, or `comfort=fail`.

## Stage 2: Stability Validation

Stage 2 should compare independent seeds for a fixed finalist config. Before launching, edit `baseline_stage2_stability.yaml` so the fixed values match the Stage 1 finalist. Keep `seed.values` as a grid axis rather than treating seed as an optimization parameter.

The default Stage 2 template uses `training.total_updates=25` so validation remains bounded. Increase it only after the comfort screen shows the finalist is pleasant to run.

Default seed set:

- `101`
- `202`
- `303`

Promote to 5 seeds only if the first 3 seeds are close enough that extra confidence is worth the runtime.

Launch the sweep:

```bash
wandb sweep conf/sweeps/wandb/baseline_stage2_stability.yaml
wandb agent <entity>/<project>/<sweep_id>
```

## Promotion Rubric

A baseline can be selected only after Stage 2 has completed at least 3 seeds for the same fixed config.

Reject a finalist if:

- Any seed collapses or produces clearly pathological behavior.
- `approx_kl` or entropy suggests unstable policy updates compared with other finalists.
- Timing metrics vary enough across seeds to make the config hard to compare fairly.
- The config fails the Stage 1 comfort gate.

Prefer the finalist that is Pareto-competitive across:

- `overall_win_rate`
- `episode_reward_mean`
- `win_rate_2p`
- `first_place_rate_4p`
- `samples_per_sec`
- `env_steps_per_sec`
- timing breakdowns: `rollout_seconds`, `ppo_seconds`, `update_seconds`
- seed-to-seed variance

Tie-breakers, in order:

1. No failed or collapsed seeds.
2. Better median `overall_win_rate` and `episode_reward_mean`.
3. Lower seed variance.
4. Higher `samples_per_sec` without comfort issues.
5. Simpler config, such as the smaller rollout format or default model.

## Sentinel Checks

Run `baseline_sentinels.yaml` after selecting a baseline and before relying on it for major future sweeps. The sentinel sweep is not a full interaction matrix. It is a bounded smoke check for obvious interactions across model capacity and task complexity.

Opponent-profile sentinels should be run as focused follow-ups because opponent and curriculum profiles must be paired deliberately. For example, `opponents=latest_only` should use `curriculum=latest_only` rather than the default historical curriculum.

Before launch, edit the fixed training values to match the selected baseline:

```bash
wandb sweep conf/sweeps/wandb/baseline_sentinels.yaml
wandb agent <entity>/<project>/<sweep_id>
```

Treat sentinel results as a warning signal. If a sentinel run changes conclusions dramatically, plan a focused follow-up sweep on that axis before locking broad conclusions.

## Baseline Evidence Template

Copy this section into an experiment note or PR when promoting a baseline.

```markdown
## Selected Baseline

Hydra overrides:

- `model=...`
- `format=...`
- `training.rollout_steps=...`
- `training.minibatch_size=...`
- `training.rollout_microbatch_envs=...`
- `training.lr=...`
- `training.ent_coef=...`

Stage 1 W&B sweep:

- Sweep ID:
- Comfort status:
- Rejected configs and reasons:

Stage 2 W&B sweep:

- Sweep ID:
- Seeds:
- Median `overall_win_rate`:
- Median `episode_reward_mean`:
- Median `samples_per_sec`:
- Seed variance notes:
- Policy-health notes:

Sentinel sweep:

- Sweep ID:
- Any interaction warning:

Decision:

- Promote / do not promote:
- Reason:
- Follow-up sweeps:
```

## Local Smoke Check

Before launching W&B agents, verify representative configs compose and can start:

```bash
uv run python -m src.train print_resolved_config=true model=attention format=mix_2p_4p_8env training.total_updates=1 training.rollout_steps=64 training.minibatch_size=256 training.rollout_microbatch_envs=4 artifacts.artifact_pipeline.enabled=false artifacts.replay.enabled=false
uv run python -m src.train model=attention format=mix_2p_4p_8env training.total_updates=2 training.rollout_steps=64 training.minibatch_size=256 training.rollout_microbatch_envs=4 telemetry.wandb.enabled=false artifacts.artifact_pipeline.enabled=false artifacts.replay.enabled=false
```
