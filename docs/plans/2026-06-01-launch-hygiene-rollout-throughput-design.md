# Launch Hygiene Rollout Throughput Design

## Problem

Launch hygiene must stay enabled during training, because it prevents degenerate
multi-launch sequences that teach the agent bad launch sizing. The current
post-hygiene training throughput is too low for the 2026-06-23 submission
deadline:

- Pre-hygiene baseline: about 9,776 env steps/sec, 1.64 sec/update.
- Current post-U7: about 950-1,100 env steps/sec, 14-17 sec/update.
- Pass band: at least 8,799 env steps/sec, at most 1.80 sec/update.

Turning hygiene off in training and accepting the regression are both ruled out.

## Profiling Evidence

The focused timing pass used:

```bash
uv run ow benchmark training \
  --preset primary \
  --label timing_post_hygiene \
  --updates 2 \
  --warmup 3 \
  --detailed-timing \
  --out /tmp/ow_timing_split.json
```

Result:

- `seconds_per_update_mean`: 14.39
- `rollout_collect_seconds_per_update_mean`: 13.68
- `ppo_seconds_per_update_mean`: 0.70
- `host_overhead_seconds_per_update_mean`: 0.005
- `default_backend`: `gpu`, `devices`: `["cuda:0"]`
- `mean_active_launches_per_turn`: 0.0

This reverses the earlier PPO-first hypothesis. The bottleneck is rollout
collection, specifically learner action sampling inside
`src/jax/action_sampling.py`, called from `src/jax/rollout/collect.py` for every
rollout step.

The pre-hygiene baseline also had `mean_active_launches_per_turn: 0.0`, so the
new cost is not caused by more launches. It is caused by extra always-on work in
the static factorized K-step rollout sampling scan.

## Hot Path

For each rollout env step, `_sample_shielded_factored_sequence_with_params`:

1. Encodes the turn once.
2. Runs a `jax.lax.scan` over `model.max_moves_k`.
3. At each sub-step, runs factorized decode.
4. Computes the trajectory shield mask.
5. Applies cumulative launch hygiene to that mask.
6. Samples source/target/bucket/stop.
7. Updates remaining ships and cumulative hygiene.

Even when all envs stop at sub-step 0, the static scan still executes the
decode/shield/hygiene body for the remaining sub-steps. With `max_moves_k=5`,
that means paying for four inactive sub-steps per turn in the observed profile.

## Failed Experiments

### All-Env Inactive Fast Path

Hypothesis: since `mean_active_launches_per_turn` is 0.0, skip decode, shield,
sampling, and hygiene when every env row has already stopped.

Result:

- `rollout_collect_seconds_per_update_mean`: 15.30
- `seconds_per_update_mean`: 16.06

This was worse than the 13.68 sec rollout baseline. XLA did not turn the dynamic
all-inactive branch into useful work avoidance for this static scan.

### Compact Rollout Hygiene Carry

Hypothesis: replace rollout's dense forbidden grid with the compact
`ForbiddenCarry` already used in PPO replay.

Result:

- `rollout_collect_seconds_per_update_mean`: 16.15
- `seconds_per_update_mean`: 16.86

This was also worse. The representation of the cumulative forbidden state is not
the main lever.

### Stop-First Rollout Sampling

Hypothesis: sample the factorized stop head after decode and before building
the trajectory shield / launch-hygiene lattice; if every active env row selected
stop, skip shield and hygiene for that sub-step.

Result:

- `rollout_collect_seconds_per_update_mean`: 13.66
- `seconds_per_update_mean`: 14.37

This was effectively neutral versus the measured 13.68 sec rollout baseline. The
observed `mean_active_launches_per_turn: 0.0` does not mean the policy chooses
the stop head immediately for every row; no-op launch sizing can still flow
through source/target/bucket sampling and force the expensive mask path.

### `task=shield_off` Diagnostic

Hypothesis: if trajectory shielding were the dominant work, overriding
`task=shield_off` should recover most throughput.

Result:

- `rollout_collect_seconds_per_update_mean`: 15.91
- `seconds_per_update_mean`: 16.66

This stayed slow. The regression is not explained by trajectory-shield mode
alone. Launch hygiene is independent of `trajectory_shield_mode`, and the
factorized rollout sampler still builds and composes legality masks.

## Current Decision

Do not continue optimizing the forbidden carry representation or adding dynamic
branches inside the K-step scan. The next implementation must either:

- replace full-lattice rollout masking with selected-action validation for the
  sampled launch, or
- add an explicit policy-level no-op/launch gate that can be sampled before
  source/target/bucket legality construction, or
- otherwise reduce the number of factorized decode/shield/hygiene passes per env
  step without weakening launch-hygiene semantics.

## Expected Impact

The current evidence points away from micro-optimizing mask representation and
toward changing rollout sampling semantics so no-op decisions do not require a
full launch-legality lattice. The safest next design is selected-action
validation: sample the candidate launch from cheaper policy masks, validate only
the selected `(source, target_slot, bucket)` against trajectory shield and launch
hygiene, and convert invalid selected launches to no-op/stop with consistent
stored log-prob semantics.

## Verification

Run in order:

1. Focused correctness tests for launch hygiene and factorized sampling.
2. `uv run ow benchmark training --preset primary --label timing_after_fastpath --updates 2 --warmup 3 --detailed-timing --out /tmp/ow_timing_after_fastpath.json`
3. `make test-launch-hygiene-throughput`
4. `make test-launch-hygiene-e2e-throughput`

The implementation is acceptable only if launch-hygiene correctness still passes
and tier-2 e2e throughput materially improves. The target remains the calibrated
baseline pass band.
