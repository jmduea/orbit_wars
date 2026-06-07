# Opponent Throughput Recovery

Use this runbook when opponent rollout sampling is large enough to block useful training iteration.

The recovery path intentionally trades opponent strength for rollout throughput. It is not a full self-play replacement; it is an operator escape hatch for keeping training moving while the opponent sampler is still expensive.

## Profiles

| Purpose | Overrides | Notes |
| --- | --- | --- |
| Recovery training | `curriculum=off opponents=throughput_recovery telemetry=opponent_recovery` | Direct random JAX opponent mode, no self-play, no historical pool. |
| Noop floor | `curriculum=off opponents=throughput_recovery_floor telemetry=opponent_recovery` | Direct noop JAX opponent mode, useful as the cheapest rollout lower bound. |
| Composition visibility | `telemetry=opponent_recovery` | Keeps core timing/progress and opponent slot metrics, disables losses-heavy telemetry. |

Example:

```bash
uv run ow train \
  task=map_pool \
  curriculum=off \
  opponents=throughput_recovery \
  telemetry=opponent_recovery \
  artifacts=disabled
```

The ce-optimize pre-loop ladder exposes the same cheap rungs:

```python
from scripts.ce_optimize.opponent_ladder_rungs import LADDER_RUNG_ORDER

assert "noop" in LADDER_RUNG_ORDER
assert "recovery" in LADDER_RUNG_ORDER
```

## Measuring

Use the offline profiler for phase shares. Do not enable `telemetry=rollout_phase_timing` on production `ow train`.

Candidate:

```bash
uv run ow benchmark rollout-phase-profile \
  --preset admission \
  --train-overrides task=map_pool curriculum=off opponents=throughput_recovery \
  --updates 5 \
  --warmup 2 \
  --out /tmp/recovery-profile.json
```

Baseline comparison:

```bash
uv run ow benchmark rollout-phase-breakdown /tmp/recovery-profile.json \
  --baseline /tmp/production-profile.json \
  --min-opponent-drop-points 10
```

The comparison is diagnostic only. It reports the opponent phase share drop in percentage points; it does not admit a training change by itself.

## Resume Safety

The recovery profiles set `opponents.snapshot.pool_size=0` and `opponents.snapshot.interval_updates=0`. Checkpoint restore keeps the freshly configured historical-pool shape when a stored checkpoint pool has a different capacity, so resuming a production checkpoint into a recovery profile does not silently re-enable historical opponents.
