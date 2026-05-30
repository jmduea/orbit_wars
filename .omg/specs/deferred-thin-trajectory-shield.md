# Thin trajectory shield (drop dynamic per-bucket simulation)

**Status:** deferred
**Slug:** `thin-trajectory-shield`
**Related:** `intercept-edge-features` (M4)
**Origin:** Descoped from M4 during ralplan iter-1 per architect+critic synthesis recommendation.

## Why deferred

During ralplan iter-1 review of the original M4 plan (which combined edge-intercept features + shield thinning), the critic surfaced three critical issues that all originate from the shield-thinning surface:

- **C1**: `sun_cross_at_intercept` feature uses `target_future_pos`, but `_static_trajectory_reason_codes_jax` and `_launch_angle_for_edge` (in `src/game/trajectory_shield.py`) use **current** target position. The two disagree for rotating targets.
- **C2**: Forcing `static_fast_path_enabled=True` bypasses the rotating-geometry guard at `src/game/trajectory_shield.py:378-384`, but the static predicate was never built to handle rotating sources/targets.
- **C3**: Proposed Python static-only legality net mirrors the same predicate, so it does not provide defense-in-depth against Kaggle simulator drift.

Architect additionally raised:
- **A1**: `jax.lax.cond` doesn't collapse trace cost — Python-time branching is needed.
- **A2**: Missing Python↔JAX static-shield parity test as a hard gate.
- **A3**: `legal_non_noop_rate` semantic collision across modes.

Both agents independently proposed decoupling: land edge features first (M4), measure W1 reward gate in isolation, then come back to shield thinning as a separate milestone evaluated on throughput gates alone.

## When to revive

After M4 (`intercept-edge-features`) lands and:
- Reward gate W1 (≥2% `episode_reward_mean` lift) is confirmed met.
- No throughput regression observed.
- A baseline policy trained against the new edge features exists for shield-thinning ablation.

At that point, this spec should be reactivated and re-planned with the critic's concerns explicitly addressed.

## Required scope adjustments before reactivation

1. Resolve C1: extend static predicate to evaluate intercept-time aim lines, OR restrict static-only mode to the existing fast-path subset (non-rotating source AND non-rotating target).
2. Resolve C2: explicit guard for rotating sources/targets in the static branch.
3. Resolve C3: defense-in-depth requires a non-mirrored legality oracle (e.g., a separate Python simulator that doesn't share the JAX static predicate's blind spots).
4. Resolve A1: Python-time branching at config-resolve time, not `jax.lax.cond`.
5. Resolve A2: ≥1000-sample fuzz parity test as a hard CI gate.
6. Resolve A3: mode-tagged shield diagnostics.

## Acceptance criteria sketch (locked at reactivation)

- Throughput improvement ≥10-15% on `env_steps_per_sec` vs M4 baseline.
- Submission validator passes for ≥100 sampled games per format, zero illegal-action rejections.
- No regression on M4 reward baseline.
- Python↔JAX parity fuzz test passes ≥1000 samples, zero divergence.

## Out of scope at reactivation

- Eliminating shield entirely.
- Action space changes.
- Encoder changes.
