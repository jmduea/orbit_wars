# RALPLAN: Self-Play Curriculum

Date: 2026-05-21
Status: Consensus approved, pending execution
Spec: `.omg/specs/deep-interview-self-play-curriculum.md`
Consensus iterations: 2
Critic verdict: Approve with reservations

## Decision

Use a new top-level `curriculum.stages` schema as the single progressive-difficulty authority for JAX mixed 2-player/4-player training.

`training_format.rollout_groups` remains the static shape/allocation surface. Legacy progressive surfaces such as `training_format.phases` and `opponent_mix.curriculum` must be migrated for canonical configs or rejected when conflicting. Runtime curriculum state flows through immutable stage views and fixed-shape JAX inputs, not through mutation of `cfg` after rollout collectors are initialized.

## Principles

- `curriculum.stages` owns progression, opponent mixtures, snapshot policy, and promotion rules.
- `training_format.rollout_groups` remains static and shape-defining.
- JAX rollout APIs receive immutable numeric stage data: stable family ids, probability vectors, masks, snapshot ids, and padded historical params.
- Canonical configs migrate forward; unknown or ambiguous legacy scheduling conflicts fail with clear validation errors.
- Telemetry must prove what stage ran, what opponents were sampled, which historical snapshots were eligible or selected, and why stage transitions happened.

## Target Schema

Add top-level `curriculum` to `TrainConfig`:

```yaml
curriculum:
  enabled: true
  snapshot:
    pool_size: 5
    interval_updates: 100
    deterministic: true
    selection: uniform
    fallback: latest
  stages:
    - id: bootstrap_random
      min_updates: 50
      cooldown_updates: 10
      promote_if:
        metric: overall_win_rate
        op: ">="
        value: 0.55
        window_updates: 5
      opponent_families:
        latest: 0.0
        historical: 0.0
        random: 0.8
        noop: 0.2
        nearest_sniper: 0.0
        turtle: 0.0
        opportunistic: 0.0
    - id: mixed_self_play
      min_updates: 100
      opponent_families:
        latest: 0.55
        historical: 0.25
        random: 0.05
        noop: 0.05
        nearest_sniper: 0.05
        turtle: 0.03
        opportunistic: 0.02
```

Validation rules:

- `stages` is required and non-empty when `curriculum.enabled` is true.
- Stage `id` must be unique, non-empty, and telemetry-safe.
- Family keys must be known stable ids: `latest`, `historical`, `random`, `noop`, `nearest_sniper`, `turtle`, `opportunistic`.
- Family weights must be finite, non-negative, and sum to greater than zero; validation normalizes to a fixed id/probability vector.
- `historical > 0` requires `snapshot.pool_size > 0` and `snapshot.interval_updates > 0`.
- Promotion metric must be from an explicit allowlist.
- `training_format.rollout_groups` may be configured, but curriculum must not mutate it during training.
- Reject simultaneous non-empty legacy `training_format.phases` or `opponent_mix.curriculum` unless handling a known canonical migration.

## Runtime Contract

Create an immutable host-side `StageView` for each update:

- `stage_id`
- `stage_index`
- `family_ids`
- `family_probs`
- `family_mask`
- `snapshot_pool_ids`
- `snapshot_valid_mask`
- `snapshot_age_updates`
- `historical_selection_probs`
- `fallback_family_id`

Implementation constraints:

- Host strings are for telemetry only; JAX receives fixed-shape arrays.
- Replace `CurriculumController.apply(cfg)` mutation with a controller that returns `StageView` plus events.
- `JaxRolloutGroup.collect_fn` accepts stage view arrays and historical pool arrays as dynamic JAX inputs.
- The per-group config captured by rollout collectors remains immutable after collector initialization.

## Historical Snapshot Pool

- Store historical policy params as padded pytrees with leading dimension `snapshot.pool_size`.
- Store aligned arrays for `snapshot_ids`, `snapshot_update`, `snapshot_valid_mask`, and `snapshot_age_updates`.
- Select historical slots with masked probabilities; invalid slots have probability zero.
- If `historical` is sampled while no valid snapshot exists, fallback to `latest` and record fallback counts.
- `latest` uses current `train_state.params`; `historical` uses selected frozen params from the padded pool.
- Snapshot insertion occurs in the training loop after updates at `snapshot.interval_updates`.
- Snapshot ids are monotonically increasing; ring-buffer slots may be reused but external ids remain stable.
- Deterministic historical tests must prove a selected historical snapshot can produce behavior distinct from latest, not only increment counters.

## 2p/4p Slot Semantics

- Sample every non-learner player slot from the same stage family probability vector.
- 2p: one opponent slot per environment.
- 4p: three opponent slots per environment.
- The learner slot is never replaced by sampled opponent behavior.
- `latest` and `historical` are policy-family actions.
- `random`, `noop`, `nearest_sniper`, `turtle`, and `opportunistic` are scripted or baseline family actions.
- Aggregated telemetry denominators are opponent slots, not environments.

## Migration Matrix

| Legacy surface | Action |
| --- | --- |
| `conf/opponent_mix/self_play_curriculum.yaml` | Migrate to top-level `curriculum.enabled=true`, `curriculum.snapshot`, and equivalent `curriculum.stages`. |
| `conf/opponent_mix/latest_only.yaml` | Allow disabled/empty curriculum or represent as `latest: 1.0` with no snapshots. |
| Non-empty `opponent_mix.curriculum` outside known canonical profile | Reject with a message directing migration to `curriculum.stages`. |
| Non-empty `training_format.phases` | Reject for JAX curriculum runs; progressive difficulty belongs in `curriculum.stages`. |
| `self_play_pool_size`, `self_play_snapshot_interval`, `self_play_latest_probability` | Migrate canonical configs to `curriculum.snapshot.*` and stage family weights; reject ad hoc conflicts after migration. |
| `ppo.rollout_groups`, `ppo.phases`, `ppo.num_envs_2p`, `ppo.num_envs_4p` | Continue rejecting in `src/config.py`. |
| Existing `training_format.rollout_groups` profiles | Keep unchanged and static. |

## Telemetry Keys

Per-update records:

- `curriculum_stage_id`
- `curriculum_stage_index`
- `curriculum_stage_update`
- `curriculum_stage_dwell_updates`
- `curriculum_family_prob_latest`
- `curriculum_family_prob_historical`
- `curriculum_family_prob_random`
- `curriculum_family_prob_noop`
- `curriculum_family_prob_nearest_sniper`
- `curriculum_family_prob_turtle`
- `curriculum_family_prob_opportunistic`
- `opponent_slots_total`
- `opponent_slots_latest`
- `opponent_slots_historical`
- `opponent_slots_random`
- `opponent_slots_noop`
- `opponent_slots_nearest_sniper`
- `opponent_slots_turtle`
- `opponent_slots_opportunistic`
- `opponent_historical_fallback_latest_slots`
- `historical_pool_size`
- `historical_pool_capacity`
- `historical_snapshot_ids`
- `historical_snapshot_ages_updates`

Structured events:

- `curriculum_stage_entered`
- `curriculum_stage_promoted`
- `curriculum_stage_promotion_blocked`
- `historical_snapshot_added`
- `historical_snapshot_evicted`
- `historical_snapshot_fallback_latest`
- `curriculum_validation_error`

## Implementation Slices

1. Schema and validation
   - Add curriculum dataclasses and validation.
   - Reject unknown families, bad weights, missing stages, historical without snapshot settings, and legacy conflicts.

2. Canonical config migration and docs
   - Migrate canonical opponent/curriculum profiles.
   - Regenerate/check `default_cfg.yaml`.
   - Update experiment and migration docs.

3. Stage controller and immutable `StageView`
   - Replace config mutation with stage views and events.
   - Define promotion window semantics before implementation. Default recommendation: rolling mean over the last `window_updates` metric values.

4. JAX rollout API and opponent-slot sampling
   - Pass stage arrays into collectors.
   - Support one non-learner slot in 2p and three in 4p.
   - Remove or compatibility-limit old `OpponentRegistry` schedule behavior.

5. Historical snapshot pool
   - Implement padded/masked param pool with ring eviction and deterministic selection.
   - Wire historical params so selected historical behavior is actually frozen and distinct from latest.

6. Telemetry and integration smoke
   - Emit all named keys/events.
   - Add mixed 2p/4p smoke coverage over latest, historical, random, noop, and at least one scripted family.

## Test Plan

- Config tests for valid staged curriculum composition, canonical profile migration, unknown families, negative/nonfinite/zero-sum weights, historical without snapshot settings, invalid promotion metric, and legacy conflict rejection.
- Controller tests for rolling-window metric promotion, minimum dwell, cooldown, blocked promotion event payloads, stable family id/probability construction, and no `TrainConfig` mutation.
- Opponent family tests for stable id mapping, alias handling if needed, and scripted family registration.
- Snapshot pool tests for padded params, valid masks, monotonic ids, ring eviction, masked sampling, empty fallback, deterministic selection, and historical action distinctness.
- JAX rollout tests for fixed-shape stage input, 2p/4p opponent-slot sampling, learner-slot preservation, sampled family counts, and telemetry shape.
- Training-loop smoke test for minimal mixed rollout groups with snapshot insertion and stage transition telemetry.

Verification commands:

```bash
uv run python scripts/generate_default_cfg.py --check
uv run --group dev pytest tests/test_configs.py tests/test_default_cfg_template.py tests/test_jax_ppo.py
uv run --group dev pytest
```

## ADR

Decision: use top-level `curriculum.stages` as the only progressive-difficulty authority for JAX self-play curriculum.

Drivers: eliminate split-brain scheduling, preserve JAX static shapes, make opponent sampling and stage transitions observable, and migrate canonical repo configs while rejecting unknown conflicts.

Alternatives considered:

- Keep `opponent_mix.curriculum`: smaller diff but preserves precedence ambiguity.
- Extend `training_format.phases`: reuses existing controller but couples opponent difficulty to rollout format.
- Implement ad hoc trainer-side snapshots only: fixes one symptom but leaves schedule authority unresolved.

Why chosen: Option B cleanly separates static rollout format from dynamic difficulty, gives the trainer one immutable per-update stage input, and lets validation prevent silent composition.

Consequences: Moderate migration across config, schema, trainer, rollout, tests, docs, and default config. Legacy opponent curriculum fields become deprecated or rejected. Public/external submissions remain a later extension.

## Critic Reservations To Preserve During Execution

- Historical per-slot parameter dispatch is the riskiest area; explicitly gather or vmap selected snapshot params per non-learner slot.
- Promotion window semantics must be fixed before code. Default to rolling mean unless a stronger reason appears.
- Old `OpponentRegistry` schedule behavior and `CurriculumController.apply(cfg)` mutation must be removed or isolated as compatibility-only paths, not wrapped in place.
