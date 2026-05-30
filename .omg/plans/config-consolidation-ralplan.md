# Config Consolidation Ralplan

## Decision

Use a two-layer compatibility migration for the configuration consolidation.

Hydra will gain a responsibility-oriented public config surface first. That surface will normalize into the current runtime `TrainConfig` paths before structured merge. Runtime consumers can then migrate in focused passes after composition, compatibility, and sweep validation are green.

## Decision Drivers

1. Reduce migration risk around self-play, curriculum, opponent mixture, rollout topology, telemetry, and artifacts.
2. Make Hydra multiruns and W&B sweeps operate on coherent responsibility axes.
3. Remove broad `experiment` presets without replacing them with another broad inherited abstraction.
4. Keep old/new ownership conflicts loud and explicit during the migration period.

## Chosen Architecture

- New public groups: `task`, `reward`, `training`, `format`, `opponents`, `curriculum`, `telemetry`, `artifacts`, and optional shallow `campaign`.
- Normalization boundary: `src/config.py`, before `OmegaConf.merge(OmegaConf.structured(TrainConfig), cfg_raw)`.
- Conflict detection: raw composed config checks before defaults obscure whether an old or new path was explicitly supplied.
- Canonical self-play owner: `opponents`.
- Canonical snapshot owner: `opponents.snapshot`.
- Static latest/historical weighting: `opponents.mix.weights`, with old `self_play_latest_probability` treated as a migration input only.
- Script APIs: replace experiment-name arguments immediately with explicit group overrides or group flags.
- `default_cfg.yaml`: remove or quarantine unless a real external consumer requires keeping it as a non-canonical runtime-schema reference.

## Alternatives Considered

- Big-bang schema-first migration: cleaner final surface quickly, but too much runtime and docs churn in one pass.
- Campaign/sweeps first: improves launch ergonomics quickly, but leaves schema ownership fragmented.
- Runtime domain split first: clean architecture eventually, but broadest blast radius before composition is stable.

## Implementation Phases

1. Inventory current launches and references.
   - Map every `conf/experiment/*.yaml` to group-based equivalents or intentional retirement.
   - Inventory docs, scripts, sweeps, tests, and guidance files that mention `experiment=` or `default_cfg.yaml`.

2. Add composition and migration tests.
   - Add `tests/test_config_consolidation.py` or equivalent.
   - Cover root compose, new group defaults, old/new conflict rejection, group-based launch replacements, script defaults, representative multirun combinations, and W&B sweep dry-validation.

3. Introduce responsibility groups and normalizer.
   - Add packaged groups for `task`, `reward`, `training`, `format`, `opponents`, `telemetry`, and `artifacts`.
   - Normalize those groups into current runtime paths in `src/config.py`.
   - Reject explicit old/new conflicts.

4. Resolve opponent/self-play ownership.
   - Public config uses `opponents.self_play`, `opponents.mix`, and `opponents.snapshot`.
   - Phase 1 may write normalized values into current runtime fields for compatibility.
   - Later runtime passes remove duplicated reads.

5. Replace experiment launch surfaces.
   - Remove `experiment` from normal root defaults once group replacements compose.
   - Update benchmark/comparison scripts to accept group overrides or group flags.
   - Remove stale `experiment=` references outside migration notes.

6. Redesign W&B and Hydra sweep examples.
   - Split W&B sweeps by campaign: capacity, budget, reward, task complexity, curriculum, and throughput.
   - Require `wandb.group` and `wandb.tags` metadata.
   - Keep each sweep focused on one coherent axis set.

7. Decide and execute generated default disposition.
   - Preferred: remove or quarantine `default_cfg.yaml`, `default_train_config_path`, and `scripts/generate_default_cfg.py` unless an external consumer is found.
   - If retained, document it as non-canonical and keep generator/check tests.

8. Update docs and guidance.
   - Add responsibility map, old-to-new launch mapping, and examples per group.
   - Update README, docs, and AGENTS guidance to stop teaching experiments as the primary path.

## Acceptance Criteria

- Root config composes without broad experiment presets.
- No active concept has two canonical owners.
- `opponents.snapshot` owns snapshot pool/cadence/selection/fallback.
- `opponents.mix` owns static latest/historical/family weighting.
- Hydra multirun examples vary groups or coherent axis sets.
- W&B sweep files are campaign-specific, grouped/tagged, and dry-compose through Hydra.
- Scripts and docs no longer teach `experiment=...` except intentional migration notes.
- `default_cfg.yaml` is either removed/quarantined or explicitly non-canonical and tested.
- Focused config, curriculum, telemetry, script, and sweep validation passes.

## Verification

Minimum focused verification after implementation:

```bash
uv run --group dev pytest tests/test_config_consolidation.py tests/test_curriculum.py tests/test_telemetry.py
```

If `default_cfg.yaml` is retained, also keep and run the generator/check verification. If it is removed, delete the generator/check requirement and update all references.

## Consensus Result

- Planner recommendation: Option B, two-layer compatibility migration.
- Architect review: approved with pre-merge normalizer, raw conflict checks, and `opponents.snapshot` ownership.
- Critic review: approved after resolving `default_cfg.yaml` and field-level self-play ownership contradictions.
