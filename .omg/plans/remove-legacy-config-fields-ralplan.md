# Remove Legacy Config Fields Ralplan

## Decision

Use a direct responsibility-group runtime migration. The public Hydra surface and runtime `TrainConfig` shape will match: `task`, `reward`, `training`, `format`, `opponents`, `curriculum`, `telemetry`, `artifacts`, and `model`.

No general legacy aliases, null anchors, or compatibility normalizers will remain. Private adapters are allowed only at hard external boundaries such as old checkpoint payload loading, and they must not reintroduce old fields on `TrainConfig`.

## Decision Drivers

1. The current config tree should be readable at a glance, with one owner per concept.
2. Runtime code should reinforce the same ownership model users see in Hydra.
3. Old override paths should fail clearly instead of silently composing through hidden adapters.

## Field Ownership Matrix

| Old runtime path | New runtime path |
| --- | --- |
| `env.candidate_count` | `task.candidate_count` |
| `env.ship_bucket_count` | `task.ship_bucket_count` |
| `env.max_fleets` | `task.max_fleets` |
| `env.player_count` | `task.player_count` |
| `env.max_ships` | `task.max_ships` |
| `env.feature_history_steps` | `task.feature_history_steps` |
| `env.trajectory_shield_*` | `task.trajectory_shield_*` |
| `env.reward_*` | `reward.reward_*` |
| `env.early_terminal_reward_shaping_*` | `reward.early_terminal_reward_shaping_*` |
| `env.terminal_reward_mode` | `reward.terminal_reward_mode` |
| `ppo.*` | `training.*` |
| `training_format.*` | `format.*` |
| `opponent` | `opponents.mode.opponent` |
| `multi_opponent_mode` | `opponents.mode.multi_opponent_mode` |
| `alternate_player_sides` | `opponents.mode.alternate_player_sides` |
| `self_play_enabled` | `opponents.self_play.enabled` |
| `self_play_update_interval` | `opponents.self_play.update_interval` |
| `self_play_deterministic` | `opponents.self_play.deterministic` |
| `self_play_pool_size` | `opponents.snapshot.pool_size` |
| `self_play_snapshot_interval` | `opponents.snapshot.interval_updates` |
| `self_play_latest_probability` | `opponents.mix.weights.latest` / `opponents.mix.weights.historical` |
| `opponent_mix.weights.*` | `opponents.mix.weights.*` |
| `opponent_mix.temperature` | `opponents.mix.temperature` |
| `opponent_mix.curriculum` | removed; use `curriculum.stages` |
| `curriculum.snapshot.*` | `opponents.snapshot.*` |
| `wandb.*` | `telemetry.wandb.*` |
| `save_dir` | `artifacts.save_dir` |
| `checkpoint_every` | `artifacts.checkpoint_every` |
| `artifact_pipeline.*` | `artifacts.artifact_pipeline.*` |
| `replay.*` | `artifacts.replay.*` |
| `checkpoint_retention.*` | `artifacts.checkpoint_retention.*` |
| `reseed_every_updates` | `training.reseed_every_updates` |
| `reseed_on_plateau` | `training.reseed_on_plateau` |
| `plateau_metric` | `training.plateau_metric` |
| `plateau_window` | `training.plateau_window` |
| `plateau_delta` | `training.plateau_delta` |

## Snapshot Ownership

`opponents.snapshot` is the sole owner for historical policy pool size, snapshot cadence, deterministic sampling, selection, and fallback. Runtime curriculum code may receive snapshot data as an argument or read `cfg.opponents.snapshot`; it must not use `cfg.curriculum.snapshot` or copy snapshot settings into curriculum.

## Implementation Checkpoints

1. **Schema and config composition**
   - Replace legacy dataclasses in `src/conf_schema.py` with canonical dataclasses.
   - Remove legacy anchors from `conf/config.yaml`.
   - Remove responsibility-to-legacy normalization and old/new conflict checks from `src/config.py`.
   - Keep validation, rewritten to canonical paths.

2. **Runtime consumers**
   - Migrate `src/env.py`, `src/features.py`, `src/jax_features.py`, `src/jax_env.py`, `src/jax_policy.py`, `src/jax_ppo.py`, `src/jax_train.py`, `src/opponents.py`, `src/opponent_pool.py`, `src/curriculum.py`, `src/replay.py`, `src/run_paths.py`, `src/telemetry.py`, and scripts from legacy paths to canonical paths.
   - Local rollout specialization may mutate copied `cfg.task.player_count` and `cfg.training.num_envs`; this is not a compatibility alias.

3. **Boundary compatibility**
   - New checkpoint payloads should store canonical `TrainConfig` shape.
   - Old checkpoint payload loading must have an explicit boundary contract. Preferred contract: reject pre-migration pickled `TrainConfig` payloads with a clear controlled error that tells the user the checkpoint was produced with the removed legacy config schema. If compatibility is chosen instead, implement a private boundary adapter and test it with a synthetic legacy payload.
   - W&B config logging should emit canonical nested keys from `asdict(cfg)`.

4. **Tests and docs**
   - Replace legacy-parse tests with negative tests proving old overrides fail.
   - Update tests that construct configs directly to use canonical paths.
   - Remove migration-era compatibility promises from docs and guidance.

## Acceptance Criteria

- `conf/config.yaml` has no legacy null anchors.
- `TrainConfig` has no `env`, `ppo`, `training_format`, `opponent_mix`, `wandb`, `artifact_pipeline`, `replay`, `checkpoint_retention`, `save_dir`, `checkpoint_every`, flat opponent fields, flat self-play fields, or flat reseed/plateau fields.
- Static checks find no runtime references to forbidden legacy attributes on any `TrainConfig`-derived variable, not just variables named `cfg`. This includes aliases such as `group_cfg.env`, `artifact_cfg = cfg.artifact_pipeline`, `self._cfg.wandb`, and direct dataclass field access in tests/scripts.
- Canonical overrides compose and legacy overrides fail.
- `opponents.snapshot` is the only runtime snapshot owner.
- Resume/checkpoint behavior has an explicit test for old-schema payload handling: either controlled rejection with a clear message, or successful private boundary conversion.
- Focused config, curriculum, telemetry, JAX policy/PPO, replay, artifact, and script tests pass or any remaining failure is documented as unrelated pre-existing work.

## Verification Plan

Run focused tests after implementation:

```bash
uv run --group dev pytest tests/test_config_consolidation.py tests/test_curriculum.py tests/test_telemetry.py tests/test_jax_policy.py tests/test_jax_ppo.py tests/test_replay.py tests/test_artifact_pipeline.py
```

Run compose/smoke checks:

```bash
uv run python -m src.train print_resolved_config=true
uv run python -m src.train training.total_updates=1 artifacts.artifact_pipeline.enabled=false artifacts.replay.enabled=false telemetry.wandb.enabled=false
```

Run negative legacy checks; these must fail:

```bash
uv run python -m src.train ppo.total_updates=1
uv run python -m src.train env.candidate_count=16
uv run python -m src.train self_play_enabled=false
uv run python -m src.train save_dir=artifacts/tmp
```

Run broad reference checks. These should include simple text searches and, if text search is too noisy, a small AST/static script that flags removed attributes regardless of variable name:

```bash
rg -n "cfg\.(env|ppo|training_format|opponent_mix|wandb|artifact_pipeline|replay|checkpoint_retention|self_play_|opponent\b|multi_opponent_mode|alternate_player_sides|save_dir|checkpoint_every|reseed_|plateau_)" src tests scripts
rg -n "\.(env|ppo|training_format|opponent_mix|wandb|artifact_pipeline|replay|checkpoint_retention|save_dir|checkpoint_every|multi_opponent_mode|alternate_player_sides)\b|\.self_play_|\.opponent\b|\.reseed_|\.plateau_" src tests scripts
```

Expected result: no forbidden runtime references except intentional negative-test strings and migration-boundary code/tests.

## Consensus Notes

- Planner recommended direct migration with private adapters only where unavoidable.
- Architect approved with constraints: `opponents.snapshot` is canonical, no general aliases, private boundary adapters only.
- Critic initially rejected for missing schema matrix, snapshot plan, staged execution, and boundary criteria; this revision addresses those issues.
