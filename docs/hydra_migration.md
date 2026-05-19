# Hydra migration guide

This document describes the exact migration from legacy `--config` CLI usage to Hydra-native CLI usage.

## Old → new command mapping

| Old command | New command |
| --- | --- |
| `python -m src.train --config default_cfg.yaml` | `python -m src.train` |
| `python -m src.train --config configs/full_training.yaml` | `python -m src.train experiment=full_training` |
| `python -m src.train --config configs/attention_training.yaml` | `python -m src.train experiment=attention_training` |
| `python -m src.train --config configs/shaped_reward_training.yaml` | `python -m src.train experiment=shaped_reward_training` |
| `python -m src.train --config configs/attention_shaped_reward_training.yaml` | `python -m src.train experiment=attention_shaped_reward` |
| `python -m src.train --config configs/attention_self_play_pool.yaml` | `python -m src.train experiment=attention_self_play_pool` |
| `python -m src.train --config configs/attention_candidates_16.yaml` | `python -m src.train experiment=attention_candidates_16` |
| `python -m src.train --config configs/attention_candidates_24.yaml` | `python -m src.train experiment=attention_candidates_24` |
| `python -m src.train --config configs/mixed_2p_4p_training.yaml` | `python -m src.train experiment=mixed_2p_4p_training` |
| `python -m src.train --config configs/jax_training.yaml` | `python -m src.train experiment=jax_training` |
| `python -m src.train --config configs/jax_self_play_shaped_reward_training.yaml` | `python -m src.train experiment=jax_self_play_shaped_reward` |
| `python -m src.train --config configs/jax_mixed_2p_4p_training.yaml` | `python -m src.train experiment=jax_mixed_2p_4p_training` |
| `python -m src.train --config configs/jax_entity_transformer_500k.yaml` | `python -m src.train experiment=jax_entity_transformer_500k` |
| `python -m src.train --config configs/jax_entity_transformer_700k.yaml` | `python -m src.train experiment=jax_entity_transformer_700k` |
| `python -m src.train --config configs/jax_entity_transformer_1m.yaml` | `python -m src.train experiment=jax_entity_transformer_1m` |

> Tip: prepend `uv run` in this repo environment, e.g. `uv run python -m src.train experiment=attention_training`.

## Compatibility timeline

- **Current state**: legacy `--config` parsing is still recognized as a compatibility path, but Hydra-native `experiment=...` is the primary interface.
- **Deprecation window**: `--config` usage is deprecated and should be removed from scripts/automation immediately.
- **Forward-safe path**: update all local scripts, CI jobs, and docs to Hydra overrides now.

## Troubleshooting common migration errors

### 1) Override key errors

Symptoms:
- `Key 'foo' is not in struct`
- `Could not override 'foo.bar'`

Fixes:
- Verify the key exists in the composed config.
- Use exact nested path names (`ppo.total_updates`, `env.player_count`, etc.).
- If you intentionally add a new key, use `+foo=bar`.

### 2) Schema mismatch / type mismatch

Symptoms:
- Value conversion errors (string vs int/bool/list)
- Dataclass/schema validation failures

Fixes:
- Pass typed values in Hydra syntax (`ppo.total_updates=2000`, `env.player_count=4`).
- Quote only when needed.
- Avoid changing shape-defining model keys when resuming from existing checkpoints.

### 3) Missing config group / missing experiment

Symptoms:
- `Could not find 'experiment/<name>'`

Fixes:
- Confirm the experiment exists under `conf/experiment/`.
- Use one of the documented experiment names.
- If migrating from a removed legacy YAML filename, map it through the table above.

