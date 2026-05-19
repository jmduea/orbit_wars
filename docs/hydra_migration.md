# Hydra migration guide

This document describes the exact migration from legacy `--config` CLI usage to Hydra-native CLI usage.

## Hydra-native command usage

Use Hydra overrides directly in scripts/automation. Example:

`uv run python -m src.train experiment=attention_training`

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
- If migrating from a removed legacy YAML filename, map it to the equivalent `experiment=<name>` preset under `conf/experiment/`.

## Canonical experiment authoring policy

- Canonical experiment editing and sweeps happen only in `conf/` (`conf/config.yaml`, `conf/experiment/*.yaml`, and config groups).
- `configs/` has been removed; use Hydra experiment selection from `conf/experiment/` for all authoring and execution.
