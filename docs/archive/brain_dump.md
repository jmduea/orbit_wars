# Brain dump (archived 2026-05-30)

Retired in favor of [ROADMAP.md](../ROADMAP.md) + GitHub issues + `.omg/workflow-manifest.json`.
Preserved for historical context only — do not add new items here.

---

# Brain Dump

Centralized repository of all the things that i've felt the need to write down in relation to this project ranging from ideas to questions and issues that need to be addressed.

## Ideas

- **Unify experiment outputs (v2)** — finish the output-standardization mental model: single map for `outputs/` vs `artifacts/` vs `wandb/`; campaign `promoted/current_best/`; upload promoted checkpoints to W&B only; sweep/multirun dirs and run names from Hydra overrides; default `wandb.group` to `output.campaign`. **Scope:** [output-storage-roadmap](../../.omg/plans/output-storage-roadmap.md). Builds on [deep-interview-output-standardization](../../.omg/specs/deep-interview-output-standardization.md) / [ralplan](../../.omg/plans/output-standardization-ralplan.md). On ROADMAP **Later**.

- Add informative wandb.tags for config groups (override/append per config group for W&B filtering).

- Separate `num_envs` via training weights instead of format YAML (2p/4p weighting).

- Local tournament / ranking eval for best agents.

- VRAM profile from W&B run data (compile time, VRAM, wall time metrics).

- Debug metric: average ships per fleet launch.

## Questions

- JAX compile time bounds for training runs.

- `survival_time` definition and relation to performance.

## Issues (migrated to ROADMAP / GitHub)

- Kaggle docker validation failure → ROADMAP **Now** [#96](https://github.com/jmduea/orbit_wars/issues/96)
- Population worker broken → ROADMAP **Done** [#97](https://github.com/jmduea/orbit_wars/issues/97) (`850568a`, `e98f452`, `20c8011`)
- Seed scheduler verification → [#99](https://github.com/jmduea/orbit_wars/issues/99)
