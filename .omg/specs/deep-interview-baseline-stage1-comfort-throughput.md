# Deep Interview Spec: Baseline Stage 1 Comfort Throughput Sweep

Date: 2026-05-22

## Goal

Revise `conf/sweeps/wandb/baseline_stage1_comfort.yaml` so Stage 1 maps aggressive throughput upper bounds across every model choice while still capturing whether each configuration remains usable during comfortable video streaming.

## Constraints

- Cover all model group choices: `mlp`, `attention`, `entity_transformer_500k`, `entity_transformer_700k`, `entity_transformer_1m`, and `gnn_pointer`.
- Keep the sweep W&B-compatible and maintainable. Prefer simple unconditional grid axes unless evidence justifies separate follow-up sweeps.
- Treat comfort as an empirical gate, not a pre-filter. Include configurations that may fail comfort if they help map the upper bound.
- Do not prune axes now for interpretability. Drop or split axes later only when sweep evidence shows they are noisy, redundant, or dominated by comfort failures.
- Preserve existing artifact/replay disabling so throughput measurements are not confounded by checkpoint or replay work.

## Target Sweep Surface

- `model`: all six model choices.
- `format`: `mix_2p_4p_8env` and `mix_2p_4p_16env`.
- `training.rollout_steps`: `64`, `128`, and `250`.
- `training.minibatch_size`: `256`, `512`, and `1024`.
- `training.rollout_microbatch_envs`: `4` and `8`.

This is an approximately 216-run grid before any W&B agent limits or early manual stopping.

## Acceptance Criteria

- The Stage 1 comfort sweep includes every model choice in `conf/model/`.
- The sweep probes both currently available mixed 2p/4p rollout formats.
- The sweep uses aggressive throughput-oriented rollout and minibatch settings aligned with the existing `throughput` sweep.
- The sweep includes a microbatch axis so comfort/load smoothing can be compared empirically.
- W&B metadata makes the new purpose clear through group and tags.
- The revised sweep composes as Hydra overrides for representative model/format combinations.
- Documentation is updated if needed so future readers understand this Stage 1 is now an aggressive upper-bound map rather than a tiny conservative comfort screen.

## Non-Goals

- Do not select or promote the final baseline in this change.
- Do not redesign Stage 2 stability validation.
- Do not add model-specific conditional sweep logic unless W&B support is clean and evidence requires it.
- Do not change training code unless config composition exposes a real bug.

## Assumptions Resolved

- The user prefers aggressive throughput probing over a small conservative comfort grid.
- Short or severe comfort failures are acceptable data points for Stage 1.
- The first version should keep all requested axes; evidence from W&B runs should drive later pruning.
- Simple unconditional W&B grid axes are preferred for maintainability.

## Ontology

- Sweep: W&B YAML file that generates Hydra training overrides.
- Comfort: foreground usability during video streaming, including video smoothness, desktop responsiveness, memory pressure, and thermal/GPU behavior.
- Throughput: primarily `samples_per_sec`, with timing breakdowns and `env_steps_per_sec` as supporting evidence.
- Model choice: Hydra `model` group entry under `conf/model/`.
- Format: Hydra `format` group controlling mixed 2p/4p rollout group environment counts.

## Interview Transcript Summary

1. The user chose aggressive all-model throughput probing over a tiny comfort-preserving sweep.
2. The user wanted maximum throughput unless configurations fail badly.
3. The user set no fixed run-count cap.
4. The user preferred W&B-compatible maintainability over complicated conditional model-specific axes.
5. The user requested the combined axis set: all models, formats, higher rollout values, minibatches, and microbatch values.
6. The user explicitly asked to keep all axes for now and prune later only with evidence.
