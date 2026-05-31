# Deep Interview Spec: Composable Multi-Value Head

## Goal

Add a composable value-head component to the JAX policy stack so value estimation can be configured independently of encoder and decoder choices. The first new behavior should support format-aware value estimation for mixed 2-player and 4-player training, allowing the critic to learn separate value functions for each game format.

## Context

- `src/jax_policy.py` already composes `ComposablePlanetPolicy` from an encoder module and decoder module.
- The current critic is an inline shared MLP inside `ComposablePlanetPolicy.__call__` using `encoder_out.value_input` and emitting a `(batch,)` value tensor.
- Mixed 2p/4p training is represented with `training_format.rollout_groups[*].player_count` in Hydra config and per-group rollout state in `src/jax_train.py`.
- Rollout batches are concatenated before PPO updates, so format identity needs to survive collection and concatenation if the value head routes by format.
- Existing feature encoders expose player-count context, but relying on feature inference would make value-head routing implicit and fragile.

## Decisions

- Introduce a value-head module as a sibling composition point to the existing encoder and decoder modules.
- Implement the current shared critic as the default value-head module so behavior remains compatible by default.
- Add a format-aware value-head strategy that routes examples to separate 2p and 4p critic heads using an explicit format signal.
- Prefer the existing policy composition style over a broad global plugin registry. If a registry-like surface is needed, keep it narrow: builder/factory logic maps Hydra config to concrete value-head modules.
- Preserve format identity explicitly in rollout/PPO data rather than inferring it from observations.

## Proposed Config Surface

Add model-level value-head configuration, for example:

```yaml
model:
  value_head: shared
```

Initial supported strategies:

- `shared`: current behavior, one critic MLP for all formats.
- `format_routed`: separate 2p and 4p critic heads selected by format id/player count.

Future strategies can be added behind the same composition point without changing encoder/decoder APIs.

## Constraints

- Keep `JaxPolicyOutput.value` shape as `(batch,)`.
- Keep shared-head behavior available and compatible as the default implementation path.
- Thread the format signal through the JAX rollout and PPO path in a way that survives `concatenate_transition_batches`.
- Avoid broad architectural rewrites outside `src/jax_policy.py`, `src/jax_ppo.py`, `src/jax_train.py`, config schema/defaults, and focused tests unless implementation reveals a direct need.
- Treat checkpoint compatibility carefully: switching from `shared` to `format_routed` changes parameter structure and should be explicit through config.

## Non-Goals

- Do not change action decoder behavior.
- Do not alter reward semantics or return calculation.
- Do not make separate policies per format; only the value head is format-aware.
- Do not rely solely on existing observation features to infer 2p vs 4p routing.
- Do not introduce a wide third-party-style plugin framework unless the local module composition pattern requires it.

## Acceptance Criteria

- `ComposablePlanetPolicy` accepts or constructs a value-head module in the same spirit as the encoder/decoder modules.
- The default/shared value-head path preserves existing output shapes and deterministic behavior.
- A `format_routed` value head can produce different values for examples marked as 2p vs 4p while preserving `(batch,)` output shape.
- Mixed 2p/4p rollout collection preserves a format id or player-count signal through batch concatenation and PPO update inputs.
- Hydra config can select the value-head strategy for experiments.
- Focused tests cover policy shape/routing behavior, mixed rollout/PPO smoke behavior, and config/default validation.

## Suggested Verification

- Run focused policy tests in `tests/test_jax_policy.py` for composable value-head shape and routing.
- Run the most relevant mixed-rollout test from `tests/test_jax_ppo.py`, especially the 2p/4p rollout group test, rather than the whole file unless needed.
- Run config/default checks if schema or generated defaults change: `uv run python scripts/generate_default_cfg.py --check` and focused config/default tests if present.

## Interview Transcript

1. Desired behavior: user chose configurable strategy for experiments.
2. Format signal plumbing: user was open to suggestions; recommendation is explicit transition-batch preservation with backward-compatible handling.
3. Success criteria: user chose policy tests, mixed rollout PPO smoke test, and Hydra configurability.
4. Strategy scope: user requested a full composable/plugin-like approach.
5. Assumption challenge: user refined the scope to match existing modular encoder/decoder composition if possible; otherwise build the broader option.
6. Default/compatibility: user chose always using composable value heads internally.

## Ontology

- Value head: critic component that maps `EncoderOutput.value_input` plus optional format signal to scalar state values.
- Shared value head: current single critic MLP behavior packaged as a module.
- Format-routed value head: critic component with separate 2p and 4p sub-heads selected per example.
- Format signal: explicit rollout/PPO field representing 2-player vs 4-player game format.
- Rollout group: configured collector group in `training_format.rollout_groups` with `player_count` and `num_envs`.

## Ambiguity Score

- Goal clarity: 18%
- Constraint clarity: 20%
- Success criteria: 20%
- Context clarity: 15%
- Weighted overall ambiguity: 18%
