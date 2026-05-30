# RALPLAN: Composable Multi-Value Head

## Decision

Implement a narrow value-head composition point in the JAX policy stack. Add `SharedValueHead` and `FormatRoutedValueHead` modules, a small config-driven factory, and explicit player-count plumbing through rollout/PPO data so mixed 2p/4p training can route values to separate critic heads.

## Decision Drivers

- Preserve the existing encoder/decoder composition style in `ComposablePlanetPolicy`.
- Keep the current shared critic behavior available as the default value-head strategy.
- Make 2p/4p routing explicit and reliable after mixed rollout batches are concatenated.
- Avoid a broad plugin registry until more value-head variants exist.
- Prioritize clean module composition over old checkpoint parameter compatibility.

## Chosen Approach

Use a narrow value-head factory alongside the existing policy builder:

- `model.value_head: shared` selects the current shared critic behavior packaged as a module.
- `model.value_head: format_routed` selects separate 2p and 4p critic heads using an explicit player-count signal.
- `ComposablePlanetPolicy` composes `encoder_module`, `decoder_module`, and `value_head_module`.
- PPO transition data carries a player-count field so format identity survives rollout collection, concatenation, minibatching, and policy re-evaluation.

## Alternatives Considered

- Full registry-style plugin surface: more extensible, but too much framework for two initial value-head strategies and duplicates the local builder style.
- Inline branch inside `ComposablePlanetPolicy`: smallest edit, but fails the composability goal and makes future strategies harder.
- Observation-inferred routing: avoids a transition schema change, but makes routing implicit and fragile after mixed-format concatenation.

## Implementation Plan

1. Add `value_head: str = "shared"` to `ModelConfig` and validate allowed values `shared` and `format_routed`.
2. Regenerate `default_cfg.yaml` after schema changes.
3. Add `SharedValueHead`, `FormatRoutedValueHead`, and `build_value_head(cfg)` in `src/jax_policy.py`.
4. Extend `ComposablePlanetPolicy` to accept a `value_head_module` sibling to encoder/decoder modules.
5. Thread an optional `player_count` or `format_id` argument through policy `__call__` methods and all `policy.apply` call sites that compute values.
6. Prefer representing format as player count values `2` and `4`; validate unsupported values clearly for `format_routed`.
7. Extend `JaxTransitionBatch` with the explicit player-count field shaped like value rows.
8. Populate the transition field in rollout collection from static `cfg.env.player_count`.
9. Ensure transition concatenation preserves the new field through the existing tree concatenation path.
10. Flatten/minibatch the field in PPO update and pass it into policy application.
11. Add focused tests for shared value-head compatibility, format-routed differentiation, mixed rollout format preservation, and PPO update consumption.

## Required Test Evidence

- Policy tests prove both value-head strategies preserve `JaxPolicyOutput.value.shape == (batch,)`.
- A format-routed policy test proves identical encoded inputs can route to different 2p/4p critic parameters based on the explicit signal.
- Mixed 2p/4p rollout group test proves concatenated transitions retain both player-count values.
- PPO update smoke path proves minibatched policy re-evaluation accepts the field without value-loss shape regressions.
- Config/default check proves `model.value_head` is available and defaults to `shared`.

## Suggested Commands

```bash
rtk uv run --group dev pytest tests/test_jax_policy.py
rtk uv run --group dev pytest tests/test_jax_ppo.py::test_jax_rollout_groups_collect_two_and_four_player_formats_under_jit
rtk uv run python scripts/generate_default_cfg.py --check
```

## Consequences

- Old model parameter checkpoints may not load after the policy module hierarchy changes; this is accepted for this feature.
- Config selection remains explicit, so `format_routed` parameter shape differences are opt-in.
- The transition schema grows by one field, but it resolves the core mixed-format routing problem directly.

## Consensus

- Planner: recommended narrow value-head factory.
- Architect: approved with explicit format plumbing and checkpoint stance.
- User decision: clean composition is more important than old model parameter compatibility.
- Critic: approved with required attention to all policy apply paths and transition plumbing.
