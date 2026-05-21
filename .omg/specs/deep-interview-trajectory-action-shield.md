# Deep Interview Spec: Trajectory Action Shield

Date: 2026-05-21
Ambiguity: 16%
Status: Pending approval

## Goal

Implement a hard trajectory action shield that prevents Orbit Wars agents from selecting or emitting launches forecast to hit the sun or leave the board before an acceptable planet collision.

The shield should use the same physics inputs available in observations, including `angular_velocity` and `initial_planets`, so moving planet positions can be predicted instead of approximated from current target geometry alone.

## Constraints

- Apply the shield to training and runtime behavior, not only packaged submission output.
- Preserve Python and JAX parity for candidate legality wherever practical.
- Preserve last-mile safety in every action emitter that turns policy choices into `[source_id, angle, ships]` moves.
- Keep Kaggle runtime within the existing 1 second action timeout.
- Make the forecast horizon configurable, with full remaining episode horizon as the default.
- Make hit semantics configurable:
  - Default: selected target must be hit before sun, bounds, or another planet.
  - Alternative: any non-friendly planet hit before hazard may count as safe.
- If no safe non-noop launch remains, no-op is valid and preferred over unsafe launch.

## Non-Goals

- Do not rely only on reward shaping to solve unsafe launches.
- Do not add a submission-only guard that hides training/runtime divergence.
- Do not weaken existing direct sun-crossing masks.
- Do not hand-edit generated default config output without regenerating it.

## Acceptance Criteria

- `GameState` or equivalent observation structures preserve enough data to forecast rotating planet positions from `initial_planets`, `angular_velocity`, and `step`.
- A shared trajectory predicate can forecast one proposed launch after ship bucket selection using fleet speed, launch offset, swept planet collision, sun segment intersection, and board bounds checks.
- Python candidate/action paths and JAX candidate/action paths agree on shield legality for representative static and rotating planet cases.
- Packaged Kaggle `main.py`, replay generation, and local opponent/action emitters skip unsafe moves and fall back to safe alternatives or no-op.
- Tests cover at least:
  - direct sun crossing remains blocked;
  - launch that misses a moving intended target and later hits sun is blocked;
  - launch that would leave board before any acceptable hit is blocked;
  - selected-target mode blocks unintended hits;
  - non-friendly mode permits enemy/neutral hits and blocks friendly hits;
  - all unsafe non-noops fall back to no-op;
  - generated submission template compiles with shield code.
- Training diagnostics expose blocked action counts and reasons so policy/shield drift can be measured.

## Assumptions Resolved

- Primary mitigation posture: hard safety shield.
- Scope: training and emitters, not emitter-only.
- Physics model: deterministic forward forecast using observation-provided rotation data.
- Hit semantics: configurable selected-target default with non-friendly alternative.
- Forecast horizon: configurable, full remaining episode horizon by default.
- Fallback: no-op rather than unsafe launch.

## Key Entities

- Trajectory shield: shared legality predicate for proposed launches.
- Forecast horizon: maximum simulated future steps for one proposed launch.
- Hit semantics: rule deciding which planet collision counts as safe.
- Hazard reason: sun collision, out-of-bounds, unintended hit, or no hit before horizon.
- Action emitter: code that converts policy target and ship bucket choices into Kaggle moves.
- Candidate mask: training/inference legality mask passed into policy logits.

## Relevant Code Areas

- `src/game_types.py`: observation parsing currently ignores rotation fields.
- `src/features.py`: Python candidate construction and direct sun mask.
- `src/jax_features.py`: JAX candidate construction and direct sun mask.
- `src/jax_env.py`: ground-truth fleet movement, planet rotation, sun, bounds, and swept-hit mechanics.
- `src/jax_ppo.py`: JAX action assembly after target and ship bucket selection.
- `src/replay.py`: checkpoint replay action emission.
- `src/opponents.py`: scripted/local action emitters.
- `scripts/validate_kaggle_docker_submission.py`: generated Kaggle `main.py` action emission and packaging tests.
- `conf/config.yaml` and `src/conf_schema.py`: shield configuration, if exposed through Hydra.

## Interview Transcript Summary

- User observed replay failures from sun-crossing collisions and out-of-bounds launches.
- Initial recommendation was action shielding plus training signal.
- User selected hard safety shield as the primary mitigation.
- User initially selected all action emitters, then rejected emitter-only parity drift.
- User clarified that observations include `angular_velocity` and `initial_planets`, enabling exact moving-planet forecasts.
- User selected configurable selected-target vs non-friendly hit semantics.
- User selected configurable horizon with full remaining horizon as the default.

## Open Implementation Questions

- Exact config names and defaults for shield enablement, hit semantics, horizon, and safety margin.
- Whether shield diagnostics should be part of PPO rollout metrics only, Docker validation output only, or both.
- Whether a small shared pure-Python shield should be mirrored in JAX, or whether JAX candidate masking should use a JAX-native vectorized equivalent from the start.
