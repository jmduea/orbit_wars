# Orbit Wars Agent Guide

Hydra-first Python 3.12 JAX PPO project. See `docs/ONBOARDING.md` for architecture depth and `docs/CURSOR.md` for plugin setup.

## Commands

```bash
uv sync --group dev
make test-fast                                    # default verification
uv run ow train training.total_updates=1000
uv run ow train print_resolved_config=true
uv run ow train kaggle
```

## Layout

| Path | Role |
|------|------|
| `src/config/` | Dataclass schema + Hydra composition |
| `src/jax/` | Training loop, env, features, PPO, rollout |
| `src/features/` | Registry + extractor (planet-edge encoding) |
| `src/game/` | Reference Python game logic |
| `conf/` | Hydra responsibility groups |
| `tests/` | pytest; `@pytest.mark.slow` / `jax` for expensive tiers |

## Test tiers

- **Default:** `make test-fast` — serial, CPU-only (`-m "not slow and not jax"`).
- **Optional JAX:** `make test-jax` — only when user requests.
- **Pre-merge:** `make test` — full suite; ask before running on WSL2.
- **Never:** `pytest-xdist`, bare full `pytest` during iteration.

## Key invariants

- Feature path: obs → `encode_turn` (`TurnBatch`); golden tests in `tests/test_feature_encoding_golden.py`.
- Rollout metrics: merge with `_merge_metric_dicts` inside `jax.lax.scan`; finalize rates after the scan.
- Priorities: human `docs/ROADMAP.md` (no automated validation).

## Learned User Preferences

- Prefer unified v2-only feature encoding; remove legacy v1 paths rather than parallel encoders.
- Favor full rework over incremental shims when simplifying encoding, rollout, or training modules.
- Daily dev loop: `make test-fast` or a domain Makefile target — not bare full `pytest` and not slow/JAX-compile smokes unless explicitly requested.
- Never use `pytest-xdist` or parallel pytest workers on WSL2/CUDA hosts.
- Commit verified work locally without asking; **do not push** to remote unless the user explicitly requests it.
- Do not start test runs when another agent/session is already running tests, or when the user says verification is already done — check the terminals folder first.
- Parallel multi-agent work: at most one full pytest/Makefile suite repo-wide; executor agents run targeted tests only; coordinator runs `make test-fast` after integration.
- Prefer `task=shield_cheap` or `task=shield_off` for training experiments; avoid `shield_tiered` unless explicitly requested.
- Hydra/config tests: prefer composition and required-key validation over asserting full resolved configs match hardcoded snapshots.
- After major training-loop or checkpoint refactors, verify with short train smoke then a ~100-update benchmark; unit tests alone may miss regressions.
- `.audit/` and `.cursor/hooks/state/` are gitignored; keep audit trails and hook state local, not in commits.

## Learned Workspace Facts

- Canonical feature path: Kaggle/JAX obs → `encode_turn` (planet-edge `TurnBatch`); golden tests live in `tests/test_feature_encoding_golden.py`.
- JAX concerns are split: rollout in `src/jax/rollout/collect.py`, PPO in `src/jax/ppo_update.py`, learner shielded sampling in `src/jax/action_sampling.py`, shield in `src/jax/shield/*`, training loop in `src/jax/train/` (loop, rollout_groups, metrics, snapshots, checkpoint, telemetry, queue, state), opponent builders in `src/opponents/jax_actions/`.
- Trajectory shield: JAX paths in `src/jax/shield/*`; Python reference helpers in `src/game/shield.py` and `shield_config.py`.
- `OPPONENT_FAMILY_*` constants live in `src/opponents/constants.py`; import `src.opponents.pool` only for pool logic (`OPPONENT_FAMILY_IDS`, sampling helpers).
- Per-format timing metrics (`*_2p`/`*_4p`) emit only when telemetry `metric_groups.debug` is enabled; `average_placement_4p` stays on the default path.
- `model.normalize_observations` appears in model YAMLs but is not wired into JAX training; treat as dead config until implemented or removed.
- Hydra dataclass defaults in `src/config/schema.py` can differ from `conf/` YAML; verify with `uv run ow train print_resolved_config=true`.
- Understand-Anything scans honor `.understandignore` for excluding non-project adjacent paths.
- OMG agent orchestration retired; use Cursor plugins per `docs/CURSOR.md`; legacy OMG/MULTI_AGENT material is under `docs/archive/omg/`.
- `docs/ROADMAP.md` is human-only — no `scripts/roadmap.py` funnel or `tests/test_roadmap.py` enforcement.
- Planned encoding: parametric edge catalog with default `intercept_anchors` `[1.0, 3.0, 6.0]` (src audit phase 4; schema still `(1.0, 6.0)` until implemented).
