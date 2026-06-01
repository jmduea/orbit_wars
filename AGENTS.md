# Orbit Wars Agent Guide

Hydra-first Python 3.12 JAX PPO project. See `docs/ONBOARDING.md` for architecture depth, `docs/AGENT_CAPABILITIES.md` for task prompts, and `docs/CURSOR.md` for plugin setup.

## Commands

```bash
uv sync --group dev
make help                                         # Makefile targets + agent-context
make agent-context                              # JSON: preflight, roadmap, recent runs
make test                                         # default verification (= test-fast)
make test-fast                                    # alias for daily CPU loop
make preflight-calibrate                          # refresh thresholds from preflight_calibrate_* campaigns
make preflight-learn-proof                        # Gates 2–3 JAX trend checks
uv run ow benchmark learn-proof --eval-checkpoint ... --baselines noop  # Gate 5 tournament win proof
uv run ow train training.total_updates=1000
uv run ow train print_resolved_config=true
uv run ow train kaggle
uv run ow eval tournament --checkpoint ... --campaign my_campaign
uv run ow eval package --checkpoint ... --validate-docker
uv run ow runs list
uv run ow runs show --run outputs/campaigns/<c>/runs/<run_id>
uv run ow eval status --run outputs/campaigns/<c>/runs/<run_id>
uv run ow eval worker --run outputs/campaigns/<c>/runs/<run_id> [--verbose]
uv run ow eval submit --checkpoint outputs/.../jax_ckpt_last.pkl
uv run ow train ... artifacts=hybrid_promotion   # strict promotion: docker + tournament async
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

- **Default:** `make test` / `make test-fast` — serial CPU (`-m "not slow and not jax and not sweep"`).
- **Daily (parallel wall clock):** `make test-daily` — `test-fast` ∥ `test-jax`; or `make test-daily-parallel` with xdist.
- **Kaggle env parity:** `make test-kaggle-parity` — `test_jax_env_parity.py` in the jax tier (not slow).
- **Pre-merge:** `make test-premerge` = `test-daily` + `test-slow` (`slow and not sweep`). Before release: `make test-sweep`.
- **Parallel CPU tiers:** `make test-fast-parallel` / `make test-jax-parallel` only (sets `ORBIT_WARS_PYTEST_XDIST=1`; never on slow/sweep/GPU).
- **Never:** bare `pytest -n` without the Makefile targets; never xdist on slow/sweep; `make test-full` only when user asks (~15 min WSL2).

## Key invariants

- Feature path: obs → `encode_turn` (`TurnBatch`); golden tests in `tests/test_feature_encoding_golden.py`.
- Rollout metrics: merge with `_merge_metric_dicts` inside `jax.lax.scan`; finalize rates after the scan.
- Priorities: human `docs/ROADMAP.md` (no automated validation).
- **Verification thresholds:** derive pass/fail bars from measured calibration or baseline runs (`docs/benchmarks/preflight-calibration.json`, `make preflight-calibrate`); never invent round numbers or relax a threshold until a run passes — that is not verification.
- **Metric gates:** before gating on a training metric, confirm its denominator and what “chance” means for that opponent and reward mode (e.g. `overall_win_rate` vs `noop_only` is not ~50%; `episode_reward_mean` under `binary_win` is not win rate; self-play ~50% is not a learning signal).
- **Seed scheduler:** default `training.reseed_every_updates=-1` auto-scales to `max(25, total_updates // 10)`; run `ow benchmark calibrate-seed-scheduler` before changing. Reseed resets rollout env state, not just the PRNG key.

## Learned User Preferences

- Prefer unified v2-only feature encoding; remove legacy v1 paths rather than parallel encoders.
- Favor full rework over incremental shims when simplifying encoding, rollout, training modules, or any refactor — no backward-compat re-export modules, `_underscore` aliases, or parallel APIs for module moves/renames in a solo project; update call sites instead.
- Daily dev loop: `make test-fast` or a domain Makefile target — not bare full `pytest` and not slow/JAX-compile smokes unless explicitly requested.
- Use `make test-fast-parallel` / `make test-jax-parallel` for CPU xdist only; never xdist on slow/sweep or with GPU pytest.
- Commit verified work locally without asking; **do not push** to remote unless the user explicitly requests it.
- Do not start test runs when another agent/session is already running tests, or when the user says verification is already done — check the terminals folder first.
- Same terminals check before starting expensive GPU training runs (calibration sweeps, long `ow train`, Gate 5 tournaments); parallel jobs contend on one GPU.
- New train/eval/benchmark capabilities belong in the `ow` CLI (`src/cli/<module>.py` + dispatch in `src/cli/__init__.py`), not standalone `scripts/*.py`; defer heavy imports (JAX) until command execution, not module load.
- Parallel multi-agent work: at most one full pytest/Makefile suite repo-wide; executor agents run targeted tests only; coordinator runs `make test-fast` after integration.
- Prefer `task=shield_cheap` or `task=shield_off` for training experiments; avoid `shield_tiered` unless explicitly requested.
- Hydra/config tests: prefer composition and required-key validation over asserting full resolved configs match hardcoded snapshots.
- After major training-loop or checkpoint refactors, verify with short train smoke then a ~100-update benchmark; unit tests alone may miss regressions.
- When the user attaches PR diff context or cites a PR number, fetch with `gh pr view` / `gh pr diff` — do not rely on inline diff blobs alone.
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
- **Hybrid promotion:** `artifacts=hybrid_promotion` sets `promotion.strategy=hybrid`, `tournament.enabled=true`, `checkpoint_eval_async=true`. Scalar metric improvements queue a composite `checkpoint_eval` worker job (Docker validation → tournament → promote). Training never writes promoted manifests on metrics alone under hybrid. Artifacts: `evaluations/checkpoint_eval_u<update>_<id>/{docker_validation,tournament}/`. Profile requires Docker on the worker host.
- **Train run layout:** `ow train` with `output.campaign=<name>` writes under `outputs/campaigns/<campaign>/runs/<run_id>/` — update metrics in `logs/*_jax.jsonl` (not `metrics.jsonl` at run root), checkpoints in `checkpoints/jax_ckpt_*.pkl`, artifact jobs in `queue/optional_jobs/`, evaluation outputs in `evaluations/`.
- **Curriculum pre-flight:** `curriculum.enabled=true` requires `opponents.snapshot.pool_size > 0` and `opponents.snapshot.interval_updates > 0`; use `curriculum=off` for isolated train smokes that don't exercise historical opponents.
- **Parametric edge catalog:** default `intercept_anchors` `[1.0, 3.0, 6.0]`; edge dim `6×N+7` (25 for default). Implemented in `src/features/catalog/edge.py` and `conf/task/base.yaml`; `docs/feature-encoding-v2.md` may still describe the older two-anchor layout.
- **Preflight gates:** Gates 2–4 read JAX learning-signal trend from `logs/*_jax.jsonl`; Gate 5 is tournament win proof on checkpoints via `kaggle_environments` (not Docker). Source of truth for thresholds: `docs/benchmarks/preflight-calibration.json`.
- **`ow benchmark` dispatch:** subcommands must be registered in `src/cli/__init__.py` (`case "benchmark"`), not only `build_parser()` in tests — verify with `uv run ow benchmark --help`.
