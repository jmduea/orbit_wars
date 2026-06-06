# Session handoff: Phase 2 env-parity picks (2026-06-06)

**Start here** for the next session.

## Phase 1 complete

- **Admission passed** on throughput-anchor worktree (`admission_passed: true` — learning + throughput on one recipe).
- **Main** has unified `admission` gate (`make gate-admission`), PPO from `base.yaml` (not profile pins), resolved config on stderr, `--repo-root` worktree harness.
- **Parity** on anchor: `make test-kaggle-parity` PASS (15 tests).
- **Docs:** `docs/solutions/workflow-issues/cherry-pick-admission-gate-unified-learn-throughput.md`

## Phase 2 goal

Build **Kaggle mechanics parity** on `throughput-baseline-integration` using **granular file/hunk picks** — not whole commits. Do **not** run tier-2 e2e or full `make gate-admission` after every hunk.

## Worktrees

| Location | Role |
|----------|------|
| `~/projects/orbit_wars` (`main`) | Gate harness, manifest, docs |
| `~/projects/orbit_wars-throughput-anchor` | Admission-passed training code (`throughput-baseline`) |
| `~/projects/orbit_wars-integration` (create) | Cherry-pick target on `throughput-baseline-integration` |

```bash
cd ~/projects/orbit_wars
git worktree add ../orbit_wars-integration throughput-baseline-integration
cd ../orbit_wars-integration
make test-kaggle-parity   # baseline before first pick
```

## Pick order (human names — SHAs only in git blocks)

1. **Offline reference libs** — `src/game/planet_generation.py`, `src/game/comet_generation.py`, constants (zero hot-path risk).
2. **Encoding contract** — `encode_learner_turn` in `src/jax/features.py` if env picks need it.
3. **Env mechanics hunks** — rotation, launch validation, combat (no comet, no `pure_callback`).
4. **Comet + kaggle paths** — only together with **Train/kaggle env split** (never land the Comet parity mega-commit without the split that removes `pure_callback` from the default train path).
5. **Legacy comet mode** — throughput preservation (`env_parity_mode=legacy` / shield config).
6. **Parity tests + schema** — `tests/test_jax_env_parity.py`, `env_parity_mode` in schema.

**Never cherry-pick whole:** Comet parity mega-commit (kitchen-sink PR) — use `git cherry-pick -n` + `git add -p` / path-limited `git add`.

Named commits for git (oldest first when whole-file picking):

| Human name | SHA |
|------------|-----|
| Comet parity mega-commit | `33b56e2` |
| Env step refactor | `0cfd762` |
| Train/kaggle env split | `b11b9b0` |
| Legacy comet mode + A/B bench | `4ebe96e` |

## Fast gates per pick (~1–3 min, not 30 min)

```bash
make test-kaggle-parity
make test-jax-trace-hygiene
# optional:
uv run ow benchmark env-parity-ab --modes legacy,train --batch-size 32 --steps 32 --warmup 1 --repeats 2 --out /tmp/ep-ab.json
# 5-update smoke (validation preset) — compare to integration head baseline
```

**Milestone only:** `make gate-admission REPO_ROOT=<integration-worktree>`

## Reject if

- Parity or trace-hygiene fails
- `legacy` arm in env-parity-ab drops >10% vs pre-pick baseline
- Comet/callback code lands without Train/kaggle env split

Record each trial in `docs/benchmarks/cherry-pick-manifest.json` → `candidates[]` with `phase: env_parity`, human `id`, `pick_granularity`, `verdict`.

## Read first

1. `docs/solutions/workflow-issues/cherry-pick-admission-gate-unified-learn-throughput.md`
2. `docs/solutions/workflow-issues/nuclear-cherry-pick-manifest-baseline-integration.md`
3. `docs/benchmarks/cherry-pick-manifest.json`

## New session prompt (copy-paste)

```
Start here: docs/session-handoff/2026-06-06-phase2-env-parity-picks.md

Phase 1 admission is done (admission_passed on throughput-anchor). Begin Phase 2:
granular file/hunk env-parity cherry-picks onto throughput-baseline-integration.

Use integration worktree at ../orbit_wars-integration. First pick: offline game
reference libs. Fast gates per pick (parity + trace hygiene), not tier-2 e2e.
Update cherry-pick-manifest.json candidates[] as we go. Human-readable names
for commits in prose; SHAs only in git commands.
```
