# Path ownership (multi-agent)

Default touch surfaces by GitHub label `area:*`. Declare narrower paths in `roadmap.py claim --path`.

| Area label | Typical paths |
|------------|----------------|
| `area:kaggle` | `src/orchestration/`, `scripts/kaggle_*`, `tests/test_kaggle_*`, `docs/kaggle*.md` |
| `area:train` | `src/jax/`, `src/training/`, `tests/test_jax_*`, `tests/test_curriculum.py` |
| `area:config` | `conf/`, `src/config/`, `tests/test_config_*` |
| `area:features` | `src/features/`, `src/jax/features.py`, `tests/test_feature_*` |
| `area:infra` | `scripts/`, `.github/workflows/`, `mcp-server/`, `Makefile` |

**Rules:** one active `claim` per issue; overlapping paths across claims are rejected. One branch per issue (`issue/NN-slug`) in a dedicated git worktree (`worktrees/issue-NN/`).

**Parallel agents**

| Variable | Purpose |
|----------|---------|
| `ORBIT_WARS_AGENT_ID` | Unique owner per worker (e.g. `cursor-issue-102`) |
| `ORBIT_WARS_ISSUE_ID` | Disambiguates branch guard when one owner has multiple claims |
| `ORBIT_WARS_BRANCH_ENFORCE` | `1` (default) blocks `src/conf/tests` edits on `main` when claim has a branch |

```bash
export ORBIT_WARS_ISSUE_ID=102 ORBIT_WARS_AGENT_ID=cursor-issue-102
uv run python scripts/roadmap.py claim --issue 102 --path tests/... --setup-worktree
# cd worktrees/issue-102 && implement
```
