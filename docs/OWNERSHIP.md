# Path ownership (multi-agent)

Default touch surfaces by GitHub label `area:*`. Declare narrower paths in `roadmap.py claim --path`.

| Area label | Typical paths |
|------------|----------------|
| `area:kaggle` | `src/orchestration/`, `scripts/kaggle_*`, `tests/test_kaggle_*`, `docs/kaggle*.md` |
| `area:train` | `src/jax/`, `src/training/`, `tests/test_jax_*`, `tests/test_curriculum.py` |
| `area:config` | `conf/`, `src/config/`, `tests/test_config_*` |
| `area:features` | `src/features/`, `src/jax/features.py`, `tests/test_feature_*` |
| `area:infra` | `scripts/`, `.github/workflows/`, `mcp-server/`, `Makefile` |

**Rules:** one active `claim` per issue; overlapping paths across claims are rejected. One branch per issue (`issue/NN-slug`).
