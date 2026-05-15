# orbit_wars

## Reinforcement learning tutorial implementation

The Orbit Wars reinforcement-learning implementation that was previously generated from
`orbit-wars-reinforcement-learning-tutorial.ipynb` now lives as versioned repository files:

- `default_cfg.yaml` for quick notebook/demo runs
- `configs/full_training.yaml` for longer reproducible training runs
- `src/`
- `eval_vs_sniper.py`
- `play_vs_sniper.py`

The notebook should be treated as a tutorial wrapper around this checked-in implementation.
For code changes, update the repository files first; do not treat notebook `%%writefile`
cells as the canonical source of the implementation.

## Dependency management

This repository uses [`uv`](https://docs.astral.sh/uv/) for Python dependency management.
Install the runtime dependencies into a local `.venv` with:

```bash
uv sync
```

Run the extracted package and scripts through `uv run`, for example:

```bash
uv run python -m src.train --config default_cfg.yaml
uv run python -m src.train --config configs/full_training.yaml
uv run python eval_vs_sniper.py --config default_cfg.yaml --deterministic
uv run python play_vs_sniper.py --config default_cfg.yaml --deterministic --output result.html
```
