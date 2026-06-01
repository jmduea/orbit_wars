# Orbit Wars

Orbit Wars is a Python 3.12 reinforcement-learning project managed with `uv` and launched through Hydra.

**Status:** [Roadmap](docs/ROADMAP.md) · [Open issues](https://github.com/jmduea/orbit_wars/issues?q=is%3Aopen)

The canonical training entrypoint composes responsibility-based config groups from `conf/`; see [conf/README.md](conf/README.md). Agent workflows: [docs/AGENT_CAPABILITIES.md](docs/AGENT_CAPABILITIES.md) and [AGENTS.md](AGENTS.md).

```bash
uv run ow train # Launch a training run with Hydra overrides
uv run ow make # Compose a sweep for W&B using composable configs
```

## Development

Install dependencies:

```bash
uv sync --group dev
```

Run tests:

```bash
make test              # CPU fast tier (~70s)
make test-daily        # test-fast ∥ test-jax (~75s wall)
make test-jax          # JAX light tier; includes Kaggle env parity
make test-kaggle-parity # JAX env mechanics (parity file)
make test-premerge     # test-daily + test-slow (no full sweep grid)
make test-sweep        # full W&B sweep grid (weekly / pre-release)
make test-fast-parallel # CPU xdist on fast tier only
make test-full         # all tests; serial — never bare pytest -n
```

For config work, start with:

```bash
make test-domain-config
```
