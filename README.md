# Orbit Wars

Orbit Wars is a Python 3.12 reinforcement-learning project managed with `uv` and launched through Hydra.

**Status:** [Roadmap](docs/ROADMAP.md) · [Open issues](https://github.com/jmduea/orbit_wars/issues?q=is%3Aopen)

The canonical training entrypoint composes responsibility-based config groups from `conf/`, see the [README.md](conf/README.md) & [GUIDANCE.md](config/GUIDANCE.md) for more info.

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
make test-fast    # CPU-only daily loop (safe on WSL2)
make test-jax     # serial JAX subset when editing JAX code
make test         # full suite incl. slow tests; serial only — never use pytest -n
```

For config work, start with:

```bash
make test-domain-config
```
