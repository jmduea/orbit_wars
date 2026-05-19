setup:
	uv sync --group dev

.PHONY: setup cfg-default cfg-default-check test

cfg-default:
	uv run python scripts/generate_default_cfg.py

cfg-default-check:
	uv run python scripts/generate_default_cfg.py --check

test:
	uv run --group dev pytest
