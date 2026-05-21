setup:
	uv sync --group dev

test:
	uv run --group dev pytest
