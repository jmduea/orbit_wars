#!/usr/bin/env python3
"""Deprecated shim — use ``scripts/kaggle_runner.py`` or ``ow train kaggle``."""

from __future__ import annotations

import sys
import warnings

from src.cli.kaggle_runner import run

warnings.warn(
    "scripts/kaggle_wandb_population.py is deprecated; use "
    "scripts/kaggle_runner.py or `uv run ow train kaggle` instead.",
    DeprecationWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    raise SystemExit(run())
