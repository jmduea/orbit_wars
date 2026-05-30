#!/usr/bin/env python3
"""Thin entrypoint for the Kaggle training runner."""

from __future__ import annotations

from src.cli.kaggle_runner import run

if __name__ == "__main__":
    raise SystemExit(run())
