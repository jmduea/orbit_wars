#!/usr/bin/env python3
"""Colab VM entrypoint for Orbit Wars remote training workers."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (SCRIPT_DIR, SCRIPT_DIR.parent):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from src.orchestration.remote_worker import run_colab_worker  # noqa: E402


def main() -> None:
    """Load packaged env and run the shared Colab worker bootstrap."""

    run_colab_worker()


if __name__ == "__main__":
    main()
