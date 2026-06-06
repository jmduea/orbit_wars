"""Benchmark implementations for ``ow benchmark`` and operator tooling.

CLI dispatch lives in ``src.cli.benchmark``; this package holds measurement,
calibration sweeps, and post-hoc extract helpers. Avoid importing JAX here.
"""
