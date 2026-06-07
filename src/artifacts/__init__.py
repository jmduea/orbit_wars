"""Artifact pipeline, checkpoints, tournament eval, and promotion helpers.

Import submodules directly (for example ``from src.artifacts.pipeline import ...``)
instead of relying on eager re-exports here — keeps lightweight CLI paths such as
``ow eval status`` from pulling replay/tournament/JAX subgraphs at package init.
"""
