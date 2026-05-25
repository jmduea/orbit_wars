"""Feature catalog assembly and slice metadata."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

from src.features.catalog._types import FeatureCatalogEntry, FeatureDefinition


class FeatureCatalog:
    """Ordered feature declarations with mechanical assembly."""

    def __init__(
        self,
        entries: Sequence[FeatureCatalogEntry],
        *,
        concat_axis: int = -1,
    ):
        self._entries = tuple(entries)
        self._concat_axis = concat_axis
        active = tuple(entry for entry in self._entries if entry.definition.active)
        if not active:
            raise ValueError("Feature catalog must contain at least one active entry")
        self._active_entries = active
        self._definitions = tuple(entry.definition for entry in active)
        self._base_slices = self._build_slices(self._definitions)

    @staticmethod
    def _build_slices(definitions: Sequence[FeatureDefinition]) -> dict[str, slice]:
        slices: dict[str, slice] = {}
        start = 0
        for definition in definitions:
            if definition.name in slices:
                raise ValueError(f"Duplicate feature name: {definition.name}")
            stop = start + definition.size
            slices[definition.name] = slice(start, stop)
            start = stop
        return slices

    @property
    def entries(self) -> tuple[FeatureCatalogEntry, ...]:
        return self._entries

    @property
    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return self._definitions

    @property
    def base_dim(self) -> int:
        return sum(definition.size for definition in self._definitions)

    def base_slice(self, feature_name: str) -> slice:
        try:
            return self._base_slices[feature_name]
        except KeyError as exc:
            raise ValueError(
                f"Feature '{feature_name}' is not active in the catalog"
            ) from exc

    def assemble(self, context: object) -> jnp.ndarray:
        """Concatenate active feature compute outputs in catalog order."""

        parts = tuple(entry.compute(context) for entry in self._active_entries)
        return jnp.concatenate(parts, axis=self._concat_axis)
