from __future__ import annotations

from dataclasses import dataclass

from src.features.catalog._core import FeatureCatalog


@dataclass(frozen=True, slots=True)
class CatalogView:
    """History-aware slice view over a single ``FeatureCatalog``."""

    catalog: FeatureCatalog
    history_steps: int = 1

    def __post_init__(self) -> None:
        if self.history_steps < 1:
            object.__setattr__(self, "history_steps", 1)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(definition.name for definition in self.catalog.definitions)

    @property
    def base_dim(self) -> int:
        return self.catalog.base_dim

    @property
    def dim(self) -> int:
        return self.base_dim * self.history_steps

    @property
    def total_dim(self) -> int:
        return self.dim

    def base_slice(self, feature_name: str) -> slice:
        return self.catalog.base_slice(feature_name)

    def with_history(self, history_steps: int) -> CatalogView:
        return CatalogView(self.catalog, history_steps=history_steps)

    def frame_slice(self, frame: int = -1) -> slice:
        """Return the full slice for one history frame."""

        frame_index = self._normalize_frame(frame)
        start = frame_index * self.base_dim
        return slice(start, start + self.base_dim)

    def slice(self, feature_name: str, frame: int = -1) -> slice:
        """Return a feature slice in the history-expanded vector."""

        base = self.base_slice(feature_name)
        frame_index = self._normalize_frame(frame)
        offset = frame_index * self.base_dim
        return slice(offset + base.start, offset + base.stop)

    def history_slices(self, feature_name: str) -> tuple[slice, ...]:
        return tuple(
            self.slice(feature_name, frame=frame) for frame in range(self.history_steps)
        )

    def _normalize_frame(self, frame: int) -> int:
        if frame < 0:
            frame += self.history_steps
            if frame < 0 or frame >= self.history_steps:
                raise IndexError(
                    f"History frame {frame} out of range for {self.history_steps} frames"
                )
        return frame
