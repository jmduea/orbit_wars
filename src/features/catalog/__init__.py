"""Single-source feature catalogs for planet, edge, and global groups."""

from src.features.catalog._core import FeatureCatalog
from src.features.catalog._types import FeatureDefinition
from src.features.catalog.edge import EDGE_FEATURE_CATALOG
from src.features.catalog.global_ import GLOBAL_FEATURE_CATALOG
from src.features.catalog.planet import PLANET_FEATURE_CATALOG

__all__ = [
    "EDGE_FEATURE_CATALOG",
    "FeatureCatalog",
    "FeatureDefinition",
    "GLOBAL_FEATURE_CATALOG",
    "PLANET_FEATURE_CATALOG",
]
