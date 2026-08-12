"""Demo catalogue: the data, its procedurally drawn imagery, and the seeding routine.

This lives inside the app package (rather than in scripts/) so the admin API can drive
seeding from the UI. In a real deployment you would strip this package out along with
the admin endpoints — the catalogue would come from your product system instead.
"""

from app.seeding.catalog_data import CATALOG, COLORS, CatalogEntry
from app.seeding.images import render
from app.seeding.service import DEMO_USERS, SeedingService

__all__ = [
    "CATALOG",
    "COLORS",
    "DEMO_USERS",
    "CatalogEntry",
    "SeedingService",
    "render",
]
