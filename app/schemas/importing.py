"""Field mapping for remote imports.

Deliberately a leaf module: it imports nothing from the rest of the app. Both
`app.schemas.admin` (for the request body) and `app.seeding.importers` (to apply it)
need this type, and putting it in the seeding package made the two packages import each
other in a cycle.
"""

from pydantic import BaseModel, Field


class FieldMapping(BaseModel):
    """Where to find each of our fields inside someone else's JSON.

    Values are dotted paths (`meta.brand`). When a path lands on a list the first element
    is taken, which is how a feed's `images: [...]` becomes a single product photo.
    Defaults match DummyJSON, the most common starting point.
    """

    items_path: str = Field(
        default="products",
        description="Dotted path to the array of records. Empty means the body is the array.",
    )
    sku: str = "id"
    name: str = "title"
    description: str = "description"
    category: str = "category"
    brand: str = "brand"
    price: str = "price"
    image_url: str = "thumbnail"
