"""Turning outside data into products, from two directions.

Both paths converge on the same `ProductDraft`, so ingestion, de-duplication, image
handling and embedding are written once:

* **offline** — the admin supplies the product records and images are *drawn* locally
  by `app.seeding.images`. No network at all.
* **remote** — a JSON feed is fetched and its fields mapped onto our shape, with product
  photos downloaded from the URLs it gives.

Field mapping is data, not code, so a new feed shape needs no new Python. Paths are
dotted (`meta.brand`); when a path lands on a list the first element is taken, which is
how `images: [...]` becomes a single photo.
"""

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from app.core.errors import ValidationError
from app.schemas.importing import FieldMapping

SKU_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


#: Ready-made mappings for feeds people actually reach for first.
PRESETS: dict[str, dict[str, Any]] = {
    "dummyjson": {
        "url": "https://dummyjson.com/products?limit=100",
        "mapping": FieldMapping().model_dump(),
        "label": "DummyJSON — 100 products with photos",
    },
    "fakestore": {
        "url": "https://fakestoreapi.com/products",
        "mapping": FieldMapping(
            items_path="", sku="id", name="title", brand="", image_url="image"
        ).model_dump(),
        "label": "Fake Store API — 20 products with photos",
    },
}


@dataclass(slots=True)
class ProductDraft:
    """A product on its way in, from either source."""

    sku: str
    name: str
    description: str = ""
    category: str = "uncategorised"
    brand: str | None = None
    price: Decimal = Decimal("0.00")
    attributes: dict[str, str] = field(default_factory=dict)
    image_url: str | None = None
    #: Set for offline imports where the image is drawn rather than downloaded.
    generate_image: bool = False


def _dig(record: Any, path: str) -> Any:
    """Follow a dotted path, taking the first element of any list on the way."""
    if not path:
        return None
    current = record
    for part in path.split("."):
        if isinstance(current, list):
            current = current[0] if current else None
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    if isinstance(current, list):
        current = current[0] if current else None
    return current


def _as_decimal(value: Any) -> Decimal:
    if value is None:
        return Decimal("0.00")
    try:
        return Decimal(str(value)).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return Decimal("0.00")


def make_sku(prefix: str, raw: Any, fallback_index: int) -> str:
    base = SKU_SAFE.sub("-", str(raw or f"item{fallback_index}")).strip("-")
    return f"{prefix}-{base}"[:64] or f"{prefix}-{fallback_index}"


def extract_items(body: Any, mapping: FieldMapping) -> list[dict[str, Any]]:
    """Pull the record array out of a fetched payload."""
    items = body if not mapping.items_path else _dig_container(body, mapping.items_path)
    if not isinstance(items, list):
        where = mapping.items_path or "the response body"
        raise ValidationError(
            f"expected a JSON array at {where!r}; got {type(items).__name__}. "
            f"Adjust items_path to point at the list of products."
        )
    return [i for i in items if isinstance(i, dict)]


def _dig_container(record: Any, path: str) -> Any:
    """Like `_dig` but preserves lists — used to locate the records array itself."""
    current = record
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def drafts_from_remote(
    body: Any, mapping: FieldMapping, *, prefix: str, limit: int
) -> list[ProductDraft]:
    """Map a fetched feed onto drafts, skipping records with no usable name."""
    drafts: list[ProductDraft] = []
    for index, record in enumerate(extract_items(body, mapping)):
        name = _dig(record, mapping.name)
        if not name:
            continue
        image = _dig(record, mapping.image_url)
        drafts.append(
            ProductDraft(
                sku=make_sku(prefix, _dig(record, mapping.sku), index),
                name=str(name)[:255],
                description=str(_dig(record, mapping.description) or "")[:4000],
                category=str(_dig(record, mapping.category) or "uncategorised")[:120],
                brand=(
                    str(_dig(record, mapping.brand))[:120] if _dig(record, mapping.brand) else None
                ),
                price=_as_decimal(_dig(record, mapping.price)),
                attributes={"source": prefix},
                image_url=str(image) if image else None,
            )
        )
        if len(drafts) >= limit:
            break
    if not drafts:
        raise ValidationError(
            "no usable products found — check the field mapping, especially `name`"
        )
    return drafts
