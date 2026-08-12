"""Product request/response schemas."""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from app.models import Product


def build_embedding_text(
    *,
    name: str,
    category: str,
    brand: str | None,
    description: str,
    attributes: dict[str, Any],
) -> str:
    """The single definition of what a product's text vector is built from.

    Ingest and re-indexing both call this, so a product re-embedded later lands in the
    same place it would have on first insert.
    """
    parts = [name, category, brand or "", description]
    extras = [f"{k}: {v}" for k, v in attributes.items()]
    return ". ".join(p for p in [*parts, *extras] if p).strip()


class ProductCreate(BaseModel):
    sku: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=255)
    description: str = ""
    category: str = Field(min_length=1, max_length=120)
    brand: str | None = Field(default=None, max_length=120)
    price: Decimal = Field(default=Decimal("0.00"), ge=0)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    attributes: dict[str, Any] = Field(default_factory=dict)

    def embedding_text(self) -> str:
        """The string that becomes the product's text vector."""
        return build_embedding_text(
            name=self.name,
            category=self.category,
            brand=self.brand,
            description=self.description,
            attributes=self.attributes,
        )


class ProductRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    sku: str
    name: str
    description: str
    category: str
    brand: str | None
    price: float
    currency: str
    image_url: str | None
    attributes: dict[str, Any]
    has_image: bool
    created_at: datetime

    @classmethod
    def from_model(cls, product: Product) -> Self:
        return cls(
            id=product.id,
            sku=product.sku,
            name=product.name,
            description=product.description,
            category=product.category,
            brand=product.brand,
            price=float(product.price),
            currency=product.currency,
            image_url=f"/images/{product.image_path}" if product.image_path else None,
            attributes=dict(product.attributes or {}),
            has_image=product.image_path is not None,
            created_at=product.created_at,
        )


class ProductList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ProductRead]
