"""Catalogue endpoints. Products are ingested as multipart so a photo can ride along."""

import json
import uuid
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, File, Form, Query, UploadFile, status
from pydantic import ValidationError as PydanticValidationError

from app.api.deps import CatalogDep, FiltersDep, read_upload
from app.core.errors import ValidationError
from app.schemas import ProductCreate, ProductList, ProductRead

router = APIRouter(prefix="/products", tags=["products"])


@router.post("", response_model=ProductRead, status_code=status.HTTP_201_CREATED)
async def create_product(
    service: CatalogDep,
    sku: Annotated[str, Form()],
    name: Annotated[str, Form()],
    category: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
    brand: Annotated[str | None, Form()] = None,
    price: Annotated[float, Form()] = 0.0,
    currency: Annotated[str, Form()] = "USD",
    attributes: Annotated[str, Form(description="JSON object of extra attributes")] = "{}",
    image: Annotated[UploadFile | None, File()] = None,
) -> ProductRead:
    """Create a product and embed its text and (optional) photo."""
    try:
        parsed_attributes = json.loads(attributes or "{}")
    except json.JSONDecodeError as exc:
        raise ValidationError(f"attributes must be a JSON object: {exc}") from exc
    if not isinstance(parsed_attributes, dict):
        raise ValidationError("attributes must be a JSON object")

    try:
        payload = ProductCreate(
            sku=sku,
            name=name,
            description=description,
            category=category,
            brand=brand,
            price=Decimal(str(price)),
            currency=currency,
            attributes=parsed_attributes,
        )
    except PydanticValidationError as exc:
        raise ValidationError(str(exc)) from exc

    product = await service.create(payload, image=await read_upload(image))
    return ProductRead.from_model(product)


@router.get("", response_model=ProductList)
async def list_products(
    service: CatalogDep,
    filters: FiltersDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ProductList:
    items, total = await service.list_products(limit=limit, offset=offset, filters=filters)
    return ProductList(
        total=total,
        limit=limit,
        offset=offset,
        items=[ProductRead.from_model(p) for p in items],
    )


@router.get("/{product_id}", response_model=ProductRead)
async def get_product(product_id: uuid.UUID, service: CatalogDep) -> ProductRead:
    return ProductRead.from_model(await service.get_or_404(product_id))


@router.put("/{product_id}/image", response_model=ProductRead)
async def replace_product_image(
    product_id: uuid.UUID,
    service: CatalogDep,
    image: Annotated[UploadFile, File()],
) -> ProductRead:
    """Attach or replace a photo, re-embedding the image and fused vectors."""
    payload = await read_upload(image)
    if payload is None:
        raise ValidationError("an image file is required")
    return ProductRead.from_model(await service.attach_image(product_id, payload))


@router.delete("/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_product(product_id: uuid.UUID, service: CatalogDep) -> None:
    await service.delete(product_id)
