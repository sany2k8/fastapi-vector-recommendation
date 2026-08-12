"""Shared FastAPI dependencies."""

from decimal import Decimal
from typing import Annotated

from fastapi import Depends, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.db import get_session
from app.core.errors import ValidationError
from app.embeddings import EmbeddingProvider, get_provider
from app.repositories import CatalogFilters
from app.services import (
    CatalogService,
    InteractionService,
    RecommendationService,
    SearchService,
)
from app.services.jobs import JobRegistry, get_job_registry

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
RegistryDep = Annotated[JobRegistry, Depends(get_job_registry)]


def resolve_provider(
    provider: Annotated[
        str | None,
        Query(description="Embedding index to search. Defaults to EMBEDDING_PROVIDER."),
    ] = None,
) -> EmbeddingProvider:
    """Pick the embedding index for this request.

    Several providers can be indexed at once, so the caller may name one; unknown or
    unconfigured names fail loudly rather than silently falling back to the default,
    which would make an A/B comparison quietly meaningless.
    """
    return get_provider(provider)


ProviderDep = Annotated[EmbeddingProvider, Depends(resolve_provider)]
DefaultProviderDep = Annotated[EmbeddingProvider, Depends(get_provider)]


def get_catalog_service(session: SessionDep, provider: DefaultProviderDep) -> CatalogService:
    return CatalogService(session, provider)


def get_search_service(session: SessionDep, provider: ProviderDep) -> SearchService:
    return SearchService(session, provider)


def get_recommendation_service(session: SessionDep, provider: ProviderDep) -> RecommendationService:
    return RecommendationService(session, provider)


def get_interaction_service(session: SessionDep) -> InteractionService:
    return InteractionService(session)


CatalogDep = Annotated[CatalogService, Depends(get_catalog_service)]
SearchDep = Annotated[SearchService, Depends(get_search_service)]
RecommendationDep = Annotated[RecommendationService, Depends(get_recommendation_service)]
InteractionDep = Annotated[InteractionService, Depends(get_interaction_service)]


async def read_upload(upload: UploadFile | None) -> bytes | None:
    """Read an optional multipart file into memory, enforcing the configured size cap."""
    if upload is None:
        return None
    settings = get_settings()
    payload = await upload.read()
    if len(payload) > settings.max_image_bytes:
        raise ValidationError(
            f"image is {len(payload)} bytes; the limit is {settings.max_image_bytes}"
        )
    return payload or None


def build_filters(
    category: str | None = None,
    brand: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
) -> CatalogFilters:
    return CatalogFilters(
        category=category,
        brand=brand,
        min_price=None if min_price is None else Decimal(str(min_price)),
        max_price=None if max_price is None else Decimal(str(max_price)),
    )


FiltersDep = Annotated[CatalogFilters, Depends(build_filters)]


def require_admin(settings: SettingsDep) -> None:
    """Gate the destructive admin endpoints behind an explicit setting."""
    if not settings.admin_enabled:
        raise ValidationError("admin endpoints are disabled (set ADMIN_ENABLED=true to allow)")


AdminGuard = Depends(require_admin)
