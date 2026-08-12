"""Liveness and readiness probes."""

from typing import Any

from fastapi import APIRouter
from sqlalchemy import text

from app.api.deps import SessionDep, SettingsDep
from app.embeddings import available_providers, get_provider, provider_capabilities
from app.repositories import EmbeddingRepository, ProductRepository

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
async def ready(session: SessionDep, settings: SettingsDep) -> dict[str, Any]:
    """DB reachability, pgvector presence, and which embedding indexes exist."""
    await session.execute(text("SELECT 1"))
    extension = await session.execute(
        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    )
    pgvector_version = extension.scalar_one_or_none()

    products = ProductRepository(session)
    embeddings = EmbeddingRepository(session)
    coverage = await embeddings.coverage()
    default = get_provider()

    return {
        "status": "ok",
        "database": "reachable",
        "pgvector": pgvector_version or "missing",
        "default_provider": default.name,
        "embedding_dim": settings.embedding_dim,
        "cross_modal": default.supports_cross_modal,
        "default_text_weight": settings.default_text_weight,
        "default_min_score_ratio": settings.default_min_score_ratio,
        "configured_providers": available_providers(),
        "indexed_providers": coverage,
        # Per-provider capabilities, so a client can explain what an index can do
        # without needing the admin endpoints enabled.
        "provider_info": {
            name: {
                "configured": name in available_providers(),
                "indexed": name in coverage,
                "supports_images": provider_capabilities(name).supports_images,
                "cross_modal": provider_capabilities(name).supports_cross_modal,
            }
            for name in sorted(set(coverage) | set(available_providers()))
        },
        "products": await products.count(),
        "products_with_images": await products.count_with_images(),
        "admin_enabled": settings.admin_enabled,
    }
