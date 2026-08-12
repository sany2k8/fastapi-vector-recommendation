"""Data access for per-provider product vectors."""

import uuid
from typing import TypedDict

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings import Vector
from app.models import ProductEmbedding


class ProviderCoverage(TypedDict):
    products: int
    with_images: int
    model: str


class EmbeddingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert(
        self,
        *,
        product_id: uuid.UUID,
        provider: str,
        model: str,
        dim: int,
        text_embedding: Vector | None,
        image_embedding: Vector | None,
        fused_embedding: Vector | None,
    ) -> None:
        """Write this provider's vectors, leaving every other provider's row untouched.

        That is the whole point of the table: re-indexing with Jina must not destroy the
        local_hash vectors you were comparing against.
        """
        stmt = insert(ProductEmbedding).values(
            product_id=product_id,
            provider=provider,
            model=model,
            dim=dim,
            text_embedding=text_embedding,
            image_embedding=image_embedding,
            fused_embedding=fused_embedding,
        )
        await self.session.execute(
            stmt.on_conflict_do_update(
                index_elements=[ProductEmbedding.product_id, ProductEmbedding.provider],
                set_={
                    "model": stmt.excluded.model,
                    "dim": stmt.excluded.dim,
                    "text_embedding": stmt.excluded.text_embedding,
                    "image_embedding": stmt.excluded.image_embedding,
                    "fused_embedding": stmt.excluded.fused_embedding,
                    "updated_at": func.now(),
                },
            )
        )

    async def coverage(self) -> dict[str, ProviderCoverage]:
        """How many products each provider has indexed, and how many carry an image vector."""
        result = await self.session.execute(
            select(
                ProductEmbedding.provider,
                func.count(ProductEmbedding.product_id),
                func.count(ProductEmbedding.image_embedding),
                func.min(ProductEmbedding.model),
            ).group_by(ProductEmbedding.provider)
        )
        return {
            str(row[0]): ProviderCoverage(
                products=int(row[1]),
                with_images=int(row[2]),
                model=str(row[3] or ""),
            )
            for row in result.all()
        }

    async def indexed_providers(self) -> list[str]:
        """Providers that actually have vectors stored — what a UI can offer to search."""
        result = await self.session.execute(
            select(ProductEmbedding.provider).distinct().order_by(ProductEmbedding.provider)
        )
        return list(result.scalars().all())

    async def product_ids_for(self, provider: str) -> list[uuid.UUID]:
        """Products this provider has already indexed — the basis for resuming a job."""
        result = await self.session.execute(
            select(ProductEmbedding.product_id).where(ProductEmbedding.provider == provider)
        )
        return list(result.scalars().all())

    async def count_for(self, provider: str) -> int:
        result = await self.session.execute(
            select(func.count(ProductEmbedding.product_id)).where(
                ProductEmbedding.provider == provider
            )
        )
        return int(result.scalar_one())

    async def clear_provider(self, provider: str) -> int:
        """Drop one provider's index, returning how many rows went."""
        removed = await self.count_for(provider)
        await self.session.execute(
            delete(ProductEmbedding).where(ProductEmbedding.provider == provider)
        )
        return removed
