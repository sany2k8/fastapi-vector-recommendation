"""Data access for user interaction events."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings import Vector
from app.models import Interaction, Product, ProductEmbedding


class InteractionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, interaction: Interaction) -> Interaction:
        self.session.add(interaction)
        await self.session.flush()
        return interaction

    async def recent_for_user(self, user_id: str, *, limit: int = 100) -> list[Interaction]:
        result = await self.session.execute(
            select(Interaction)
            .where(Interaction.user_id == user_id)
            .order_by(Interaction.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def recent_with_vectors(
        self, user_id: str, *, provider: str, limit: int = 100
    ) -> list[tuple[Interaction, Vector]]:
        """Interactions paired with the product's vector for `provider`, newest first.

        Products this provider has not indexed are skipped — an item with no vector in
        the space being searched cannot contribute to a profile built in that space.
        """
        result = await self.session.execute(
            select(Interaction, ProductEmbedding.fused_embedding)
            .join(Product, Product.id == Interaction.product_id)
            .join(ProductEmbedding, ProductEmbedding.product_id == Product.id)
            .where(
                Interaction.user_id == user_id,
                ProductEmbedding.provider == provider,
                ProductEmbedding.fused_embedding.is_not(None),
            )
            .order_by(Interaction.created_at.desc())
            .limit(limit)
        )
        return [(row[0], list(row[1])) for row in result.all()]

    async def interacted_product_ids(self, user_id: str) -> list[uuid.UUID]:
        result = await self.session.execute(
            select(Interaction.product_id).where(Interaction.user_id == user_id).distinct()
        )
        return list(result.scalars().all())
