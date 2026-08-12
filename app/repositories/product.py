"""Data access for the product catalogue, including all vector SQL.

Retrieval is deliberately two-stage:

  1. ANN over `product_embeddings.fused_embedding` (partial HNSW per provider, cosine)
     to pull a candidate pool cheaply.
  2. Exact re-rank over `text_embedding` and `image_embedding` for those candidates, so
     the caller's own text/image weighting is applied without needing an index per
     weighting.

Every query is scoped to one `provider`. Vectors from different providers live in the
same column but in unrelated spaces, so the provider predicate is never optional.
"""

import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings import Vector
from app.models import Interaction, Product, ProductEmbedding

VectorColumn = Literal["fused", "text", "image"]

_VECTOR_COLUMNS = {
    "fused": ProductEmbedding.fused_embedding,
    "text": ProductEmbedding.text_embedding,
    "image": ProductEmbedding.image_embedding,
}


@dataclass(slots=True)
class CatalogFilters:
    """Structured (non-vector) predicates applied before similarity ranking."""

    category: str | None = None
    brand: str | None = None
    min_price: Decimal | None = None
    max_price: Decimal | None = None
    exclude_ids: list[uuid.UUID] = field(default_factory=list)
    require_image: bool = False


@dataclass(slots=True)
class ScoredProduct:
    """A candidate plus its per-modality similarities (None when that vector is absent).

    `fused_vector` rides along because MMR needs candidate-to-candidate similarity;
    fetching it here keeps that a single query instead of one per candidate.
    """

    product: Product
    text_similarity: float | None
    image_similarity: float | None
    fused_vector: Vector | None


class ProductRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # --- CRUD ---------------------------------------------------------------
    async def add(self, product: Product) -> Product:
        self.session.add(product)
        await self.session.flush()
        return product

    async def get(self, product_id: uuid.UUID) -> Product | None:
        return await self.session.get(Product, product_id)

    async def get_by_sku(self, sku: str) -> Product | None:
        result = await self.session.execute(select(Product).where(Product.sku == sku))
        return result.scalar_one_or_none()

    async def list_products(
        self, *, limit: int = 50, offset: int = 0, filters: CatalogFilters | None = None
    ) -> list[Product]:
        stmt = self._apply_filters(select(Product), filters or CatalogFilters())
        stmt = stmt.order_by(Product.created_at.desc()).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_all(self) -> list[Product]:
        """Every product, oldest first — a stable order for batched re-indexing."""
        result = await self.session.execute(
            select(Product).order_by(Product.created_at, Product.id)
        )
        return list(result.scalars().all())

    async def count(self, filters: CatalogFilters | None = None) -> int:
        stmt = self._apply_filters(select(func.count(Product.id)), filters or CatalogFilters())
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def count_with_images(self) -> int:
        result = await self.session.execute(
            select(func.count(Product.id)).where(Product.image_path.is_not(None))
        )
        return int(result.scalar_one())

    async def delete(self, product: Product) -> None:
        await self.session.delete(product)

    # --- vector retrieval ---------------------------------------------------
    async def ann_candidates(
        self,
        query_vector: Vector,
        *,
        provider: str,
        limit: int,
        filters: CatalogFilters | None = None,
        column: VectorColumn = "fused",
    ) -> list[uuid.UUID]:
        """Stage 1: nearest neighbours on one of the stored vector columns.

        `fused` is the indexed path (partial HNSW per provider) and the default.
        `text`/`image` exist for providers that are not cross-modal, where mixing an
        image component into a text query's target would only add noise; those run as
        exact scans.
        """
        target = _VECTOR_COLUMNS[column]
        stmt = (
            select(ProductEmbedding.product_id)
            .join(Product, Product.id == ProductEmbedding.product_id)
            .where(ProductEmbedding.provider == provider, target.is_not(None))
        )
        stmt = self._apply_filters(stmt, filters or CatalogFilters())
        stmt = stmt.order_by(target.cosine_distance(query_vector)).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def score_candidates(
        self, product_ids: list[uuid.UUID], query_vector: Vector, *, provider: str
    ) -> list[ScoredProduct]:
        """Stage 2: exact cosine similarity against each modality for a candidate set."""
        if not product_ids:
            return []

        text_sim = (1 - ProductEmbedding.text_embedding.cosine_distance(query_vector)).label(
            "text_sim"
        )
        image_sim = (1 - ProductEmbedding.image_embedding.cosine_distance(query_vector)).label(
            "image_sim"
        )

        result = await self.session.execute(
            select(Product, text_sim, image_sim, ProductEmbedding.fused_embedding)
            .join(ProductEmbedding, ProductEmbedding.product_id == Product.id)
            .where(
                ProductEmbedding.provider == provider,
                Product.id.in_(product_ids),
            )
        )
        return [
            ScoredProduct(
                product=row[0],
                text_similarity=None if row[1] is None else float(row[1]),
                image_similarity=None if row[2] is None else float(row[2]),
                fused_vector=None if row[3] is None else list(row[3]),
            )
            for row in result.all()
        ]

    async def fused_vector(self, product_id: uuid.UUID, provider: str) -> Vector | None:
        """The seed vector for an item-to-item recommendation."""
        result = await self.session.execute(
            select(ProductEmbedding.fused_embedding).where(
                ProductEmbedding.product_id == product_id,
                ProductEmbedding.provider == provider,
            )
        )
        vector = result.scalar_one_or_none()
        return None if vector is None else list(vector)

    async def most_popular(
        self, *, limit: int, filters: CatalogFilters | None = None
    ) -> list[Product]:
        """Cold-start fallback: rank by interaction volume, newest first as a tiebreak."""
        popularity = (
            select(Interaction.product_id, func.count(Interaction.id).label("hits"))
            .group_by(Interaction.product_id)
            .subquery()
        )
        stmt = (
            select(Product)
            .outerjoin(popularity, popularity.c.product_id == Product.id)
            .order_by(func.coalesce(popularity.c.hits, 0).desc(), Product.created_at.desc())
        )
        stmt = self._apply_filters(stmt, filters or CatalogFilters())
        result = await self.session.execute(stmt.limit(limit))
        return list(result.scalars().all())

    # --- helpers ------------------------------------------------------------
    @staticmethod
    def _apply_filters(stmt: Select[Any], filters: CatalogFilters) -> Select[Any]:
        if filters.category:
            stmt = stmt.where(Product.category == filters.category)
        if filters.brand:
            stmt = stmt.where(Product.brand == filters.brand)
        if filters.min_price is not None:
            stmt = stmt.where(Product.price >= filters.min_price)
        if filters.max_price is not None:
            stmt = stmt.where(Product.price <= filters.max_price)
        if filters.require_image:
            stmt = stmt.where(Product.image_path.is_not(None))
        if filters.exclude_ids:
            stmt = stmt.where(Product.id.not_in(filters.exclude_ids))
        return stmt
