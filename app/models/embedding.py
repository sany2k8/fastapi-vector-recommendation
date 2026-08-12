"""Per-provider vectors for a product.

One row per (product, provider), so several embedding backends can be indexed at the
same time and a query can pick which one to search — that is what makes an A/B of
"local_hash vs Jina vs Cohere on the same query" possible without re-indexing.

Every provider is pinned to the same `EMBEDDING_DIM`, which is what lets one column
type and one index definition serve all of them. Vectors from different providers are
never compared with each other; the provider is always part of the query predicate.
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, column, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.config import ALL_PROVIDERS, get_settings
from app.core.db import Base

EMBEDDING_DIM = get_settings().embedding_dim


class ProductEmbedding(Base):
    __tablename__ = "product_embeddings"

    product_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("products.id", ondelete="CASCADE"), primary_key=True
    )
    provider: Mapped[str] = mapped_column(String(32), primary_key=True)
    model: Mapped[str] = mapped_column(String(128), default="")
    dim: Mapped[int] = mapped_column(Integer, default=EMBEDDING_DIM)

    text_embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), default=None)
    image_embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), default=None)
    fused_embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # A partial HNSW index per provider. A single index over the whole table would mix
    # unrelated vector spaces into one graph, so every provider gets its own.
    __table_args__ = tuple(
        Index(
            f"ix_product_embeddings_fused_{name}",
            "fused_embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"fused_embedding": "vector_cosine_ops"},
            postgresql_where=column("provider") == name,
        )
        for name in ALL_PROVIDERS
    )

    def __repr__(self) -> str:
        return f"<ProductEmbedding {self.product_id} {self.provider}>"
