"""Move vectors off `products` into a per-provider `product_embeddings` table.

Lets several embedding backends be indexed at once so the same query can be compared
across them. Existing vectors are carried over and attributed to the provider named by
EMBEDDING_PROVIDER at the time this migration runs.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from app.core.config import get_settings

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

settings = get_settings()
DIM = settings.embedding_dim
CURRENT_PROVIDER = settings.embedding_provider

#: Frozen on purpose — the providers that existed when this migration was written.
#:
#: Iterating the live `ALL_PROVIDERS` here was a bug: adding `gemini` and `voyage` to
#: that Literal retroactively changed what this migration creates, so a fresh database
#: built five indexes and 0003 then failed with "relation already exists". A migration
#: is a snapshot of history and must never read config that can change under it.
#: New providers get their index in a *new* migration.
PROVIDERS_AT_0002 = ("local_hash", "jina", "cohere")


def upgrade() -> None:
    op.create_table(
        "product_embeddings",
        sa.Column(
            "product_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("products.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("provider", sa.String(32), primary_key=True),
        sa.Column("model", sa.String(128), nullable=False, server_default=""),
        sa.Column("dim", sa.Integer(), nullable=False, server_default=str(DIM)),
        sa.Column("text_embedding", Vector(DIM), nullable=True),
        sa.Column("image_embedding", Vector(DIM), nullable=True),
        sa.Column("fused_embedding", Vector(DIM), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Carry the existing single-provider vectors across rather than forcing a re-seed.
    op.execute(
        sa.text(
            """
            INSERT INTO product_embeddings
                (product_id, provider, model, dim,
                 text_embedding, image_embedding, fused_embedding)
            SELECT id, :provider, '', :dim, text_embedding, image_embedding, fused_embedding
            FROM products
            WHERE text_embedding IS NOT NULL OR image_embedding IS NOT NULL
            """
        ).bindparams(provider=CURRENT_PROVIDER, dim=DIM)
    )

    # One partial HNSW index per provider: a single shared index would merge unrelated
    # vector spaces into one graph and wreck recall for every provider in it.
    for name in PROVIDERS_AT_0002:
        op.create_index(
            f"ix_product_embeddings_fused_{name}",
            "product_embeddings",
            ["fused_embedding"],
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"fused_embedding": "vector_cosine_ops"},
            postgresql_where=sa.text(f"provider = '{name}'"),
        )

    op.drop_index("ix_products_fused_hnsw", table_name="products")
    op.drop_column("products", "fused_embedding")
    op.drop_column("products", "image_embedding")
    op.drop_column("products", "text_embedding")


def downgrade() -> None:
    op.add_column("products", sa.Column("text_embedding", Vector(DIM), nullable=True))
    op.add_column("products", sa.Column("image_embedding", Vector(DIM), nullable=True))
    op.add_column("products", sa.Column("fused_embedding", Vector(DIM), nullable=True))

    op.execute(
        sa.text(
            """
            UPDATE products p
            SET text_embedding = e.text_embedding,
                image_embedding = e.image_embedding,
                fused_embedding = e.fused_embedding
            FROM product_embeddings e
            WHERE e.product_id = p.id AND e.provider = :provider
            """
        ).bindparams(provider=CURRENT_PROVIDER)
    )

    op.create_index(
        "ix_products_fused_hnsw",
        "products",
        ["fused_embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"fused_embedding": "vector_cosine_ops"},
    )
    op.drop_table("product_embeddings")
