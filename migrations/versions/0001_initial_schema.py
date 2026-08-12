"""Initial schema: products with multimodal vectors, interactions, HNSW index.

The vector width comes from EMBEDDING_DIM at migration time. Changing that setting
means downgrading and re-running this migration, then re-embedding the catalogue —
pgvector columns are fixed width.

Revision ID: 0001
Revises:
Create Date: 2026-08-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from app.core.config import get_settings

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DIM = get_settings().embedding_dim


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "products",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("sku", sa.String(64), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("category", sa.String(120), nullable=False),
        sa.Column("brand", sa.String(120), nullable=True),
        sa.Column("price", sa.Numeric(12, 2), nullable=False, server_default="0.00"),
        sa.Column("currency", sa.String(3), nullable=False, server_default="USD"),
        sa.Column("image_path", sa.String(512), nullable=True),
        sa.Column(
            "attributes",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("text_embedding", Vector(DIM), nullable=True),
        sa.Column("image_embedding", Vector(DIM), nullable=True),
        sa.Column("fused_embedding", Vector(DIM), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_products_sku", "products", ["sku"], unique=True)
    op.create_index("ix_products_category", "products", ["category"])

    # Cosine HNSW over the fused vector — this is the ANN index every search rides on.
    op.create_index(
        "ix_products_fused_hnsw",
        "products",
        ["fused_embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"fused_embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "interactions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(128), nullable=False),
        sa.Column(
            "product_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("products.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_interactions_user_id", "interactions", ["user_id"])
    op.create_index("ix_interactions_product_id", "interactions", ["product_id"])
    op.create_index("ix_interactions_created_at", "interactions", ["created_at"])
    op.create_index("ix_interactions_user_created", "interactions", ["user_id", "created_at"])


def downgrade() -> None:
    op.drop_table("interactions")
    op.drop_index("ix_products_fused_hnsw", table_name="products")
    op.drop_table("products")
