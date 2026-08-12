"""Add partial HNSW indexes for the gemini and voyage providers.

Every provider needs its own partial index: one shared index over `fused_embedding`
would merge unrelated vector spaces into a single HNSW graph and wreck recall for all
of them. Adding a provider to `ALL_PROVIDERS` therefore always needs a migration too —
without it queries still work, just via a sequential scan.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

NEW_PROVIDERS = ("gemini", "voyage")


def upgrade() -> None:
    for name in NEW_PROVIDERS:
        op.create_index(
            f"ix_product_embeddings_fused_{name}",
            "product_embeddings",
            ["fused_embedding"],
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"fused_embedding": "vector_cosine_ops"},
            postgresql_where=sa.text(f"provider = '{name}'"),
        )


def downgrade() -> None:
    for name in NEW_PROVIDERS:
        op.drop_index(f"ix_product_embeddings_fused_{name}", table_name="product_embeddings")
