"""User interaction events — the raw signal behind personalised recommendations."""

import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, ForeignKey, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class InteractionType(StrEnum):
    """Implicit-feedback event types, ordered by how much intent they signal."""

    VIEW = "view"
    CLICK = "click"
    LIKE = "like"
    CART = "cart"
    PURCHASE = "purchase"


#: How strongly each event pulls the user profile vector toward a product.
INTERACTION_WEIGHTS: dict[InteractionType, float] = {
    InteractionType.VIEW: 0.5,
    InteractionType.CLICK: 1.0,
    InteractionType.LIKE: 2.0,
    InteractionType.CART: 3.0,
    InteractionType.PURCHASE: 4.0,
}


class Interaction(Base):
    __tablename__ = "interactions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    product_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("products.id", ondelete="CASCADE"), index=True
    )
    event: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    __table_args__ = (Index("ix_interactions_user_created", "user_id", "created_at"),)

    def __repr__(self) -> str:
        return f"<Interaction {self.user_id} {self.event} {self.product_id}>"
