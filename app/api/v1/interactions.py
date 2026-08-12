"""Interaction ingestion — the feedback loop that personalisation depends on."""

from fastapi import APIRouter, status

from app.api.deps import InteractionDep
from app.schemas import InteractionCreate, InteractionRead

router = APIRouter(prefix="/interactions", tags=["interactions"])


@router.post("", response_model=InteractionRead, status_code=status.HTTP_201_CREATED)
async def record_interaction(
    payload: InteractionCreate, service: InteractionDep
) -> InteractionRead:
    """Record a view/click/like/cart/purchase event for a user and product."""
    interaction = await service.record(payload.user_id, payload.product_id, payload.event)
    return InteractionRead.model_validate(interaction)
