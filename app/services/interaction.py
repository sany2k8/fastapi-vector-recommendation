"""Records the implicit-feedback events that personalisation feeds on."""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.models import Interaction, InteractionType
from app.repositories import InteractionRepository, ProductRepository

log = get_logger(__name__)


class InteractionService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.products = ProductRepository(session)
        self.interactions = InteractionRepository(session)

    async def record(
        self, user_id: str, product_id: uuid.UUID, event: InteractionType
    ) -> Interaction:
        if await self.products.get(product_id) is None:
            raise NotFoundError(f"product {product_id} not found")

        interaction = Interaction(
            id=uuid.uuid4(),
            user_id=user_id,
            product_id=product_id,
            event=event.value,
        )
        await self.interactions.add(interaction)
        # NB: structlog reserves `event` for the message itself — hence `event_type`.
        log.info("interaction.recorded", user_id=user_id, event_type=event.value)
        return interaction
