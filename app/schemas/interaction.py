"""Interaction event schemas."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models import InteractionType


class InteractionCreate(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    product_id: uuid.UUID
    event: InteractionType = InteractionType.VIEW


class InteractionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: str
    product_id: uuid.UUID
    event: str
    created_at: datetime
