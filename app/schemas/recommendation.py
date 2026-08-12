"""Recommendation response schemas."""

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.search import SearchHit

RecommendationStrategy = Literal["content_similarity", "user_profile", "popularity_fallback"]


class RecommendationResponse(BaseModel):
    strategy: RecommendationStrategy = Field(
        description=(
            "content_similarity: neighbours of a seed product. "
            "user_profile: neighbours of a vector built from the user's interactions. "
            "popularity_fallback: the user has no usable history yet (cold start)."
        )
    )
    provider: str
    signals_used: int = Field(description="Number of interactions folded into the profile vector.")
    diversity: float = Field(description="MMR lambda actually applied (0 = max diversity).")
    took_ms: float
    items: list[SearchHit]
