"""Recommendation endpoints: item-to-item and personalised."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import FiltersDep, RecommendationDep
from app.schemas import RecommendationResponse

router = APIRouter(tags=["recommendations"])

DiversityQuery = Annotated[
    float,
    Query(
        ge=0.0,
        le=1.0,
        description=(
            "MMR lambda. 1.0 ranks purely by relevance; lower values trade relevance "
            "for variety so the results are not five versions of the same product."
        ),
    ),
]
ScoreQuery = Annotated[float | None, Query(ge=-1.0, le=1.0)]
RatioQuery = Annotated[float | None, Query(ge=0.0, le=1.0)]


@router.get("/products/{product_id}/similar", response_model=RecommendationResponse)
async def similar_products(
    product_id: uuid.UUID,
    service: RecommendationDep,
    filters: FiltersDep,
    top_k: Annotated[int, Query(ge=1, le=100)] = 10,
    text_weight: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    diversity: DiversityQuery = 1.0,
    min_score: ScoreQuery = None,
    min_score_ratio: RatioQuery = None,
) -> RecommendationResponse:
    """ "More like this" — nearest neighbours of the seed product's fused vector."""
    return await service.similar_to_product(
        product_id,
        top_k=top_k,
        text_weight=text_weight,
        diversity=diversity,
        min_score=min_score,
        min_score_ratio=min_score_ratio,
        filters=filters,
    )


@router.get("/users/{user_id}/recommendations", response_model=RecommendationResponse)
async def recommendations_for_user(
    user_id: str,
    service: RecommendationDep,
    filters: FiltersDep,
    top_k: Annotated[int, Query(ge=1, le=100)] = 10,
    text_weight: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    diversity: DiversityQuery = 0.7,
    min_score: ScoreQuery = None,
    min_score_ratio: RatioQuery = None,
    history_limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> RecommendationResponse:
    """Personalised feed built from the user's interaction history.

    Falls back to a popularity ranking when the user has no usable history yet;
    the response's `strategy` field says which path was taken.
    """
    return await service.for_user(
        user_id,
        top_k=top_k,
        text_weight=text_weight,
        diversity=diversity,
        min_score=min_score,
        min_score_ratio=min_score_ratio,
        history_limit=history_limit,
        filters=filters,
    )
