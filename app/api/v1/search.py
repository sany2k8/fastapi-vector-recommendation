"""Search endpoints: text, image, and the two combined.

Every endpoint takes an optional `provider`, so the same query can be run against each
indexed embedding backend and the results compared side by side.
"""

from typing import Annotated

from fastapi import APIRouter, File, Form, Query, UploadFile

from app.api.deps import FiltersDep, SearchDep, SessionDep, read_upload
from app.core.errors import ValidationError
from app.embeddings import get_provider
from app.repositories import CatalogFilters
from app.schemas import SearchResponse, TextSearchRequest
from app.services.search import SearchService

router = APIRouter(prefix="/search", tags=["search"])

ScoreQuery = Annotated[float | None, Query(ge=-1.0, le=1.0)]
RatioQuery = Annotated[float | None, Query(ge=0.0, le=1.0)]


@router.post("/text", response_model=SearchResponse)
async def search_by_text(
    request: TextSearchRequest, service: SearchDep, session: SessionDep
) -> SearchResponse:
    """Semantic search over the catalogue.

    With a cross-modal provider this also matches product *photos*, because the query
    sentence and the images live in the same vector space.
    """
    # The provider may arrive in the JSON body here rather than the query string.
    active = (
        service
        if request.provider is None
        else SearchService(session, get_provider(request.provider))
    )
    filters = CatalogFilters(
        category=request.filters.category,
        brand=request.filters.brand,
        min_price=request.filters.min_price,
        max_price=request.filters.max_price,
    )
    return await active.search(
        text=request.query,
        top_k=request.top_k,
        text_weight=request.text_weight,
        min_score=request.min_score,
        min_score_ratio=request.min_score_ratio,
        filters=filters,
    )


@router.post("/image", response_model=SearchResponse)
async def search_by_image(
    service: SearchDep,
    filters: FiltersDep,
    image: Annotated[UploadFile, File()],
    top_k: Annotated[int, Query(ge=1, le=100)] = 10,
    min_score: ScoreQuery = None,
    min_score_ratio: RatioQuery = None,
) -> SearchResponse:
    """Reverse image search — "find me products that look like this photo"."""
    payload = await read_upload(image)
    if payload is None:
        raise ValidationError("an image file is required")
    return await service.search(
        image=payload,
        top_k=top_k,
        min_score=min_score,
        min_score_ratio=min_score_ratio,
        filters=filters,
    )


@router.post("/multimodal", response_model=SearchResponse)
async def search_multimodal(
    service: SearchDep,
    filters: FiltersDep,
    query: Annotated[str | None, Form()] = None,
    image: Annotated[UploadFile | None, File()] = None,
    top_k: Annotated[int, Query(ge=1, le=100)] = 10,
    text_weight: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    min_score: ScoreQuery = None,
    min_score_ratio: RatioQuery = None,
) -> SearchResponse:
    """Search with a sentence, a photo, or both blended together.

    Sending both is the interesting case: "this shoe, but in red" is expressed as the
    photo's vector nudged toward the text's vector, controlled by `text_weight`.
    """
    payload = await read_upload(image)
    if not query and payload is None:
        raise ValidationError("provide `query`, `image`, or both")
    return await service.search(
        text=query,
        image=payload,
        top_k=top_k,
        text_weight=text_weight,
        min_score=min_score,
        min_score_ratio=min_score_ratio,
        filters=filters,
    )
