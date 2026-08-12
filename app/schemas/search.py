"""Search request/response schemas."""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.product import ProductRead

QueryModality = Literal["text", "image", "multimodal"]


class SearchFilters(BaseModel):
    category: str | None = None
    brand: str | None = None
    min_price: Decimal | None = Field(default=None, ge=0)
    max_price: Decimal | None = Field(default=None, ge=0)


class TextSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=100)
    provider: str | None = Field(
        default=None,
        description="Which indexed embedding provider to search. Defaults to EMBEDDING_PROVIDER.",
    )
    text_weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "How much the product's text vector counts versus its image vector when "
            "scoring. 1.0 = text only, 0.0 = image only. Defaults to DEFAULT_TEXT_WEIGHT."
        ),
    )
    min_score: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Absolute floor on the fused score. Provider-specific; usually leave unset.",
    )
    min_score_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Keep only hits scoring at least this fraction of the best hit. Scale-free, so "
            "it behaves the same across providers. 0 disables it."
        ),
    )
    filters: SearchFilters = Field(default_factory=SearchFilters)


class SearchHit(BaseModel):
    product: ProductRead
    score: float = Field(description="Fused similarity in [-1, 1]; higher is closer.")
    text_similarity: float | None
    image_similarity: float | None


class SearchResponse(BaseModel):
    query_modality: QueryModality
    provider: str
    cross_modal: bool = Field(
        description=(
            "True when the active provider embeds text and images into one shared space. "
            "When false, a text query cannot meaningfully match an image vector."
        )
    )
    text_weight: float
    min_score: float | None
    min_score_ratio: float
    candidates_considered: int
    below_threshold: int = Field(
        description="Candidates dropped for scoring too far below the best hit."
    )
    took_ms: float
    hits: list[SearchHit]
