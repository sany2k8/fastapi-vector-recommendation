"""Multimodal retrieval: turn a text query, an image, or both into ranked products."""

import time

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.errors import ValidationError
from app.core.logging import get_logger
from app.embeddings import EmbeddingProvider, Vector
from app.repositories import CatalogFilters, ProductRepository, ScoredProduct, VectorColumn
from app.schemas import ProductRead, QueryModality, SearchHit, SearchResponse
from app.services.images import validate_image
from app.services.ranking import apply_relevance_threshold, fuse_similarity

log = get_logger(__name__)


class SearchService:
    def __init__(self, session: AsyncSession, provider: EmbeddingProvider) -> None:
        self.session = session
        self.provider = provider
        self.products = ProductRepository(session)
        self.settings = get_settings()

    async def search(
        self,
        *,
        text: str | None = None,
        image: bytes | None = None,
        top_k: int = 10,
        text_weight: float | None = None,
        min_score: float | None = None,
        min_score_ratio: float | None = None,
        filters: CatalogFilters | None = None,
    ) -> SearchResponse:
        if not text and image is None:
            raise ValidationError("provide a text query, an image, or both")
        if image is not None:
            validate_image(image)

        started = time.perf_counter()
        modality = self._modality(text, image)
        weight = self._effective_weight(modality, text_weight)
        floor = self.settings.min_score_for(self.provider.name) if min_score is None else min_score
        ratio = (
            self.settings.default_min_score_ratio if min_score_ratio is None else min_score_ratio
        )

        query_vector = await self.provider.embed_query(
            text=text or None, image=image, text_weight=weight
        )
        scored = await self._retrieve(query_vector, top_k, filters, modality)

        ranked = self._rank(scored, weight)
        kept = apply_relevance_threshold(ranked, min_score=floor, min_score_ratio=ratio)
        hits = kept[:top_k]

        took_ms = (time.perf_counter() - started) * 1000
        log.info(
            "search.completed",
            modality=modality,
            provider=self.provider.name,
            candidates=len(scored),
            below_threshold=len(ranked) - len(kept),
            returned=len(hits),
            took_ms=round(took_ms, 2),
        )
        return SearchResponse(
            query_modality=modality,
            provider=self.provider.name,
            cross_modal=self.provider.supports_cross_modal,
            text_weight=weight,
            min_score=floor,
            min_score_ratio=ratio,
            candidates_considered=len(scored),
            below_threshold=len(ranked) - len(kept),
            took_ms=round(took_ms, 2),
            hits=hits,
        )

    async def _retrieve(
        self,
        query_vector: Vector,
        top_k: int,
        filters: CatalogFilters | None,
        modality: QueryModality,
    ) -> list[ScoredProduct]:
        """Stage 1 (ANN) then stage 2 (exact re-rank) over the candidate pool."""
        pool_size = min(
            top_k * self.settings.ann_candidate_multiplier, self.settings.max_top_k * 10
        )
        candidate_ids = await self.products.ann_candidates(
            query_vector,
            provider=self.provider.name,
            limit=pool_size,
            filters=filters,
            column=self._ann_column(modality),
        )
        return await self.products.score_candidates(
            candidate_ids, query_vector, provider=self.provider.name
        )

    def _ann_column(self, modality: QueryModality) -> VectorColumn:
        """Which stored vector the candidate pull should compare against.

        With a cross-modal provider everything lives in one space, so the indexed
        fused column is right for any query. Without one, a text query must be
        compared to text vectors only — the image half of the fusion is unrelated
        noise and would wreck the candidate pool.
        """
        if self.provider.supports_cross_modal:
            return "fused"
        return "image" if modality == "image" else "text"

    def _effective_weight(self, modality: QueryModality, requested: float | None) -> float:
        weight = self.settings.default_text_weight if requested is None else requested
        if self.provider.supports_cross_modal:
            return weight
        # The stub provider has no shared space: score purely within the queried modality.
        return 0.0 if modality == "image" else 1.0

    @staticmethod
    def _modality(text: str | None, image: bytes | None) -> QueryModality:
        if text and image is not None:
            return "multimodal"
        return "image" if image is not None else "text"

    @staticmethod
    def _rank(scored: list[ScoredProduct], text_weight: float) -> list[SearchHit]:
        hits = [
            SearchHit(
                product=ProductRead.from_model(item.product),
                score=fuse_similarity(item.text_similarity, item.image_similarity, text_weight),
                text_similarity=item.text_similarity,
                image_similarity=item.image_similarity,
            )
            for item in scored
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits
