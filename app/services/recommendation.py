"""Recommendations: item-to-item similarity and personalised user-profile retrieval.

Both paths end in the same place — a query vector, an ANN pull, an exact re-rank, a
relevance threshold, then MMR for diversity. What differs is where the query vector
comes from:

  * "more like this"  -> the seed product's own fused vector
  * "for you"         -> a weighted average of the vectors of everything the user
                         touched, with stronger events and recent events counting more

Everything is scoped to one provider, since vectors from different providers are not
comparable.
"""

import time
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.errors import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.embeddings import EmbeddingProvider, Vector, weighted_merge
from app.models import Interaction
from app.repositories import (
    CatalogFilters,
    InteractionRepository,
    ProductRepository,
    ScoredProduct,
)
from app.schemas import ProductRead, RecommendationResponse, SearchHit
from app.services.ranking import (
    DiversityCandidate,
    apply_relevance_threshold,
    fuse_similarity,
    interaction_weight,
    mmr_rerank,
)

log = get_logger(__name__)


class RecommendationService:
    def __init__(self, session: AsyncSession, provider: EmbeddingProvider) -> None:
        self.session = session
        self.provider = provider
        self.products = ProductRepository(session)
        self.interactions = InteractionRepository(session)
        self.settings = get_settings()

    async def similar_to_product(
        self,
        product_id: uuid.UUID,
        *,
        top_k: int = 10,
        text_weight: float | None = None,
        diversity: float = 1.0,
        min_score: float | None = None,
        min_score_ratio: float | None = None,
        filters: CatalogFilters | None = None,
    ) -> RecommendationResponse:
        started = time.perf_counter()
        if await self.products.get(product_id) is None:
            raise NotFoundError(f"product {product_id} not found")

        seed_vector = await self.products.fused_vector(product_id, self.provider.name)
        if seed_vector is None:
            raise ValidationError(
                f"product {product_id} has no {self.provider.name} embedding yet — "
                f"re-index with this provider first"
            )

        active_filters = filters or CatalogFilters()
        active_filters.exclude_ids = [*active_filters.exclude_ids, product_id]

        items = await self._retrieve_and_rank(
            query_vector=seed_vector,
            top_k=top_k,
            text_weight=text_weight,
            diversity=diversity,
            min_score=min_score,
            min_score_ratio=min_score_ratio,
            filters=active_filters,
        )

        took_ms = (time.perf_counter() - started) * 1000
        log.info("recs.similar", product_id=str(product_id), returned=len(items))
        return RecommendationResponse(
            strategy="content_similarity",
            provider=self.provider.name,
            signals_used=1,
            diversity=diversity,
            took_ms=round(took_ms, 2),
            items=items,
        )

    async def for_user(
        self,
        user_id: str,
        *,
        top_k: int = 10,
        text_weight: float | None = None,
        diversity: float = 0.7,
        min_score: float | None = None,
        min_score_ratio: float | None = None,
        history_limit: int = 100,
        filters: CatalogFilters | None = None,
    ) -> RecommendationResponse:
        started = time.perf_counter()
        history = await self.interactions.recent_with_vectors(
            user_id, provider=self.provider.name, limit=history_limit
        )

        active_filters = filters or CatalogFilters()
        if history:
            seen = [interaction.product_id for interaction, _ in history]
            active_filters.exclude_ids = [*active_filters.exclude_ids, *seen]

        profile = self._build_profile_vector(history)
        if profile is None:
            items = await self._popular(top_k, active_filters)
            took_ms = (time.perf_counter() - started) * 1000
            log.info("recs.cold_start", user_id=user_id, returned=len(items))
            return RecommendationResponse(
                strategy="popularity_fallback",
                provider=self.provider.name,
                signals_used=0,
                diversity=diversity,
                took_ms=round(took_ms, 2),
                items=items,
            )

        items = await self._retrieve_and_rank(
            query_vector=profile,
            top_k=top_k,
            text_weight=text_weight,
            diversity=diversity,
            min_score=min_score,
            min_score_ratio=min_score_ratio,
            filters=active_filters,
        )

        took_ms = (time.perf_counter() - started) * 1000
        log.info("recs.user_profile", user_id=user_id, signals=len(history), returned=len(items))
        return RecommendationResponse(
            strategy="user_profile",
            provider=self.provider.name,
            signals_used=len(history),
            diversity=diversity,
            took_ms=round(took_ms, 2),
            items=items,
        )

    # --- internals ----------------------------------------------------------
    @staticmethod
    def _build_profile_vector(history: list[tuple[Interaction, Vector]]) -> Vector | None:
        """Collapse a user's history into a single taste vector."""
        vectors: list[Vector] = []
        weights: list[float] = []
        for interaction, vector in history:
            vectors.append(vector)
            weights.append(interaction_weight(interaction.event, interaction.created_at))

        if not vectors or sum(weights) <= 0.0:
            return None
        return weighted_merge(vectors, weights)

    async def _retrieve_and_rank(
        self,
        *,
        query_vector: Vector,
        top_k: int,
        text_weight: float | None,
        diversity: float,
        min_score: float | None,
        min_score_ratio: float | None,
        filters: CatalogFilters,
    ) -> list[SearchHit]:
        pool_size = min(
            top_k * self.settings.ann_candidate_multiplier, self.settings.max_top_k * 10
        )
        candidate_ids = await self.products.ann_candidates(
            query_vector,
            provider=self.provider.name,
            limit=pool_size,
            filters=filters,
            column="fused",
        )
        scored = await self.products.score_candidates(
            candidate_ids, query_vector, provider=self.provider.name
        )
        weight = self.settings.default_text_weight if text_weight is None else text_weight
        floor = self.settings.min_score_for(self.provider.name) if min_score is None else min_score
        ratio = (
            self.settings.default_min_score_ratio if min_score_ratio is None else min_score_ratio
        )

        candidates = [
            DiversityCandidate(
                key=item,
                relevance=fuse_similarity(item.text_similarity, item.image_similarity, weight),
                vector=item.fused_vector,
            )
            for item in scored
            if item.fused_vector is not None
        ]
        # Threshold before diversifying: MMR should choose between plausible items, not
        # be handed junk it then dutifully spreads across the results page.
        candidates.sort(key=lambda c: c.relevance, reverse=True)
        kept = apply_relevance_threshold(
            [_Scored(c) for c in candidates], min_score=floor, min_score_ratio=ratio
        )
        selected = mmr_rerank(
            [item.candidate for item in kept], top_k=top_k, diversity_lambda=diversity
        )
        return [self._to_hit(c.key, weight) for c in selected]

    async def _popular(self, top_k: int, filters: CatalogFilters) -> list[SearchHit]:
        products = await self.products.most_popular(limit=top_k, filters=filters)
        return [
            SearchHit(
                product=ProductRead.from_model(product),
                score=0.0,
                text_similarity=None,
                image_similarity=None,
            )
            for product in products
        ]

    @staticmethod
    def _to_hit(item: ScoredProduct, text_weight: float) -> SearchHit:
        return SearchHit(
            product=ProductRead.from_model(item.product),
            score=fuse_similarity(item.text_similarity, item.image_similarity, text_weight),
            text_similarity=item.text_similarity,
            image_similarity=item.image_similarity,
        )


class _Scored:
    """Adapts a DiversityCandidate to the `.score` shape the threshold helper expects."""

    __slots__ = ("candidate",)

    def __init__(self, candidate: DiversityCandidate[ScoredProduct]) -> None:
        self.candidate = candidate

    @property
    def score(self) -> float:
        return self.candidate.relevance
