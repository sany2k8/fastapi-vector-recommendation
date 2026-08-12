"""Ranking maths: fusion, recency decay, and MMR diversity."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from app.models import InteractionType
from app.services.ranking import (
    RECENCY_HALF_LIFE_DAYS,
    DiversityCandidate,
    apply_relevance_threshold,
    fuse_similarity,
    interaction_weight,
    mmr_rerank,
    recency_decay,
)


class TestFuseSimilarity:
    def test_blends_both_modalities_by_weight(self) -> None:
        assert fuse_similarity(1.0, 0.0, 0.75) == pytest.approx(0.75)
        assert fuse_similarity(0.0, 1.0, 0.75) == pytest.approx(0.25)

    def test_weight_extremes_select_a_single_modality(self) -> None:
        assert fuse_similarity(0.2, 0.9, 1.0) == pytest.approx(0.2)
        assert fuse_similarity(0.2, 0.9, 0.0) == pytest.approx(0.9)

    def test_product_without_image_is_scored_on_text_alone(self) -> None:
        # Not penalised down toward zero — an image-less item must stay rankable.
        assert fuse_similarity(0.8, None, 0.5) == pytest.approx(0.8)

    def test_image_only_product_is_scored_on_its_image(self) -> None:
        assert fuse_similarity(None, 0.6, 0.9) == pytest.approx(0.6)

    def test_no_vectors_at_all_scores_zero(self) -> None:
        assert fuse_similarity(None, None, 0.5) == 0.0


@dataclass
class Hit:
    name: str
    score: float


class TestRelevanceThreshold:
    @staticmethod
    def _hits(*scores: float) -> list[Hit]:
        return [Hit(name=f"h{i}", score=s) for i, s in enumerate(scores)]

    def test_ratio_keeps_only_hits_near_the_best(self) -> None:
        kept = apply_relevance_threshold(
            self._hits(0.90, 0.62, 0.44, 0.11, 0.02), min_score=None, min_score_ratio=0.5
        )
        assert [h.score for h in kept] == [0.90, 0.62]

    def test_ratio_of_zero_disables_the_cutoff(self) -> None:
        hits = self._hits(0.9, 0.1, 0.01)
        assert apply_relevance_threshold(hits, min_score=None, min_score_ratio=0.0) == hits

    def test_absolute_floor_applies_independently(self) -> None:
        kept = apply_relevance_threshold(
            self._hits(0.9, 0.5, 0.2), min_score=0.45, min_score_ratio=0.0
        )
        assert [h.score for h in kept] == [0.9, 0.5]

    def test_both_cutoffs_compose(self) -> None:
        kept = apply_relevance_threshold(
            self._hits(0.9, 0.5, 0.2), min_score=0.3, min_score_ratio=0.9
        )
        assert [h.score for h in kept] == [0.9]

    def test_a_uniformly_weak_result_set_still_returns_its_best(self) -> None:
        """A weak-but-plausible best answer should survive; only the tail is trimmed."""
        kept = apply_relevance_threshold(
            self._hits(0.20, 0.18, 0.03), min_score=None, min_score_ratio=0.5
        )
        assert [h.score for h in kept] == [0.20, 0.18]

    def test_non_positive_best_score_skips_the_ratio(self) -> None:
        """Scaling a negative top score is meaningless, so the ratio must not apply."""
        hits = self._hits(-0.1, -0.5)
        assert apply_relevance_threshold(hits, min_score=None, min_score_ratio=0.5) == hits

    def test_everything_can_be_filtered_out(self) -> None:
        kept = apply_relevance_threshold(self._hits(0.2, 0.1), min_score=0.9, min_score_ratio=0.5)
        assert kept == []

    def test_empty_input_is_safe(self) -> None:
        assert apply_relevance_threshold([], min_score=0.5, min_score_ratio=0.5) == []


class TestRecency:
    def test_fresh_interaction_keeps_full_weight(self) -> None:
        now = datetime.now(UTC)
        assert recency_decay(now, now=now) == pytest.approx(1.0)

    def test_one_half_life_halves_the_weight(self) -> None:
        now = datetime.now(UTC)
        old = now - timedelta(days=RECENCY_HALF_LIFE_DAYS)
        assert recency_decay(old, now=now) == pytest.approx(0.5, rel=1e-6)

    def test_naive_datetimes_are_treated_as_utc(self) -> None:
        now = datetime.now(UTC)
        naive = now.replace(tzinfo=None)
        assert recency_decay(naive, now=now) == pytest.approx(1.0, abs=1e-6)

    def test_purchase_outweighs_view_at_equal_age(self) -> None:
        now = datetime.now(UTC)
        purchase = interaction_weight(InteractionType.PURCHASE.value, now, now=now)
        view = interaction_weight(InteractionType.VIEW.value, now, now=now)
        assert purchase > view

    def test_unknown_event_falls_back_to_neutral_weight(self) -> None:
        now = datetime.now(UTC)
        assert interaction_weight("teleported", now, now=now) == pytest.approx(1.0)

    def test_recent_view_can_outweigh_an_ancient_purchase(self) -> None:
        now = datetime.now(UTC)
        fresh_view = interaction_weight("view", now, now=now)
        old_purchase = interaction_weight("purchase", now - timedelta(days=120), now=now)
        assert fresh_view > old_purchase


class TestMMR:
    @staticmethod
    def _candidates() -> list[DiversityCandidate[str]]:
        # a and b are near-identical; c is orthogonal to both.
        return [
            DiversityCandidate(key="a", relevance=0.90, vector=[1.0, 0.0, 0.0]),
            DiversityCandidate(key="b", relevance=0.89, vector=[0.999, 0.045, 0.0]),
            DiversityCandidate(key="c", relevance=0.60, vector=[0.0, 1.0, 0.0]),
        ]

    def test_lambda_one_is_pure_relevance(self) -> None:
        picked = mmr_rerank(self._candidates(), top_k=2, diversity_lambda=1.0)
        assert [c.key for c in picked] == ["a", "b"]

    def test_diversity_displaces_the_near_duplicate(self) -> None:
        picked = mmr_rerank(self._candidates(), top_k=2, diversity_lambda=0.5)
        assert [c.key for c in picked] == ["a", "c"]

    def test_top_k_is_respected_and_never_repeats(self) -> None:
        picked = mmr_rerank(self._candidates(), top_k=2, diversity_lambda=0.3)
        assert len(picked) == 2
        assert len({c.key for c in picked}) == 2

    def test_empty_input_is_safe(self) -> None:
        assert mmr_rerank([], top_k=5, diversity_lambda=0.5) == []

    def test_top_k_larger_than_pool_returns_everything(self) -> None:
        picked = mmr_rerank(self._candidates(), top_k=99, diversity_lambda=0.5)
        assert len(picked) == 3
