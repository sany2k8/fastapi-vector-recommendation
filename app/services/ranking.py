"""Pure ranking maths. No I/O here, so every rule below is directly unit-testable."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import numpy as np

from app.embeddings import Vector
from app.models import INTERACTION_WEIGHTS, InteractionType

#: An interaction this many days old counts half as much as a fresh one.
RECENCY_HALF_LIFE_DAYS = 14.0


class HasScore(Protocol):
    """Anything carrying a fused relevance score."""

    @property
    def score(self) -> float: ...


def fuse_similarity(
    text_similarity: float | None,
    image_similarity: float | None,
    text_weight: float,
) -> float:
    """Blend per-modality similarities into one score.

    A product with no image is scored on its text alone rather than being penalised
    for the missing modality — otherwise image-less items could never rank, which is
    the wrong behaviour for a real catalogue where photos arrive late.
    """
    if text_similarity is None and image_similarity is None:
        return 0.0
    if image_similarity is None:
        return float(text_similarity or 0.0)
    if text_similarity is None:
        return float(image_similarity)
    return float(text_weight * text_similarity + (1.0 - text_weight) * image_similarity)


def apply_relevance_threshold[ScoreT: HasScore](
    hits: Sequence[ScoreT],
    *,
    min_score: float | None,
    min_score_ratio: float,
) -> list[ScoreT]:
    """Drop weak hits. Expects `hits` already sorted best-first.

    Two independent cutoffs, because neither alone is enough:

    * `min_score` is an absolute floor. Precise, but the "right" number differs per
      provider and per modality — a strong `local_hash` text match sits near 0.6 while
      a strong Cohere match sits near 0.9, so a hard-coded floor is wrong somewhere.
    * `min_score_ratio` keeps only hits within a fraction of the *best* hit. Scale-free,
      so it transfers across providers, and it is what stops a query with three good
      answers from padding the page out to twenty bad ones.

    The ratio is skipped when the best score is not positive: with nothing genuinely
    similar in the catalogue, scaling a negative or zero top score is meaningless.
    """
    filtered = list(hits)
    if min_score is not None:
        filtered = [hit for hit in filtered if hit.score >= min_score]

    if filtered and min_score_ratio > 0.0:
        best = filtered[0].score
        if best > 0.0:
            floor = best * min_score_ratio
            filtered = [hit for hit in filtered if hit.score >= floor]
    return filtered


def recency_decay(created_at: datetime, *, now: datetime | None = None) -> float:
    """Exponential half-life decay in [0, 1]."""
    reference = now or datetime.now(UTC)
    stamp = created_at if created_at.tzinfo else created_at.replace(tzinfo=UTC)
    age_days = max((reference - stamp).total_seconds() / 86_400.0, 0.0)
    return float(0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS))


def interaction_weight(event: str, created_at: datetime, *, now: datetime | None = None) -> float:
    """Intent strength of an event, discounted by how long ago it happened."""
    try:
        base = INTERACTION_WEIGHTS[InteractionType(event)]
    except ValueError:
        base = 1.0
    return base * recency_decay(created_at, now=now)


@dataclass(slots=True)
class DiversityCandidate[T]:
    """Anything that can be MMR-reranked: a relevance score plus its own vector."""

    key: T
    relevance: float
    vector: Vector


def mmr_rerank[T](
    candidates: Sequence[DiversityCandidate[T]],
    *,
    top_k: int,
    diversity_lambda: float,
) -> list[DiversityCandidate[T]]:
    """Maximal Marginal Relevance.

    Greedily picks the item maximising

        lambda * relevance - (1 - lambda) * max_similarity_to_already_picked

    so a recommendation set does not collapse into five near-identical products.
    `diversity_lambda == 1.0` is pure relevance and short-circuits.
    """
    if not candidates:
        return []
    if diversity_lambda >= 1.0:
        ranked = sorted(candidates, key=lambda c: c.relevance, reverse=True)
        return list(ranked[:top_k])

    matrix = np.asarray([c.vector for c in candidates], dtype=np.float32)
    # Vectors are stored L2-normalised, so a dot product is the cosine similarity.
    pairwise = matrix @ matrix.T

    remaining = list(range(len(candidates)))
    selected: list[int] = []

    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda i: candidates[i].relevance)
        else:
            best = max(
                remaining,
                key=lambda i: (
                    diversity_lambda * candidates[i].relevance
                    - (1.0 - diversity_lambda) * float(pairwise[i, selected].max())
                ),
            )
        selected.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected]
