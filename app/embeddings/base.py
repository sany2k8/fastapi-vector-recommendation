"""Embedding provider contract.

Every provider maps content into a single fixed-width vector space of `dim` floats,
L2-normalised, so cosine similarity reduces to a dot product.

`supports_cross_modal` is the important flag: real multimodal models (Jina v4,
Cohere embed-v4) place text and images in the *same* space, which is what makes
"find products whose photo matches this sentence" work. The offline stand-in does
not, and callers must degrade gracefully rather than return nonsense.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from app.core.errors import UnsupportedOperationError

Vector = list[float]
InputKind = Literal["query", "document"]

BATCH_SIZE = 32


def l2_normalize(vec: npt.ArrayLike) -> Vector:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return [0.0] * len(arr)
    return (arr / norm).astype(np.float32).tolist()  # type: ignore[no-any-return]


def weighted_merge(vectors: Sequence[Vector], weights: Sequence[float]) -> Vector:
    """Weighted sum of vectors, re-normalised back onto the unit sphere.

    Used both to fuse a product's text and image vectors and to build a user
    profile vector out of the items they interacted with.
    """
    if not vectors:
        raise ValueError("weighted_merge requires at least one vector")
    stacked = np.asarray(vectors, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
    merged = (stacked * w).sum(axis=0)
    return l2_normalize(merged)


class EmbeddingProvider(ABC):
    """Base class for all embedding backends."""

    name: str

    #: Whether this backend can embed images at all. Text-only models (Gemini
    #: embedding-001, most open text encoders) set this False; callers must skip the
    #: image half rather than making the provider invent a vector for it.
    supports_images: bool = True

    #: Whether text and images land in the *same* space. Always False when
    #: `supports_images` is False — there is no second modality to share with.
    supports_cross_modal: bool

    #: Images are far heavier per request than text, so they are chunked smaller to
    #: keep a single payload from ballooning past provider size limits.
    image_batch_size: int = 8

    def __init__(self, dim: int, image_batch_size: int | None = None) -> None:
        self.dim = dim
        if image_batch_size is not None:
            self.image_batch_size = image_batch_size

    @abstractmethod
    async def embed_texts(self, texts: Sequence[str], kind: InputKind = "document") -> list[Vector]:
        """Embed a batch of strings."""

    async def embed_images(
        self, images: Sequence[bytes], kind: InputKind = "document"
    ) -> list[Vector]:
        """Embed a batch of raw image bytes.

        Text-only providers inherit this refusal instead of returning something
        meaningless; check `supports_images` before calling.
        """
        raise UnsupportedOperationError(
            f"embedding provider {self.name!r} cannot embed images. "
            f"Index the catalogue with a multimodal provider to search by picture."
        )

    async def embed_text(self, text: str, kind: InputKind = "document") -> Vector:
        return (await self.embed_texts([text], kind))[0]

    async def embed_image(self, image: bytes, kind: InputKind = "document") -> Vector:
        return (await self.embed_images([image], kind))[0]

    async def embed_query(
        self,
        text: str | None = None,
        image: bytes | None = None,
        text_weight: float = 0.5,
    ) -> Vector:
        """Build a single query vector from text, an image, or both.

        With both, the two vectors are blended by `text_weight` so a caller can say
        "mostly this sentence, a little of that photo".
        """
        if text is None and image is None:
            raise ValueError("embed_query requires text, image, or both")
        if image is not None and not self.supports_images:
            raise UnsupportedOperationError(
                f"embedding provider {self.name!r} is text-only; it cannot search by image"
            )
        if image is None:
            assert text is not None
            return await self.embed_text(text, "query")
        if text is None:
            return await self.embed_image(image, "query")
        text_vec = await self.embed_text(text, "query")
        image_vec = await self.embed_image(image, "query")
        return weighted_merge([text_vec, image_vec], [text_weight, 1.0 - text_weight])

    async def aclose(self) -> None:  # noqa: B027 - optional hook; local providers hold nothing
        """Release any held resources (HTTP clients)."""

    def _validate_dim(self, vectors: list[Vector]) -> list[Vector]:
        for vec in vectors:
            if len(vec) != self.dim:
                raise ValueError(
                    f"{self.name} returned a {len(vec)}-d vector but EMBEDDING_DIM is {self.dim}"
                )
        return vectors


def chunked[ItemT](items: Sequence[ItemT], size: int = BATCH_SIZE) -> list[Sequence[ItemT]]:
    return [items[i : i + size] for i in range(0, len(items), size)]
