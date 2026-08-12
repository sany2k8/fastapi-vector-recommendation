"""Google Gemini `gemini-embedding-001` — text only, but the most generous free tier.

Docs: https://ai.google.dev/gemini-api/docs/embeddings
POST https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents

Two details that bite:

* **embedding-001 is text-only.** Only `gemini-embedding-2` is multimodal. So this
  provider sets `supports_images = False`, the catalogue stores no image vector for it,
  and image search against this index is refused rather than faked.
* **Truncated output is not normalised.** Google returns unit-length vectors only at
  the native 3072 dims; at any `outputDimensionality` below that you must L2-normalise
  yourself, or every cosine comparison is silently wrong.
"""

from collections.abc import Sequence
from typing import Any

import httpx

from app.core.errors import EmbeddingError
from app.core.logging import get_logger
from app.embeddings.base import (
    BATCH_SIZE,
    EmbeddingProvider,
    InputKind,
    Vector,
    chunked,
    l2_normalize,
)
from app.embeddings.http import RateLimiter, post_json

log = get_logger(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
_TASK_BY_KIND = {"query": "RETRIEVAL_QUERY", "document": "RETRIEVAL_DOCUMENT"}
NATIVE_DIM = 3072


class GeminiProvider(EmbeddingProvider):
    name = "gemini"
    supports_images = False
    supports_cross_modal = False

    def __init__(
        self,
        dim: int,
        api_key: str,
        model: str,
        timeout: float,
        requests_per_minute: float = 0.0,
        image_batch_size: int | None = None,
    ) -> None:
        super().__init__(dim, image_batch_size)
        self._model = model
        self._limiter = RateLimiter(requests_per_minute)
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        )

    async def embed_texts(self, texts: Sequence[str], kind: InputKind = "document") -> list[Vector]:
        vectors: list[Vector] = []
        for batch in chunked(texts, BATCH_SIZE):
            vectors.extend(await self._post(list(batch), kind))
        return self._validate_dim(vectors)

    async def _post(self, batch: list[str], kind: InputKind) -> list[Vector]:
        payload: dict[str, Any] = {
            "requests": [
                {
                    "model": f"models/{self._model}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": _TASK_BY_KIND[kind],
                    "outputDimensionality": self.dim,
                }
                for text in batch
            ]
        }
        url = f"{API_ROOT}/{self._model}:batchEmbedContents"

        body = await post_json(
            client=self._client,
            url=url,
            payload=payload,
            provider=self.name,
            limiter=self._limiter,
        )
        return self._parse(body, len(batch), self.dim)

    @staticmethod
    def _parse(body: dict[str, Any], expected: int, dim: int) -> list[Vector]:
        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != expected:
            raise EmbeddingError(
                f"Gemini returned {len(embeddings) if isinstance(embeddings, list) else 'no'} "
                f"embeddings for {expected} inputs"
            )
        vectors = [[float(x) for x in item["values"]] for item in embeddings]
        # Only the native 3072-d output arrives unit-length; anything truncated must be
        # renormalised here or cosine distance in pgvector is meaningless.
        if dim != NATIVE_DIM:
            vectors = [l2_normalize(vec) for vec in vectors]
        return vectors

    async def aclose(self) -> None:
        await self._client.aclose()
