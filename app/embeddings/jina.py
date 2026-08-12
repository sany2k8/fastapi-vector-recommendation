"""Jina Embeddings v4 — text and images in one shared vector space.

Docs: https://api.jina.ai/redoc  (POST https://api.jina.ai/v1/embeddings)

Input items are `{"text": ...}` or `{"image": <url|base64>}`; `dimensions` truncates
the 2048-d output via Matryoshka representation learning, and `task` selects the
asymmetric query/passage heads that matter for retrieval quality.
"""

import base64
from collections.abc import Sequence
from typing import Any

import httpx

from app.core.errors import EmbeddingError
from app.core.logging import get_logger
from app.embeddings.base import BATCH_SIZE, EmbeddingProvider, InputKind, Vector, chunked
from app.embeddings.http import RateLimiter, post_json

log = get_logger(__name__)

API_URL = "https://api.jina.ai/v1/embeddings"
_TASK_BY_KIND = {"query": "retrieval.query", "document": "retrieval.passage"}


class JinaProvider(EmbeddingProvider):
    name = "jina"
    supports_cross_modal = True

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
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def embed_texts(self, texts: Sequence[str], kind: InputKind = "document") -> list[Vector]:
        return await self._embed([{"text": t} for t in texts], kind, BATCH_SIZE)

    async def embed_images(
        self, images: Sequence[bytes], kind: InputKind = "document"
    ) -> list[Vector]:
        items = [{"image": base64.b64encode(b).decode("ascii")} for b in images]
        return await self._embed(items, kind, self.image_batch_size)

    async def _embed(
        self, items: list[dict[str, str]], kind: InputKind, batch_size: int
    ) -> list[Vector]:
        vectors: list[Vector] = []
        for batch in chunked(items, batch_size):
            vectors.extend(await self._post(list(batch), kind))
        return self._validate_dim(vectors)

    async def _post(self, batch: list[dict[str, str]], kind: InputKind) -> list[Vector]:
        payload: dict[str, Any] = {
            "model": self._model,
            "task": _TASK_BY_KIND[kind],
            "input": batch,
            "dimensions": self.dim,
            "normalized": True,
            "embedding_type": "float",
        }
        body = await post_json(
            client=self._client,
            url=API_URL,
            payload=payload,
            provider=self.name,
            limiter=self._limiter,
        )
        return self._parse(body, len(batch))

    @staticmethod
    def _parse(body: dict[str, Any], expected: int) -> list[Vector]:
        data = body.get("data")
        if not isinstance(data, list) or len(data) != expected:
            raise EmbeddingError(
                f"Jina returned {len(data) if isinstance(data, list) else 'no'} "
                f"embeddings for {expected} inputs"
            )
        # The API does not guarantee ordering, so index each item back into place.
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        return [[float(x) for x in item["embedding"]] for item in ordered]

    async def aclose(self) -> None:
        await self._client.aclose()
