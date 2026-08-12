"""Voyage AI multimodal embeddings — text and images in one shared space.

Docs: https://docs.voyageai.com/reference/multimodal-embeddings-api
POST https://api.voyageai.com/v1/multimodalembeddings

`inputs` is a list of `{"content": [...]}` where each part is `{"type": "text", "text": …}`
or `{"type": "image_base64", "image_base64": "data:image/jpeg;base64,…"}`.

**Dimension caveat.** `voyage-multimodal-3.5` offers 256/512/1024/2048; the older
`voyage-multimodal-3` is fixed at 1024 with no truncation. The `output_dimension`
parameter is not documented in the API reference, so it is only sent when set — and if
the service ignores it, `_validate_dim` in the base class turns the mismatch into a
loud error instead of a corrupt index. With `voyage-multimodal-3` you must run the whole
stack at `EMBEDDING_DIM=1024`.
"""

from collections.abc import Sequence
from typing import Any

import httpx

from app.core.errors import EmbeddingError
from app.core.logging import get_logger
from app.embeddings.base import BATCH_SIZE, EmbeddingProvider, InputKind, Vector, chunked
from app.embeddings.cohere import to_data_url
from app.embeddings.http import RateLimiter, post_json

log = get_logger(__name__)

API_URL = "https://api.voyageai.com/v1/multimodalembeddings"
_INPUT_TYPE_BY_KIND = {"query": "query", "document": "document"}
#: Models that accept a configurable output width. Others emit their native size.
_TRUNCATABLE_MODELS = {"voyage-multimodal-3.5"}


class VoyageProvider(EmbeddingProvider):
    name = "voyage"
    supports_images = True
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
        inputs = [{"content": [{"type": "text", "text": t}]} for t in texts]
        return await self._embed(inputs, kind, BATCH_SIZE)

    async def embed_images(
        self, images: Sequence[bytes], kind: InputKind = "document"
    ) -> list[Vector]:
        inputs = [
            {"content": [{"type": "image_base64", "image_base64": to_data_url(b)}]} for b in images
        ]
        return await self._embed(inputs, kind, self.image_batch_size)

    async def _embed(
        self, inputs: list[dict[str, Any]], kind: InputKind, batch_size: int
    ) -> list[Vector]:
        vectors: list[Vector] = []
        for batch in chunked(inputs, batch_size):
            vectors.extend(await self._post(list(batch), kind))
        return self._validate_dim(vectors)

    async def _post(self, batch: list[dict[str, Any]], kind: InputKind) -> list[Vector]:
        payload: dict[str, Any] = {
            "model": self._model,
            "inputs": batch,
            "input_type": _INPUT_TYPE_BY_KIND[kind],
        }
        if self._model in _TRUNCATABLE_MODELS:
            payload["output_dimension"] = self.dim

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
                f"Voyage returned {len(data) if isinstance(data, list) else 'no'} "
                f"embeddings for {expected} inputs"
            )
        # Ordering is not guaranteed, so index each item back into place.
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        return [[float(x) for x in item["embedding"]] for item in ordered]

    async def aclose(self) -> None:
        await self._client.aclose()
