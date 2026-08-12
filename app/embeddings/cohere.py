"""Cohere embed-v4.0 — text and images share one 1536-d space (truncatable).

Docs: https://docs.cohere.com/docs/multimodal-embeddings (POST https://api.cohere.com/v2/embed)

v2 takes a unified `inputs` list where each entry has a `content` array of
`{"type": "text"}` / `{"type": "image_url"}` parts. Images must arrive as base64
data URLs of at most 5 MB.
"""

import base64
import io
from collections.abc import Sequence
from typing import Any

import httpx
from PIL import Image

from app.core.errors import EmbeddingError
from app.core.logging import get_logger
from app.embeddings.base import BATCH_SIZE, EmbeddingProvider, InputKind, Vector, chunked
from app.embeddings.http import RateLimiter, post_json

log = get_logger(__name__)

API_URL = "https://api.cohere.com/v2/embed"
_INPUT_TYPE_BY_KIND = {"query": "search_query", "document": "search_document"}
_MIME_BY_FORMAT = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}


def to_data_url(payload: bytes) -> str:
    """Wrap raw image bytes in a base64 data URL, sniffing the format with Pillow."""
    try:
        with Image.open(io.BytesIO(payload)) as img:
            fmt = (img.format or "").upper()
    except Exception as exc:
        raise EmbeddingError(f"could not decode image: {exc}") from exc

    mime = _MIME_BY_FORMAT.get(fmt)
    if mime is None:
        raise EmbeddingError(f"unsupported image format {fmt!r}; use JPEG, PNG, WEBP or GIF")
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


class CohereProvider(EmbeddingProvider):
    name = "cohere"
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
            {"content": [{"type": "image_url", "image_url": {"url": to_data_url(b)}}]}
            for b in images
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
            "embedding_types": ["float"],
            "output_dimension": self.dim,
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
        embeddings = body.get("embeddings") or {}
        # The SDK exposes this as `float_`; the raw HTTP field is `float`.
        floats = embeddings.get("float") or embeddings.get("float_")
        if not isinstance(floats, list) or len(floats) != expected:
            raise EmbeddingError(
                f"Cohere returned {len(floats) if isinstance(floats, list) else 'no'} "
                f"embeddings for {expected} inputs"
            )
        return [[float(x) for x in vec] for vec in floats]

    async def aclose(self) -> None:
        await self._client.aclose()
