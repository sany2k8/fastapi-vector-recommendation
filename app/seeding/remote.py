"""Fetching a product feed and its photos from a remote source."""

import json
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.errors import ValidationError
from app.core.logging import get_logger
from app.core.net import fetch
from app.schemas.importing import FieldMapping
from app.seeding.importers import ProductDraft, drafts_from_remote

log = get_logger(__name__)


class RemoteSource:
    """A guarded client for one import run.

    Holds a single HTTP client so the feed and every image share connection pooling,
    and applies the SSRF and size checks in `app.core.net` to each request.
    """

    def __init__(self, timeout: float | None = None) -> None:
        settings = get_settings()
        self._settings = settings
        self._client = httpx.AsyncClient(
            timeout=timeout or settings.import_timeout_seconds,
            headers={"User-Agent": "vector-recsys-importer/1.0"},
        )

    async def fetch_feed(self, url: str) -> Any:
        body, content_type = await fetch(
            url,
            client=self._client,
            max_bytes=self._settings.max_import_bytes,
            accept="application/json",
        )
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"{url} did not return JSON (content-type {content_type or 'unknown'}): {exc}"
            ) from exc

    async def fetch_image(self, url: str) -> bytes:
        body, content_type = await fetch(
            url, client=self._client, max_bytes=self._settings.max_image_bytes, accept="image/*"
        )
        if content_type and not content_type.startswith("image/"):
            raise ValidationError(f"{url} returned {content_type}, not an image")
        return body

    async def drafts(
        self, url: str, mapping: FieldMapping, *, prefix: str, limit: int
    ) -> list[ProductDraft]:
        body = await self.fetch_feed(url)
        drafts = drafts_from_remote(body, mapping, prefix=prefix, limit=limit)
        log.info("import.fetched", url=url, count=len(drafts))
        return drafts

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "RemoteSource":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
