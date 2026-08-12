"""Shared HTTP plumbing for hosted embedding providers: rate limiting and retries.

Free tiers are strict — Voyage without a payment method allows **3 requests per minute**
— so two things matter more than raw speed:

* **Pace requests before sending them.** A client-side limiter turns a guaranteed 429
  into a wait, which is far cheaper than burning a retry budget discovering the limit.
* **Honour `Retry-After`.** A fixed exponential backoff of 1s/2s/4s cannot satisfy a
  20-second-per-request budget; the server tells us how long to wait, so we listen.
"""

import asyncio
from typing import Any

import httpx

from app.core.errors import EmbeddingError
from app.core.logging import get_logger

log = get_logger(__name__)

MAX_ATTEMPTS = 4
#: Status codes worth retrying. Everything else (401, 400, 422) is a bug or a bad key.
RETRYABLE = frozenset({408, 429, 500, 502, 503, 504})
#: Cap on a single sleep, so a hostile Retry-After cannot hang a job indefinitely.
MAX_BACKOFF_SECONDS = 90.0


class RateLimiter:
    """Spaces requests at least `interval` apart. One instance per provider."""

    def __init__(self, requests_per_minute: float) -> None:
        self._interval = 0.0 if requests_per_minute <= 0 else 60.0 / requests_per_minute
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    @property
    def enabled(self) -> bool:
        return self._interval > 0.0

    @property
    def interval(self) -> float:
        return self._interval

    async def acquire(self) -> None:
        if not self.enabled:
            return
        # The lock serialises the reservation so concurrent callers queue up rather than
        # all reading the same "next allowed" instant and firing together.
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = loop.time()
            self._next_allowed = now + self._interval


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None  # HTTP-date form; fall back to exponential backoff


async def post_json(
    *,
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    provider: str,
    limiter: RateLimiter | None = None,
) -> dict[str, Any]:
    """POST with pacing and retries, returning the decoded JSON body."""
    last_error = "unknown error"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if limiter is not None:
            await limiter.acquire()

        delay: float | None = None
        try:
            response = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            last_error = f"transport error: {exc}"
        else:
            if response.status_code == 200:
                body: dict[str, Any] = response.json()
                return body

            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code not in RETRYABLE:
                break
            delay = _retry_after_seconds(response)

        if attempt < MAX_ATTEMPTS:
            # Without a Retry-After, exponential backoff alone is useless against a
            # per-minute quota: 2s/4s/8s all land inside the same window. Fall back to
            # at least one full pacing interval so the retry lands in the next one.
            floor = limiter.interval if limiter is not None else 0.0
            fallback = max(float(2**attempt), floor)
            sleep_for = min(delay if delay is not None else fallback, MAX_BACKOFF_SECONDS)
            log.warning(
                "embedding.retry",
                provider=provider,
                attempt=attempt,
                sleep_seconds=round(sleep_for, 1),
                error=last_error,
            )
            await asyncio.sleep(sleep_for)

    raise EmbeddingError(f"{provider} embedding request failed: {last_error}")
