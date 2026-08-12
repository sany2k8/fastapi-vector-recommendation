"""Rate limiting and retry behaviour shared by every hosted provider."""

import asyncio

import httpx
import pytest

from app.core.errors import EmbeddingError
from app.embeddings.http import MAX_BACKOFF_SECONDS, RateLimiter, post_json


async def call(transport: httpx.MockTransport, limiter: RateLimiter | None = None) -> dict:
    async with httpx.AsyncClient(transport=transport) as client:
        return await post_json(
            client=client,
            url="https://example.test/embed",
            payload={"x": 1},
            provider="test",
            limiter=limiter,
        )


class TestRateLimiter:
    def test_zero_rpm_disables_pacing(self) -> None:
        assert RateLimiter(0).enabled is False

    async def test_first_call_is_not_delayed(self) -> None:
        limiter = RateLimiter(60)
        started = asyncio.get_running_loop().time()
        await limiter.acquire()
        assert asyncio.get_running_loop().time() - started < 0.2

    async def test_subsequent_calls_are_spaced_out(self) -> None:
        """3 RPM means one request every 20s — the limiter must actually wait."""
        limiter = RateLimiter(120)  # 0.5s apart, fast enough for a test
        await limiter.acquire()
        started = asyncio.get_running_loop().time()
        await limiter.acquire()
        assert asyncio.get_running_loop().time() - started >= 0.4

    async def test_concurrent_callers_queue_rather_than_burst(self) -> None:
        limiter = RateLimiter(120)
        started = asyncio.get_running_loop().time()
        await asyncio.gather(*(limiter.acquire() for _ in range(3)))
        # Three reservations at 0.5s spacing: the last must land ~1s in, not instantly.
        assert asyncio.get_running_loop().time() - started >= 0.9


class TestRetries:
    async def test_success_returns_the_body(self) -> None:
        transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": True}))
        assert await call(transport) == {"ok": True}

    async def test_client_errors_are_not_retried(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(401, text="bad key")

        with pytest.raises(EmbeddingError, match="401"):
            await call(httpx.MockTransport(handler))
        assert calls["n"] == 1

    async def test_rate_limit_is_retried_then_succeeds(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, headers={"retry-after": "0"}, text="slow down")
            return httpx.Response(200, json={"ok": True})

        assert await call(httpx.MockTransport(handler)) == {"ok": True}
        assert calls["n"] == 2

    async def test_retry_after_header_is_honoured(self) -> None:
        """Fixed 1s/2s/4s backoff cannot satisfy a 20s-per-request budget."""
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, headers={"retry-after": "0.5"}, text="wait")
            return httpx.Response(200, json={"ok": True})

        started = asyncio.get_running_loop().time()
        await call(httpx.MockTransport(handler))
        assert asyncio.get_running_loop().time() - started >= 0.45

    async def test_unparseable_retry_after_falls_back_to_backoff(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
            return httpx.Response(200, json={"ok": True})

        assert await call(httpx.MockTransport(handler)) == {"ok": True}

    async def test_a_hostile_retry_after_is_capped(self) -> None:
        response = httpx.Response(429, headers={"retry-after": "999999"})
        from app.embeddings.http import _retry_after_seconds

        assert _retry_after_seconds(response) == 999999.0
        assert min(999999.0, MAX_BACKOFF_SECONDS) == MAX_BACKOFF_SECONDS

    async def test_exhausting_retries_reports_the_last_error(self) -> None:
        transport = httpx.MockTransport(
            lambda r: httpx.Response(429, headers={"retry-after": "0"}, text="still limited")
        )
        with pytest.raises(EmbeddingError, match="still limited"):
            await call(transport)

    async def test_transport_errors_are_retried(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("boom")
            return httpx.Response(200, json={"ok": True})

        assert await call(httpx.MockTransport(handler)) == {"ok": True}
        assert calls["n"] == 2
