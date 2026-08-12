"""Admin endpoints: status, background jobs, and the destructive actions."""

import asyncio

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.integration


async def wait_for(client: AsyncClient, job_id: str, *, timeout: float = 60.0) -> dict:
    """Poll a background job to completion."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        body = (await client.get(f"/api/v1/admin/jobs/{job_id}")).json()
        if body["status"] in ("succeeded", "failed"):
            return body
        await asyncio.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


class TestStatus:
    async def test_reports_providers_and_catalogue_size(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/admin/status")).json()
        assert body["embedding_dim"] > 0
        assert isinstance(body["providers"], list)

        names = {p["name"] for p in body["providers"]}
        assert "local_hash" in names

        local = next(p for p in body["providers"] if p["name"] == "local_hash")
        assert local["configured"] is True
        assert local["cross_modal"] is False

    async def test_unconfigured_providers_are_flagged_not_hidden(self, client: AsyncClient) -> None:
        """Every known provider must be listed even with no API key.

        Hiding them makes the panel useless for discovering what could be enabled, and
        `configured: false` already carries the distinction.
        """
        from app.core.config import ALL_PROVIDERS

        body = (await client.get("/api/v1/admin/status")).json()
        listed = {p["name"] for p in body["providers"]}
        assert set(ALL_PROVIDERS) <= listed

        # Keys are blanked for the whole test session, so only local_hash is usable.
        by_name = {p["name"]: p for p in body["providers"]}
        assert by_name["voyage"]["configured"] is False
        assert by_name["local_hash"]["configured"] is True

    async def test_capabilities_are_reported_per_provider(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/admin/status")).json()
        by_name = {p["name"]: p for p in body["providers"]}

        # Gemini embedding-001 is text-only; Voyage is fully multimodal.
        assert by_name["gemini"]["supports_images"] is False
        assert by_name["gemini"]["cross_modal"] is False
        assert by_name["voyage"]["supports_images"] is True
        assert by_name["voyage"]["cross_modal"] is True


class TestJobs:
    async def test_unknown_job_is_404(self, client: AsyncClient) -> None:
        assert (await client.get("/api/v1/admin/jobs/nope")).status_code == 404

    async def test_reindex_runs_as_a_background_job(self, client: AsyncClient) -> None:
        response = await client.post("/api/v1/admin/reindex", json={"providers": ["local_hash"]})
        assert response.status_code == 202

        job = response.json()
        assert job["kind"] == "reindex"
        finished = await wait_for(client, job["id"])
        assert finished["status"] == "succeeded", finished["error"]
        assert "local_hash" in finished["detail"]

    async def test_rejects_a_provider_with_no_credentials(self, client: AsyncClient) -> None:
        """Silently falling back to the default would make an A/B quietly meaningless."""
        response = await client.post("/api/v1/admin/reindex", json={"providers": ["jina"]})
        assert response.status_code == 422
        assert "not configured" in response.json()["error"]["message"]

    async def test_rejects_an_unknown_provider(self, client: AsyncClient) -> None:
        response = await client.post("/api/v1/admin/reindex", json={"providers": ["banana"]})
        assert response.status_code == 422

    async def test_second_job_is_refused_while_one_runs(self, client: AsyncClient) -> None:
        """Overlapping seeds and re-indexes would interleave into a half-written index."""
        from app.services.jobs import get_job_registry

        registry = get_job_registry()
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow(job) -> str:  # type: ignore[no-untyped-def]
            started.set()
            await release.wait()
            return "done"

        blocker = registry.start("reindex", slow)
        await started.wait()
        try:
            response = await client.post(
                "/api/v1/admin/reindex", json={"providers": ["local_hash"]}
            )
            assert response.status_code == 409
            assert response.json()["error"]["code"] == "job_in_progress"
        finally:
            release.set()
            await wait_for(client, blocker.id, timeout=5)

    async def test_failing_job_is_recorded_not_swallowed(self, client: AsyncClient) -> None:
        from app.services.jobs import get_job_registry

        async def boom(job) -> str:  # type: ignore[no-untyped-def]
            raise RuntimeError("kaboom")

        job = get_job_registry().start("reindex", boom)
        finished = await wait_for(client, job.id, timeout=5)
        assert finished["status"] == "failed"
        assert "kaboom" in finished["error"]


class TestAdminDisabled:
    async def test_endpoints_refuse_when_admin_is_off(self, client: AsyncClient) -> None:
        from app.core.config import get_settings

        settings = get_settings()
        settings.admin_enabled = False
        try:
            response = await client.get("/api/v1/admin/status")
            assert response.status_code == 422
            assert "disabled" in response.json()["error"]["message"]
        finally:
            settings.admin_enabled = True
