"""Admin endpoints: status, background jobs, and the destructive actions."""

import asyncio
import uuid

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


class TestOfflineImport:
    """Adding products with no network: the admin supplies text, we draw the images."""

    @staticmethod
    def _payload(tag: str, **over: object) -> dict:
        return {
            "products": [
                {
                    "name": f"Copper Pour-Over Kettle {tag}",
                    "category": "kettles",
                    "description": "Gooseneck kettle with a thermometer for pour-over coffee",
                    "brand": "Halden",
                    "price": 74.5,
                    "color": "copper",
                },
                {
                    "name": f"Slate Yoga Mat {tag}",
                    "category": "fitness",
                    "description": "Dense closed-cell yoga mat with an alignment stripe",
                    "brand": "Volta",
                    "price": 62.0,
                    "color": "slate",
                },
            ],
            "providers": ["local_hash"],
            "sku_prefix": tag,
            **over,
        }

    async def test_products_are_created_and_indexed_without_network(
        self, client: AsyncClient
    ) -> None:
        tag = uuid.uuid4().hex[:6].upper()
        r = await client.post("/api/v1/admin/import/offline", json=self._payload(tag))
        assert r.status_code == 202

        job = await wait_for(client, r.json()["id"])
        assert job["status"] == "succeeded", job["error"]
        assert "imported 2" in job["detail"]

        listing = (await client.get("/api/v1/products", params={"category": "kettles"})).json()
        names = [i["name"] for i in listing["items"]]
        assert any(tag in n for n in names)
        assert all(i["has_image"] for i in listing["items"] if tag in i["name"])

    async def test_imported_products_are_searchable(self, client: AsyncClient) -> None:
        tag = uuid.uuid4().hex[:6].upper()
        job = await wait_for(
            client,
            (await client.post("/api/v1/admin/import/offline", json=self._payload(tag))).json()[
                "id"
            ],
        )
        assert job["status"] == "succeeded", job["error"]

        # Query the run's unique tag: jobs commit outside the test transaction, so runs
        # can leave near-identical products behind and a generic query would be ranking
        # them against each other rather than testing retrievability.
        body = (
            await client.post(
                "/api/v1/search/text",
                json={
                    "query": f"Copper Pour-Over Kettle {tag} gooseneck thermometer",
                    "top_k": 5,
                    "provider": "local_hash",
                    "min_score_ratio": 0.0,
                },
            )
        ).json()
        assert any(tag in h["product"]["name"] for h in body["hits"])

    async def test_unknown_categories_and_colours_still_render(self, client: AsyncClient) -> None:
        """Imported catalogues use words the built-in palette has never seen."""
        from app.seeding.images import colour_for, render

        a, b = render("quantum-widgets", "ultraviolet"), render("quantum-widgets", "ochre")
        assert a[:2] == b"\xff\xd8" and len(a) > 1000  # a real JPEG
        assert a != b, "different colours must produce different images"
        assert colour_for("ultraviolet") == colour_for("ultraviolet")  # deterministic
        assert colour_for("ultraviolet") != colour_for("ochre")

    async def test_re_importing_skips_rather_than_duplicating(self, client: AsyncClient) -> None:
        tag = uuid.uuid4().hex[:6].upper()
        first = await wait_for(
            client,
            (await client.post("/api/v1/admin/import/offline", json=self._payload(tag))).json()[
                "id"
            ],
        )
        assert "imported 2" in first["detail"]

        second = await wait_for(
            client,
            (await client.post("/api/v1/admin/import/offline", json=self._payload(tag))).json()[
                "id"
            ],
        )
        assert "imported 0" in second["detail"]
        assert "skipped 2" in second["detail"]

    async def test_an_unconfigured_provider_is_rejected(self, client: AsyncClient) -> None:
        r = await client.post(
            "/api/v1/admin/import/offline", json=self._payload("X", providers=["jina"])
        )
        assert r.status_code == 422


class TestRemoteImport:
    async def test_presets_are_listed_with_their_mapping(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/admin/import/presets")).json()
        assert "dummyjson" in body["presets"]
        assert body["presets"]["dummyjson"]["mapping"]["name"] == "title"

    async def test_a_private_url_is_refused_before_any_fetch(self, client: AsyncClient) -> None:
        r = await client.post(
            "/api/v1/admin/import/remote/preview",
            json={"url": "http://127.0.0.1:5439/products", "limit": 5},
        )
        assert r.status_code == 422
        assert "non-public" in r.json()["error"]["message"]

    async def test_unknown_preset_is_rejected(self, client: AsyncClient) -> None:
        r = await client.post("/api/v1/admin/import/remote/preview", json={"preset": "nope"})
        assert r.status_code == 422
        assert "unknown preset" in r.json()["error"]["message"]

    async def test_missing_url_and_preset_is_rejected(self, client: AsyncClient) -> None:
        r = await client.post("/api/v1/admin/import/remote/preview", json={"limit": 5})
        assert r.status_code == 422


class TestImageNormalisation:
    async def test_oversized_uploads_are_downscaled_before_storage(
        self, client: AsyncClient
    ) -> None:
        """Voyage bills image embeddings by pixel, so huge photos cost real money."""
        import io

        from PIL import Image

        from app.core.config import get_settings
        from app.services.images import normalize_image

        buf = io.BytesIO()
        Image.new("RGB", (3000, 2000), (200, 40, 40)).save(buf, format="JPEG")
        canonical, suffix = normalize_image(buf.getvalue())

        with Image.open(io.BytesIO(canonical)) as out:
            assert max(out.size) == get_settings().max_image_dimension
        assert suffix == ".jpg"
        assert len(canonical) < len(buf.getvalue())

    async def test_small_images_are_left_untouched(self, client: AsyncClient) -> None:
        import io

        from PIL import Image

        from app.services.images import normalize_image

        buf = io.BytesIO()
        Image.new("RGB", (200, 200), (10, 80, 200)).save(buf, format="PNG")
        canonical, suffix = normalize_image(buf.getvalue())
        assert canonical == buf.getvalue()
        assert suffix == ".png"


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
