"""End-to-end API behaviour against a real Postgres + pgvector."""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.embeddings import get_provider
from app.models import ProductEmbedding
from app.repositories import EmbeddingRepository, ProductRepository
from app.services import CatalogService
from tests.integration.conftest import ImageFactory

pytestmark = pytest.mark.integration


async def create_product(
    client: AsyncClient,
    *,
    sku: str,
    name: str,
    category: str,
    description: str = "",
    brand: str = "TestBrand",
    price: float = 50.0,
    image: bytes | None = None,
) -> dict:
    data = {
        "sku": sku,
        "name": name,
        "category": category,
        "description": description,
        "brand": brand,
        "price": str(price),
    }
    files = {"image": ("p.jpg", image, "image/jpeg")} if image else None
    response = await client.post("/api/v1/products", data=data, files=files)
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture
async def catalogue(client: AsyncClient, product_image: ImageFactory) -> dict[str, dict]:
    """A tiny catalogue with deliberately separated text and colour clusters."""
    tag = uuid.uuid4().hex[:8]
    return {
        "jacket": await create_product(
            client,
            sku=f"T-JKT-{tag}",
            name="Waterproof Storm Jacket",
            category="jackets",
            description="Fully waterproof seam-sealed rain jacket for storms and winter hiking",
            image=product_image((30, 70, 190)),
        ),
        "jacket2": await create_product(
            client,
            sku=f"T-JKT2-{tag}",
            name="Rain Shell Jacket",
            category="jackets",
            description="Packable waterproof rain shell jacket with taped seams for wet weather",
            image=product_image((40, 80, 200)),
        ),
        "mug": await create_product(
            client,
            sku=f"T-MUG-{tag}",
            name="Stoneware Coffee Mug",
            category="mugs",
            description="Hand thrown speckled stoneware coffee mug with a reactive ceramic glaze",
            image=product_image((210, 60, 40), shape="ellipse"),
        ),
        "no_image": await create_product(
            client,
            sku=f"T-NOI-{tag}",
            name="Waterproof Trail Trousers",
            category="trousers",
            description="Waterproof trail trousers for storms and winter hiking in wet weather",
        ),
    }


class TestHealth:
    async def test_liveness(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    async def test_readiness_reports_pgvector_and_provider(self, client: AsyncClient) -> None:
        body = (await client.get("/health/ready")).json()
        assert body["database"] == "reachable"
        assert body["pgvector"] != "missing"
        assert body["embedding_dim"] > 0
        assert "cross_modal" in body


class TestProducts:
    async def test_create_with_image_embeds_both_modalities(
        self, client: AsyncClient, product_image: ImageFactory
    ) -> None:
        body = await create_product(
            client,
            sku=f"T-{uuid.uuid4().hex[:8]}",
            name="Alpine Puffer",
            category="jackets",
            image=product_image(),
        )
        assert body["has_image"] is True
        assert body["image_url"].startswith("/images/")

    async def test_create_without_image_still_succeeds(self, client: AsyncClient) -> None:
        body = await create_product(
            client, sku=f"T-{uuid.uuid4().hex[:8]}", name="Plain Item", category="misc"
        )
        assert body["has_image"] is False
        assert body["image_url"] is None

    async def test_duplicate_sku_is_rejected(self, client: AsyncClient) -> None:
        sku = f"T-{uuid.uuid4().hex[:8]}"
        await create_product(client, sku=sku, name="First", category="misc")
        response = await client.post(
            "/api/v1/products", data={"sku": sku, "name": "Second", "category": "misc"}
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    async def test_rejects_a_file_that_is_not_an_image(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/products",
            data={"sku": f"T-{uuid.uuid4().hex[:8]}", "name": "Bad", "category": "misc"},
            files={"image": ("x.jpg", b"totally not an image", "image/jpeg")},
        )
        assert response.status_code == 422

    async def test_unknown_product_is_404(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/v1/products/{uuid.uuid4()}")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"

    async def test_list_can_filter_by_category(
        self, client: AsyncClient, catalogue: dict[str, dict]
    ) -> None:
        body = (await client.get("/api/v1/products", params={"category": "mugs"})).json()
        assert all(item["category"] == "mugs" for item in body["items"])
        assert body["total"] >= 1

    async def test_delete_removes_the_product(
        self, client: AsyncClient, catalogue: dict[str, dict]
    ) -> None:
        pid = catalogue["mug"]["id"]
        assert (await client.delete(f"/api/v1/products/{pid}")).status_code == 204
        assert (await client.get(f"/api/v1/products/{pid}")).status_code == 404

    async def test_image_can_be_attached_after_creation(
        self, client: AsyncClient, product_image: ImageFactory
    ) -> None:
        created = await create_product(
            client, sku=f"T-{uuid.uuid4().hex[:8]}", name="Late Photo", category="misc"
        )
        response = await client.put(
            f"/api/v1/products/{created['id']}/image",
            files={"image": ("p.jpg", product_image(), "image/jpeg")},
        )
        assert response.status_code == 200
        assert response.json()["has_image"] is True


class TestReindex:
    @staticmethod
    async def _vectors(session: AsyncSession, product_id: uuid.UUID, provider: str):
        result = await session.execute(
            select(ProductEmbedding).where(
                ProductEmbedding.product_id == product_id,
                ProductEmbedding.provider == provider,
            )
        )
        return result.scalar_one_or_none()

    async def test_reembedding_covers_every_product_and_is_stable(
        self, client: AsyncClient, session: AsyncSession, catalogue: dict[str, dict]
    ) -> None:
        """Re-embedding with the same provider must be a no-op, not a drift."""
        own_ids = [uuid.UUID(p["id"]) for p in catalogue.values()]
        before = {}
        for pid in own_ids:
            row = await self._vectors(session, pid, "local_hash")
            assert row is not None
            before[pid] = (list(row.text_embedding or []), list(row.fused_embedding or []))

        count = await CatalogService(session, get_provider("local_hash")).reembed_all(batch_size=2)
        assert count >= len(catalogue)
        session.expire_all()

        for pid in own_ids:
            row = await self._vectors(session, pid, "local_hash")
            assert row is not None
            old_text, old_fused = before[pid]
            assert list(row.text_embedding or []) == pytest.approx(old_text, abs=1e-6)
            assert list(row.fused_embedding or []) == pytest.approx(old_fused, abs=1e-6)

    async def test_reembedding_survives_a_missing_image_file(
        self, client: AsyncClient, session: AsyncSession, catalogue: dict[str, dict]
    ) -> None:
        """A vanished image must cost the product its image vector, not the whole run."""
        product_id = uuid.UUID(catalogue["jacket"]["id"])
        target = await ProductRepository(session).get(product_id)
        assert target is not None and target.image_path
        (get_settings().image_dir / target.image_path).unlink()

        count = await CatalogService(session, get_provider("local_hash")).reembed_all()
        assert count >= len(catalogue)
        session.expire_all()

        row = await self._vectors(session, product_id, "local_hash")
        assert row is not None
        assert row.image_embedding is None
        assert row.fused_embedding is not None  # still retrievable on text alone

    async def test_reindexing_one_provider_leaves_the_others_intact(
        self, client: AsyncClient, session: AsyncSession, catalogue: dict[str, dict]
    ) -> None:
        """The whole point of the per-provider table: indexes must not clobber each other."""
        product_id = uuid.UUID(catalogue["jacket"]["id"])
        embeddings = EmbeddingRepository(session)

        # Stand in for a second provider's index without spending anyone's API quota.
        sentinel = [0.0] * (get_settings().embedding_dim - 1) + [1.0]
        await embeddings.upsert(
            product_id=product_id,
            provider="pretend_hosted",
            model="pretend-v1",
            dim=get_settings().embedding_dim,
            text_embedding=sentinel,
            image_embedding=None,
            fused_embedding=sentinel,
        )
        await session.flush()

        await CatalogService(session, get_provider("local_hash")).reembed_all()
        session.expire_all()

        survivor = await self._vectors(session, product_id, "pretend_hosted")
        assert survivor is not None
        assert list(survivor.fused_embedding or []) == pytest.approx(sentinel, abs=1e-6)

        coverage = await embeddings.coverage()
        assert "pretend_hosted" in coverage
        assert coverage["local_hash"]["products"] >= len(catalogue)

    async def test_clearing_one_index_keeps_products_and_other_indexes(
        self, client: AsyncClient, session: AsyncSession, catalogue: dict[str, dict]
    ) -> None:
        embeddings = EmbeddingRepository(session)
        product_id = uuid.UUID(catalogue["jacket"]["id"])
        await embeddings.upsert(
            product_id=product_id,
            provider="pretend_hosted",
            model="pretend-v1",
            dim=get_settings().embedding_dim,
            text_embedding=[0.0] * get_settings().embedding_dim,
            image_embedding=None,
            fused_embedding=[0.0] * get_settings().embedding_dim,
        )
        await session.flush()

        removed = await embeddings.clear_provider("pretend_hosted")
        assert removed == 1
        assert "pretend_hosted" not in await embeddings.coverage()
        assert await ProductRepository(session).get(product_id) is not None
        assert (await embeddings.count_for("local_hash")) >= len(catalogue)


class TestBatchedReindex:
    """Re-indexing must be cheap in requests and must not lose work on failure."""

    @staticmethod
    def _counting_provider():  # type: ignore[no-untyped-def]
        """Wraps local_hash to count how many embed calls (i.e. requests) are made."""
        from app.embeddings.local_hash import LocalHashProvider

        class Counting(LocalHashProvider):
            def __init__(self, dim: int) -> None:
                super().__init__(dim)
                self.text_calls = 0
                self.image_calls = 0
                self.fail_after_batches: int | None = None

            async def embed_texts(self, texts, kind="document"):  # type: ignore[no-untyped-def]
                self.text_calls += 1
                if (
                    self.fail_after_batches is not None
                    and self.text_calls > self.fail_after_batches
                ):
                    raise RuntimeError("simulated rate limit")
                return await super().embed_texts(texts, kind)

            async def embed_images(self, images, kind="document"):  # type: ignore[no-untyped-def]
                self.image_calls += 1
                return await super().embed_images(images, kind)

        return Counting(get_settings().embedding_dim)

    async def test_a_batch_costs_one_request_per_modality(
        self, client: AsyncClient, session: AsyncSession, catalogue: dict[str, dict]
    ) -> None:
        """Per-product embedding made 2 requests each — fatal on a 3-RPM key."""
        provider = self._counting_provider()
        service = CatalogService(session, provider)

        count = await service.reembed_all(batch_size=100)

        assert count >= len(catalogue)
        # One text call for the whole batch, and images chunked by image_batch_size.
        assert provider.text_calls == 1
        assert provider.image_calls == 1

    async def test_completed_batches_survive_a_later_failure(
        self, client: AsyncClient, session: AsyncSession, catalogue: dict[str, dict]
    ) -> None:
        """A rate limit at product 20 must keep the first 19."""
        provider = self._counting_provider()
        provider.fail_after_batches = 1
        service = CatalogService(session, provider)

        embeddings = EmbeddingRepository(session)
        await embeddings.clear_provider(provider.name)
        await session.flush()
        assert await embeddings.count_for(provider.name) == 0

        with pytest.raises(RuntimeError, match="simulated rate limit"):
            await service.reembed_all(batch_size=2)

        # The first batch was flushed before the second one blew up.
        assert await embeddings.count_for(provider.name) == 2

    async def test_skip_existing_resumes_instead_of_restarting(
        self, client: AsyncClient, session: AsyncSession, catalogue: dict[str, dict]
    ) -> None:
        provider = self._counting_provider()
        service = CatalogService(session, provider)

        await service.reembed_all(batch_size=100)
        provider.text_calls = 0
        provider.image_calls = 0

        # Everything is indexed, so a resume run has nothing left to do and spends
        # nothing — the behaviour that makes retrying a rationed provider viable.
        count = await service.reembed_all(batch_size=100, skip_existing=True)
        assert count == 0
        assert provider.text_calls == 0

    async def test_text_only_provider_stores_no_image_vector(
        self, client: AsyncClient, session: AsyncSession, catalogue: dict[str, dict]
    ) -> None:
        from app.embeddings.local_hash import LocalHashProvider

        class TextOnly(LocalHashProvider):
            name = "pretend_text_only"
            supports_images = False

        provider = TextOnly(get_settings().embedding_dim)
        await CatalogService(session, provider).reembed_all(batch_size=100)
        await session.flush()

        result = await session.execute(
            select(ProductEmbedding).where(ProductEmbedding.provider == "pretend_text_only")
        )
        rows = list(result.scalars().all())
        assert rows, "expected the text-only provider to index the catalogue"
        assert all(r.image_embedding is None for r in rows)
        # Still retrievable: fused falls back to the text vector.
        assert all(r.fused_embedding is not None for r in rows)


class TestSearch:
    async def test_text_search_ranks_the_matching_product_first(
        self, client: AsyncClient, catalogue: dict[str, dict]
    ) -> None:
        response = await client.post(
            "/api/v1/search/text",
            json={"query": "waterproof rain jacket for storms", "top_k": 5},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["query_modality"] == "text"
        assert body["hits"], "expected at least one hit"

        names = [hit["product"]["name"] for hit in body["hits"]]
        jacket_names = {catalogue["jacket"]["name"], catalogue["jacket2"]["name"]}
        assert names[0] in jacket_names
        assert (
            names.index(catalogue["mug"]["name"]) > 0 if catalogue["mug"]["name"] in names else True
        )

    async def test_text_search_reports_the_active_provider(
        self, client: AsyncClient, catalogue: dict[str, dict]
    ) -> None:
        body = (
            await client.post("/api/v1/search/text", json={"query": "jacket", "top_k": 3})
        ).json()
        assert body["provider"]
        assert isinstance(body["cross_modal"], bool)
        assert body["took_ms"] >= 0

    async def test_a_product_without_an_image_can_still_rank(
        self, client: AsyncClient, catalogue: dict[str, dict]
    ) -> None:
        body = (
            await client.post(
                "/api/v1/search/text",
                json={"query": "waterproof trail trousers for wet weather", "top_k": 5},
            )
        ).json()
        names = [hit["product"]["name"] for hit in body["hits"]]
        assert catalogue["no_image"]["name"] in names

    async def test_filters_narrow_the_result_set(
        self, client: AsyncClient, catalogue: dict[str, dict]
    ) -> None:
        body = (
            await client.post(
                "/api/v1/search/text",
                json={"query": "waterproof jacket", "top_k": 10, "filters": {"category": "mugs"}},
            )
        ).json()
        assert all(hit["product"]["category"] == "mugs" for hit in body["hits"])

    async def test_empty_query_is_rejected(self, client: AsyncClient) -> None:
        assert (await client.post("/api/v1/search/text", json={"query": ""})).status_code == 422

    async def test_image_search_returns_visually_similar_items(
        self, client: AsyncClient, catalogue: dict[str, dict], product_image: ImageFactory
    ) -> None:
        response = await client.post(
            "/api/v1/search/image",
            files={"image": ("q.jpg", product_image((35, 75, 195)), "image/jpeg")},
            params={"top_k": 5},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["query_modality"] == "image"
        assert body["hits"]
        # The blue jackets should beat the red mug on a blue query image.
        top_names = [hit["product"]["name"] for hit in body["hits"][:2]]
        assert catalogue["mug"]["name"] not in top_names
        assert all(
            hit["image_similarity"] is not None
            for hit in body["hits"]
            if hit["product"]["image_url"]
        )

    async def test_multimodal_accepts_text_and_image_together(
        self, client: AsyncClient, catalogue: dict[str, dict], product_image: ImageFactory
    ) -> None:
        response = await client.post(
            "/api/v1/search/multimodal",
            data={"query": "waterproof rain jacket"},
            files={"image": ("q.jpg", product_image((35, 75, 195)), "image/jpeg")},
            params={"top_k": 5, "text_weight": 0.5},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["query_modality"] == "multimodal"
        assert body["hits"]

    async def test_multimodal_requires_at_least_one_input(self, client: AsyncClient) -> None:
        response = await client.post("/api/v1/search/multimodal", data={})
        assert response.status_code == 422


class TestRecommendations:
    async def test_similar_excludes_the_seed_product(
        self, client: AsyncClient, catalogue: dict[str, dict]
    ) -> None:
        seed = catalogue["jacket"]
        body = (await client.get(f"/api/v1/products/{seed['id']}/similar?top_k=5")).json()
        assert body["strategy"] == "content_similarity"
        assert all(item["product"]["id"] != seed["id"] for item in body["items"])

    async def test_similar_surfaces_the_sibling_jacket(
        self, client: AsyncClient, catalogue: dict[str, dict]
    ) -> None:
        body = (
            await client.get(f"/api/v1/products/{catalogue['jacket']['id']}/similar?top_k=3")
        ).json()
        names = [item["product"]["name"] for item in body["items"]]
        assert catalogue["jacket2"]["name"] in names

    async def test_similar_for_unknown_product_is_404(self, client: AsyncClient) -> None:
        assert (await client.get(f"/api/v1/products/{uuid.uuid4()}/similar")).status_code == 404

    async def test_cold_start_user_falls_back_to_popularity(
        self, client: AsyncClient, catalogue: dict[str, dict]
    ) -> None:
        body = (
            await client.get(f"/api/v1/users/new-user-{uuid.uuid4().hex[:6]}/recommendations")
        ).json()
        assert body["strategy"] == "popularity_fallback"
        assert body["signals_used"] == 0

    async def test_interactions_switch_a_user_to_the_profile_strategy(
        self, client: AsyncClient, catalogue: dict[str, dict]
    ) -> None:
        user = f"tester-{uuid.uuid4().hex[:6]}"
        response = await client.post(
            "/api/v1/interactions",
            json={"user_id": user, "product_id": catalogue["jacket"]["id"], "event": "purchase"},
        )
        assert response.status_code == 201

        body = (await client.get(f"/api/v1/users/{user}/recommendations?top_k=5")).json()
        assert body["strategy"] == "user_profile"
        assert body["signals_used"] == 1
        # Already-purchased items must not be recommended back to the user.
        assert all(item["product"]["id"] != catalogue["jacket"]["id"] for item in body["items"])

    async def test_profile_recommends_the_neighbouring_product(
        self, client: AsyncClient, catalogue: dict[str, dict]
    ) -> None:
        user = f"tester-{uuid.uuid4().hex[:6]}"
        await client.post(
            "/api/v1/interactions",
            json={"user_id": user, "product_id": catalogue["jacket"]["id"], "event": "purchase"},
        )
        body = (await client.get(f"/api/v1/users/{user}/recommendations?top_k=3")).json()
        names = [item["product"]["name"] for item in body["items"]]
        assert catalogue["jacket2"]["name"] in names

    async def test_interaction_for_unknown_product_is_404(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/interactions",
            json={"user_id": "u", "product_id": str(uuid.uuid4()), "event": "view"},
        )
        assert response.status_code == 404

    async def test_invalid_event_type_is_rejected(
        self, client: AsyncClient, catalogue: dict[str, dict]
    ) -> None:
        response = await client.post(
            "/api/v1/interactions",
            json={
                "user_id": "u",
                "product_id": catalogue["jacket"]["id"],
                "event": "teleported",
            },
        )
        assert response.status_code == 422
