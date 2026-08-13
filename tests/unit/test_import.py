"""Remote-import safety and field mapping."""

import httpx
import pytest

from app.core.config import get_settings
from app.core.errors import ValidationError
from app.core.net import assert_fetchable, fetch
from app.schemas.importing import FieldMapping
from app.seeding.importers import drafts_from_remote, extract_items, make_sku

DUMMYJSON = {
    "products": [
        {
            "id": 1,
            "title": "Essence Mascara",
            "description": "Lash volume",
            "category": "beauty",
            "brand": "Essence",
            "price": 9.99,
            "thumbnail": "https://cdn.example.com/1.png",
            "images": ["https://cdn.example.com/1a.png"],
        },
        {
            "id": 2,
            "title": "Eyeshadow Palette",
            "description": "Twelve shades",
            "category": "beauty",
            "brand": "Glamour",
            "price": 19.99,
            "thumbnail": "https://cdn.example.com/2.png",
        },
    ],
    "total": 2,
}


class TestSsrfGuard:
    """Fetching an admin-supplied URL is SSRF unless every hop is checked."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:5439/",
            "http://127.0.0.1/",
            "http://169.254.169.254/latest/meta-data/",  # cloud metadata
            "http://10.0.0.5/internal",
            "http://192.168.1.1/",
            "http://[::1]/",
        ],
    )
    def test_private_and_metadata_addresses_are_refused(self, url: str) -> None:
        with pytest.raises(ValidationError, match=r"non-public|resolve"):
            assert_fetchable(url)

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "gopher://x/"])
    def test_non_http_schemes_are_refused(self, url: str) -> None:
        with pytest.raises(ValidationError, match="only http and https"):
            assert_fetchable(url)

    def test_public_hosts_are_allowed(self) -> None:
        assert_fetchable("https://dummyjson.com/products")  # resolves publicly

    def test_escape_hatch_is_opt_in(self) -> None:
        settings = get_settings()
        settings.allow_private_import_hosts = True
        try:
            assert_fetchable("http://localhost:5439/")  # permitted only because opted in
        finally:
            settings.allow_private_import_hosts = False

    async def test_redirect_to_a_private_address_is_blocked(self) -> None:
        """Auto-following redirects is the classic bypass: hop 1 is public, hop 2 is not."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "dummyjson.com":
                return httpx.Response(302, headers={"location": "http://127.0.0.1/secrets"})
            return httpx.Response(200, json={"leaked": True})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ValidationError, match=r"non-public|resolve"):
                await fetch("https://dummyjson.com/x", client=client, max_bytes=10_000)

    async def test_oversized_response_is_cut_off(self) -> None:
        body = b"x" * 5000
        transport = httpx.MockTransport(lambda r: httpx.Response(200, content=body))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValidationError, match="import limit"):
                await fetch("https://dummyjson.com/x", client=client, max_bytes=1000)

    async def test_http_errors_surface_clearly(self) -> None:
        transport = httpx.MockTransport(lambda r: httpx.Response(404, text="nope"))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValidationError, match="HTTP 404"):
                await fetch("https://dummyjson.com/x", client=client, max_bytes=10_000)


class TestFieldMapping:
    def test_default_mapping_reads_dummyjson(self) -> None:
        drafts = drafts_from_remote(DUMMYJSON, FieldMapping(), prefix="DJ", limit=10)
        assert [d.name for d in drafts] == ["Essence Mascara", "Eyeshadow Palette"]
        first = drafts[0]
        assert first.sku == "DJ-1"
        assert first.category == "beauty"
        assert first.brand == "Essence"
        assert float(first.price) == 9.99
        assert first.image_url == "https://cdn.example.com/1.png"

    def test_a_list_valued_path_takes_the_first_entry(self) -> None:
        drafts = drafts_from_remote(
            DUMMYJSON, FieldMapping(image_url="images"), prefix="DJ", limit=10
        )
        assert drafts[0].image_url == "https://cdn.example.com/1a.png"

    def test_top_level_array_feeds_need_no_items_path(self) -> None:
        body = [{"id": 7, "title": "Kettle", "price": 25, "image": "https://x.test/k.png"}]
        drafts = drafts_from_remote(
            body, FieldMapping(items_path="", brand="", image_url="image"), prefix="FS", limit=5
        )
        assert drafts[0].sku == "FS-7"
        assert drafts[0].brand is None

    def test_records_without_a_name_are_skipped(self) -> None:
        body = {"products": [{"id": 1}, {"id": 2, "title": "Real"}]}
        drafts = drafts_from_remote(body, FieldMapping(), prefix="P", limit=10)
        assert [d.name for d in drafts] == ["Real"]

    def test_limit_is_respected(self) -> None:
        assert len(drafts_from_remote(DUMMYJSON, FieldMapping(), prefix="DJ", limit=1)) == 1

    def test_a_wrong_items_path_explains_itself(self) -> None:
        with pytest.raises(ValidationError, match="items_path"):
            extract_items(DUMMYJSON, FieldMapping(items_path="nope"))

    def test_nothing_usable_is_an_error_not_a_silent_empty_import(self) -> None:
        with pytest.raises(ValidationError, match="no usable products"):
            drafts_from_remote({"products": [{"id": 1}]}, FieldMapping(), prefix="P", limit=5)

    def test_missing_price_defaults_rather_than_failing(self) -> None:
        body = {"products": [{"id": 1, "title": "X", "price": "not-a-number"}]}
        assert drafts_from_remote(body, FieldMapping(), prefix="P", limit=5)[0].price == 0

    def test_skus_are_sanitised_and_bounded(self) -> None:
        sku = make_sku("SRC", "a/b c?d" * 40, 0)
        assert len(sku) <= 64
        assert "/" not in sku and " " not in sku
