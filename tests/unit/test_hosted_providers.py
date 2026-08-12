"""Hosted provider clients, exercised against a mock transport.

These assert the exact request shapes the Jina and Cohere APIs expect, so a typo in a
field name fails here rather than silently at runtime with someone's API quota.
"""

import base64
import io
import json
from typing import Any

import httpx
import pytest
from PIL import Image

from app.core.errors import EmbeddingError, UnsupportedOperationError
from app.embeddings.cohere import CohereProvider, to_data_url
from app.embeddings.gemini import GeminiProvider
from app.embeddings.jina import JinaProvider
from app.embeddings.voyage import VoyageProvider

DIM = 4


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 120, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def capture(handler: Any) -> tuple[httpx.MockTransport, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return handler(request)

    return httpx.MockTransport(wrapped), seen


class TestJina:
    @staticmethod
    def provider(transport: httpx.MockTransport) -> JinaProvider:
        p = JinaProvider(dim=DIM, api_key="k", model="jina-embeddings-v4", timeout=5)
        p._client = httpx.AsyncClient(transport=transport)
        return p

    async def test_text_request_shape(self) -> None:
        transport, seen = capture(
            lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0, 0, 0]}]})
        )
        await self.provider(transport).embed_text("a red shoe", "query")

        body = seen[0]
        assert body["model"] == "jina-embeddings-v4"
        assert body["task"] == "retrieval.query"
        assert body["dimensions"] == DIM
        assert body["normalized"] is True
        assert body["input"] == [{"text": "a red shoe"}]

    async def test_documents_use_the_passage_task(self) -> None:
        transport, seen = capture(
            lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0, 0, 0]}]})
        )
        await self.provider(transport).embed_text("a red shoe", "document")
        assert seen[0]["task"] == "retrieval.passage"

    async def test_images_are_sent_base64_under_the_image_key(self) -> None:
        payload = png_bytes()
        transport, seen = capture(
            lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [0, 1, 0, 0]}]})
        )
        await self.provider(transport).embed_image(payload)

        item = seen[0]["input"][0]
        assert set(item) == {"image"}
        assert base64.b64decode(item["image"]) == payload

    async def test_out_of_order_results_are_reindexed(self) -> None:
        transport, _ = capture(
            lambda r: httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [0, 1, 0, 0]},
                        {"index": 0, "embedding": [1, 0, 0, 0]},
                    ]
                },
            )
        )
        vectors = await self.provider(transport).embed_texts(["first", "second"])
        assert vectors == [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]

    async def test_dimension_mismatch_is_caught(self) -> None:
        transport, _ = capture(
            lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0]}]})
        )
        with pytest.raises(ValueError, match="EMBEDDING_DIM"):
            await self.provider(transport).embed_text("x")

    async def test_client_error_is_not_retried(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(401, text="bad key")

        transport, _ = capture(handler)
        with pytest.raises(EmbeddingError, match="401"):
            await self.provider(transport).embed_text("x")
        assert calls["n"] == 1

    async def test_rate_limit_is_retried_then_succeeds(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, text="slow down")
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0, 0, 0]}]})

        transport, _ = capture(handler)
        assert await self.provider(transport).embed_text("x") == [1.0, 0.0, 0.0, 0.0]
        assert calls["n"] == 2


class TestGemini:
    @staticmethod
    def provider(transport: httpx.MockTransport, dim: int = DIM) -> GeminiProvider:
        p = GeminiProvider(dim=dim, api_key="k", model="gemini-embedding-001", timeout=5)
        p._client = httpx.AsyncClient(transport=transport)
        return p

    async def test_text_request_shape(self) -> None:
        transport, seen = capture(
            lambda r: httpx.Response(200, json={"embeddings": [{"values": [1, 0, 0, 0]}]})
        )
        await self.provider(transport).embed_text("a red shoe", "query")

        request = seen[0]["requests"][0]
        assert request["model"] == "models/gemini-embedding-001"
        assert request["taskType"] == "RETRIEVAL_QUERY"
        assert request["outputDimensionality"] == DIM
        assert request["content"] == {"parts": [{"text": "a red shoe"}]}

    async def test_documents_use_the_document_task(self) -> None:
        transport, seen = capture(
            lambda r: httpx.Response(200, json={"embeddings": [{"values": [1, 0, 0, 0]}]})
        )
        await self.provider(transport).embed_text("x", "document")
        assert seen[0]["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"

    async def test_truncated_output_is_renormalised(self) -> None:
        """Google only returns unit-length vectors at 3072 dims; below that we must fix it."""
        transport, _ = capture(
            lambda r: httpx.Response(200, json={"embeddings": [{"values": [3, 4, 0, 0]}]})
        )
        vector = await self.provider(transport).embed_text("x")
        assert vector == pytest.approx([0.6, 0.8, 0.0, 0.0])

    async def test_is_text_only_and_refuses_images(self) -> None:
        transport = httpx.MockTransport(lambda r: httpx.Response(200, json={}))
        provider = self.provider(transport)
        assert provider.supports_images is False
        assert provider.supports_cross_modal is False

        with pytest.raises(UnsupportedOperationError, match="cannot embed images"):
            await provider.embed_image(png_bytes())

    async def test_image_query_is_refused_before_any_request(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json={"embeddings": [{"values": [1, 0, 0, 0]}]})

        transport, _ = capture(handler)
        with pytest.raises(UnsupportedOperationError, match="text-only"):
            await self.provider(transport).embed_query(text="a shoe", image=png_bytes())
        assert calls["n"] == 0

    async def test_wrong_result_count_raises(self) -> None:
        transport, _ = capture(lambda r: httpx.Response(200, json={"embeddings": []}))
        with pytest.raises(EmbeddingError, match="embeddings"):
            await self.provider(transport).embed_texts(["a", "b"])


class TestVoyage:
    @staticmethod
    def provider(
        transport: httpx.MockTransport, model: str = "voyage-multimodal-3.5"
    ) -> VoyageProvider:
        p = VoyageProvider(dim=DIM, api_key="k", model=model, timeout=5)
        p._client = httpx.AsyncClient(transport=transport)
        return p

    async def test_text_request_shape(self) -> None:
        transport, seen = capture(
            lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0, 0, 0]}]})
        )
        await self.provider(transport).embed_text("a red shoe", "query")

        body = seen[0]
        assert body["model"] == "voyage-multimodal-3.5"
        assert body["input_type"] == "query"
        assert body["output_dimension"] == DIM
        assert body["inputs"] == [{"content": [{"type": "text", "text": "a red shoe"}]}]

    async def test_images_are_sent_as_base64_data_urls(self) -> None:
        transport, seen = capture(
            lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [0, 1, 0, 0]}]})
        )
        await self.provider(transport).embed_image(png_bytes())

        part = seen[0]["inputs"][0]["content"][0]
        assert part["type"] == "image_base64"
        assert part["image_base64"].startswith("data:image/png;base64,")

    async def test_fixed_dimension_model_omits_output_dimension(self) -> None:
        """voyage-multimodal-3 has no truncation, so sending the field would be a lie."""
        transport, seen = capture(
            lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0, 0, 0]}]})
        )
        await self.provider(transport, model="voyage-multimodal-3").embed_text("x")
        assert "output_dimension" not in seen[0]

    async def test_a_dimension_mismatch_fails_loudly(self) -> None:
        """Better a hard error than an index quietly built at the wrong width."""
        transport, _ = capture(
            lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0]}]})
        )
        with pytest.raises(ValueError, match="EMBEDDING_DIM"):
            await self.provider(transport).embed_text("x")

    async def test_out_of_order_results_are_reindexed(self) -> None:
        transport, _ = capture(
            lambda r: httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [0, 1, 0, 0]},
                        {"index": 0, "embedding": [1, 0, 0, 0]},
                    ]
                },
            )
        )
        vectors = await self.provider(transport).embed_texts(["first", "second"])
        assert vectors == [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]

    async def test_declares_cross_modal_support(self) -> None:
        transport = httpx.MockTransport(lambda r: httpx.Response(200, json={}))
        provider = self.provider(transport)
        assert provider.supports_images is True
        assert provider.supports_cross_modal is True

    async def test_images_are_chunked_by_image_batch_size(self) -> None:
        """The knob that decides whether a free-tier re-index completes.

        Measured against Voyage: 4 images per request succeed, 8 return 429. Text and
        images must therefore chunk independently.
        """
        requests: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            count = len(body["inputs"])
            requests.append(count)
            return httpx.Response(
                200,
                json={"data": [{"index": i, "embedding": [1, 0, 0, 0]} for i in range(count)]},
            )

        provider = VoyageProvider(
            dim=DIM, api_key="k", model="voyage-multimodal-3.5", timeout=5, image_batch_size=4
        )
        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        await provider.embed_images([png_bytes()] * 10)
        assert requests == [4, 4, 2]

        # Text is not subject to the image cap — batching it hard is what keeps the
        # request count down.
        requests.clear()
        await provider.embed_texts(["a"] * 10)
        assert requests == [10]


class TestCohere:
    @staticmethod
    def provider(transport: httpx.MockTransport) -> CohereProvider:
        p = CohereProvider(dim=DIM, api_key="k", model="embed-v4.0", timeout=5)
        p._client = httpx.AsyncClient(transport=transport)
        return p

    async def test_text_request_shape(self) -> None:
        transport, seen = capture(
            lambda r: httpx.Response(200, json={"embeddings": {"float": [[1, 0, 0, 0]]}})
        )
        await self.provider(transport).embed_text("a red shoe", "query")

        body = seen[0]
        assert body["model"] == "embed-v4.0"
        assert body["input_type"] == "search_query"
        assert body["output_dimension"] == DIM
        assert body["embedding_types"] == ["float"]
        assert body["inputs"] == [{"content": [{"type": "text", "text": "a red shoe"}]}]

    async def test_images_are_sent_as_data_urls(self) -> None:
        transport, seen = capture(
            lambda r: httpx.Response(200, json={"embeddings": {"float": [[0, 1, 0, 0]]}})
        )
        await self.provider(transport).embed_image(png_bytes())

        part = seen[0]["inputs"][0]["content"][0]
        assert part["type"] == "image_url"
        assert part["image_url"]["url"].startswith("data:image/png;base64,")

    async def test_sdk_style_float_underscore_key_is_accepted(self) -> None:
        transport, _ = capture(
            lambda r: httpx.Response(200, json={"embeddings": {"float_": [[0, 0, 1, 0]]}})
        )
        assert await self.provider(transport).embed_text("x") == [0.0, 0.0, 1.0, 0.0]

    async def test_missing_embeddings_raise(self) -> None:
        transport, _ = capture(lambda r: httpx.Response(200, json={"embeddings": {}}))
        with pytest.raises(EmbeddingError, match="embeddings"):
            await self.provider(transport).embed_text("x")

    def test_data_url_sniffs_the_mime_type(self) -> None:
        assert to_data_url(png_bytes()).startswith("data:image/png;base64,")

    def test_data_url_rejects_non_images(self) -> None:
        with pytest.raises(EmbeddingError, match="could not decode"):
            to_data_url(b"not an image")

    def test_provider_declares_cross_modal_support(self) -> None:
        transport = httpx.MockTransport(lambda r: httpx.Response(200, json={}))
        assert self.provider(transport).supports_cross_modal is True
