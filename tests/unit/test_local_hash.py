"""The offline provider: determinism, geometry, and whether it discriminates at all."""

import io

import numpy as np
import pytest
from PIL import Image

from app.embeddings.base import weighted_merge
from app.embeddings.local_hash import LocalHashProvider

DIM = 256


@pytest.fixture
def provider() -> LocalHashProvider:
    return LocalHashProvider(dim=DIM)


def make_image(color: tuple[int, int, int], *, shape: str = "block") -> bytes:
    img = Image.new("RGB", (128, 128), (245, 245, 245))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    if shape == "block":
        draw.rectangle([30, 30, 98, 98], fill=color)
    else:
        draw.ellipse([20, 45, 108, 85], fill=color)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class TestTextEmbedding:
    async def test_is_deterministic(self, provider: LocalHashProvider) -> None:
        assert await provider.embed_text("blue running shoe") == await provider.embed_text(
            "blue running shoe"
        )

    async def test_has_configured_width_and_unit_length(self, provider: LocalHashProvider) -> None:
        vec = await provider.embed_text("waterproof hiking jacket")
        assert len(vec) == DIM
        assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-5)

    async def test_related_text_scores_above_unrelated(self, provider: LocalHashProvider) -> None:
        query = await provider.embed_text("waterproof rain jacket")
        close = await provider.embed_text("waterproof rain shell jacket for storms")
        far = await provider.embed_text("stoneware coffee mug with reactive glaze")
        assert np.dot(query, close) > np.dot(query, far)

    async def test_empty_text_yields_a_zero_vector(self, provider: LocalHashProvider) -> None:
        vec = await provider.embed_text("")
        assert len(vec) == DIM
        assert float(np.linalg.norm(vec)) == pytest.approx(0.0)

    async def test_batches_map_one_to_one(self, provider: LocalHashProvider) -> None:
        vectors = await provider.embed_texts(["alpha", "beta", "gamma"])
        assert len(vectors) == 3
        assert vectors[0] != vectors[1]


class TestImageEmbedding:
    async def test_is_deterministic(self, provider: LocalHashProvider) -> None:
        payload = make_image((200, 30, 40))
        assert await provider.embed_image(payload) == await provider.embed_image(payload)

    async def test_same_colour_scores_above_different_colour(
        self, provider: LocalHashProvider
    ) -> None:
        red = await provider.embed_image(make_image((200, 30, 40)))
        red2 = await provider.embed_image(make_image((205, 35, 45)))
        blue = await provider.embed_image(make_image((30, 60, 200)))
        assert np.dot(red, red2) > np.dot(red, blue)

    async def test_background_does_not_saturate_similarity(
        self, provider: LocalHashProvider
    ) -> None:
        """Two different products on the same backdrop must not look identical.

        This is the failure the mean-centred features exist to prevent: raw cell means
        made every catalogue image sit above 0.94 cosine of every other one.
        """
        red_block = await provider.embed_image(make_image((200, 30, 40)))
        blue_ellipse = await provider.embed_image(make_image((30, 60, 200), shape="ellipse"))
        assert np.dot(red_block, blue_ellipse) < 0.9

    async def test_rejects_bytes_that_are_not_an_image(self, provider: LocalHashProvider) -> None:
        with pytest.raises(Exception):  # noqa: B017 - Pillow raises its own type here
            await provider.embed_image(b"definitely not a picture")


class TestQueryFusion:
    async def test_text_only_query_matches_plain_text_embedding(
        self, provider: LocalHashProvider
    ) -> None:
        assert await provider.embed_query(text="red sneaker") == await provider.embed_text(
            "red sneaker", "query"
        )

    async def test_blended_query_sits_between_its_two_inputs(
        self, provider: LocalHashProvider
    ) -> None:
        image = make_image((200, 30, 40))
        text_vec = await provider.embed_text("red sneaker", "query")
        image_vec = await provider.embed_image(image, "query")
        blended = await provider.embed_query(text="red sneaker", image=image, text_weight=0.5)

        assert np.dot(blended, text_vec) > 0.5
        assert np.dot(blended, image_vec) > 0.5

    async def test_weight_shifts_the_query_toward_the_heavier_modality(
        self, provider: LocalHashProvider
    ) -> None:
        image = make_image((200, 30, 40))
        text_vec = await provider.embed_text("red sneaker", "query")

        mostly_text = await provider.embed_query(text="red sneaker", image=image, text_weight=0.9)
        mostly_image = await provider.embed_query(text="red sneaker", image=image, text_weight=0.1)
        assert np.dot(mostly_text, text_vec) > np.dot(mostly_image, text_vec)

    async def test_requires_at_least_one_input(self, provider: LocalHashProvider) -> None:
        with pytest.raises(ValueError, match="requires text, image, or both"):
            await provider.embed_query()

    def test_weighted_merge_returns_a_unit_vector(self) -> None:
        merged = weighted_merge([[1.0, 0.0], [0.0, 1.0]], [3.0, 1.0])
        assert float(np.linalg.norm(merged)) == pytest.approx(1.0, abs=1e-6)
        assert merged[0] > merged[1]

    def test_provider_declares_itself_not_cross_modal(self, provider: LocalHashProvider) -> None:
        # Callers key off this flag to avoid promising text->image matching it cannot do.
        assert provider.supports_cross_modal is False
