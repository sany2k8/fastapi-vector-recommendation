"""Catalogue management: ingest products and keep each provider's vectors in sync."""

import uuid
from collections.abc import Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.errors import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.embeddings import (
    EmbeddingProvider,
    Vector,
    get_provider,
    provider_model_name,
    weighted_merge,
)
from app.models import Product
from app.repositories import CatalogFilters, EmbeddingRepository, ProductRepository
from app.schemas import ProductCreate
from app.schemas.product import build_embedding_text
from app.services.images import remove_image, store_image

log = get_logger(__name__)

ProgressHook = Callable[[int, int], Awaitable[None]]


class CatalogService:
    """Ingest and re-index the catalogue for one embedding provider at a time."""

    def __init__(self, session: AsyncSession, provider: EmbeddingProvider) -> None:
        self.session = session
        self.provider = provider
        self.products = ProductRepository(session)
        self.embeddings = EmbeddingRepository(session)

    async def create(
        self, payload: ProductCreate, *, image: bytes | None = None, embed: bool = True
    ) -> Product:
        """Create a product, embedding it immediately unless `embed` is False.

        Bulk importers pass `embed=False` and then run `reembed_all`, which batches a
        whole page of products into one request instead of two per product — the
        difference between a handful of API calls and hundreds.
        """
        if await self.products.get_by_sku(payload.sku) is not None:
            raise ValidationError(f"a product with sku {payload.sku!r} already exists")

        product = Product(
            id=uuid.uuid4(),
            sku=payload.sku,
            name=payload.name,
            description=payload.description,
            category=payload.category,
            brand=payload.brand,
            price=payload.price,
            currency=payload.currency,
            attributes=payload.attributes,
        )
        stored: bytes | None = None
        if image is not None:
            product.image_path, stored = store_image(image, product.id)
        await self.products.add(product)
        if embed:
            await self._embed_product(product, image=stored)

        log.info(
            "catalog.product_created",
            sku=product.sku,
            provider=self.provider.name if embed else "deferred",
            with_image=image is not None,
        )
        return product

    async def get_or_404(self, product_id: uuid.UUID) -> Product:
        product = await self.products.get(product_id)
        if product is None:
            raise NotFoundError(f"product {product_id} not found")
        return product

    async def list_products(
        self, *, limit: int, offset: int, filters: CatalogFilters
    ) -> tuple[list[Product], int]:
        items = await self.products.list_products(limit=limit, offset=offset, filters=filters)
        total = await self.products.count(filters)
        return items, total

    async def delete(self, product_id: uuid.UUID) -> None:
        product = await self.get_or_404(product_id)
        remove_image(product.image_path)
        await self.products.delete(product)
        # Flush here rather than leaving it to the session teardown, so the DELETE (and
        # any FK complaint it raises) happens inside this call rather than silently later.
        await self.session.flush()
        log.info("catalog.product_deleted", product_id=str(product_id))

    async def attach_image(self, product_id: uuid.UUID, image: bytes) -> Product:
        """Add or replace a product photo and re-embed it for every indexed provider.

        Re-embedding only the current provider would leave the others holding a vector
        for a photo that no longer exists, which then quietly skews their results.
        """
        product = await self.get_or_404(product_id)
        remove_image(product.image_path)
        product.image_path, stored = store_image(image, product.id)
        await self.session.flush()

        for name in await self.embeddings.indexed_providers():
            try:
                provider = get_provider(name)
            except ValidationError:
                # Indexed earlier but no longer configured (key removed) — leave it be.
                log.warning("catalog.provider_unavailable", provider=name)
                continue
            await self._embed_product(product, image=stored, provider=provider)

        log.info("catalog.image_attached", product_id=str(product_id))
        return product

    async def reembed_all(
        self,
        *,
        batch_size: int | None = None,
        progress: ProgressHook | None = None,
        skip_existing: bool = False,
        commit_each_batch: bool = False,
    ) -> int:
        """(Re)compute this provider's vectors for every product.

        Other providers' rows are untouched, which is what allows several indexes to
        coexist and be compared on the same query.

        Three things make this survivable on a rate-limited free tier:

        * **Batching.** One request per batch of products, not two per product. On a
          3-requests-per-minute key that is the difference between ~2 minutes and ~20.
        * **`commit_each_batch`.** Long jobs persist progress as they go, so a rate-limit
          failure at product 20 keeps the first 19 instead of rolling the lot back.
        * **`skip_existing`.** Re-running after a failure resumes rather than starting
          over, which is what you want when every request is rationed.
        """
        size = batch_size or get_settings().reindex_batch_size
        already = (
            set(await self.embeddings.product_ids_for(self.provider.name))
            if skip_existing
            else set()
        )

        pending = [p for p in await self.products.list_all() if p.id not in already]
        total = len(pending)
        processed = 0

        for start in range(0, total, size):
            batch = pending[start : start + size]
            try:
                await self._embed_batch(batch)
            except Exception:
                # Keep whatever earlier batches achieved before surfacing the failure.
                if commit_each_batch:
                    await self.session.commit()
                log.warning(
                    "catalog.reembed_interrupted",
                    provider=self.provider.name,
                    completed=processed,
                    total=total,
                )
                raise

            if commit_each_batch:
                await self.session.commit()
            else:
                await self.session.flush()

            processed += len(batch)
            if progress is not None:
                await progress(processed, total)

        log.info(
            "catalog.reembedded",
            count=processed,
            skipped=len(already),
            provider=self.provider.name,
        )
        return processed

    async def _embed_batch(self, products: list[Product]) -> None:
        """Embed a whole batch in as few requests as the provider allows."""
        if not products:
            return

        texts = [
            build_embedding_text(
                name=p.name,
                category=p.category,
                brand=p.brand,
                description=p.description,
                attributes=dict(p.attributes or {}),
            )
            for p in products
        ]
        text_vectors = await self.provider.embed_texts(texts, "document")

        # Only products that actually have a readable file go to the image endpoint;
        # results are mapped back by position.
        image_vectors: dict[uuid.UUID, Vector] = {}
        if self.provider.supports_images:
            with_images = [(p, data) for p in products if (data := self._read_image(p)) is not None]
            if with_images:
                vectors = await self.provider.embed_images(
                    [data for _, data in with_images], "document"
                )
                image_vectors = {
                    p.id: vec for (p, _), vec in zip(with_images, vectors, strict=True)
                }

        for product, text_vector in zip(products, text_vectors, strict=True):
            image_vector = image_vectors.get(product.id)
            await self.embeddings.upsert(
                product_id=product.id,
                provider=self.provider.name,
                model=provider_model_name(self.provider.name),
                dim=self.provider.dim,
                text_embedding=text_vector,
                image_embedding=image_vector,
                fused_embedding=self._fuse(text_vector, image_vector),
            )

    # --- internals ----------------------------------------------------------
    def _read_image(self, product: Product) -> bytes | None:
        if not product.image_path:
            return None
        path = get_settings().image_dir / product.image_path
        if not path.is_file():
            log.warning("catalog.image_missing", sku=product.sku, path=str(path))
            return None
        return path.read_bytes()

    async def _embed_product(
        self,
        product: Product,
        *,
        image: bytes | None,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        active = provider or self.provider
        text_vector = await active.embed_text(
            build_embedding_text(
                name=product.name,
                category=product.category,
                brand=product.brand,
                description=product.description,
                attributes=dict(product.attributes or {}),
            ),
            "document",
        )
        # A text-only provider (Gemini embedding-001) simply stores no image vector.
        # Products stay retrievable because `_fuse` falls back to the text vector.
        image_vector: Vector | None = None
        if image is not None and active.supports_images:
            image_vector = await active.embed_image(image, "document")

        await self.embeddings.upsert(
            product_id=product.id,
            provider=active.name,
            model=provider_model_name(active.name),
            dim=active.dim,
            text_embedding=text_vector,
            image_embedding=image_vector,
            fused_embedding=self._fuse(text_vector, image_vector),
        )

    @staticmethod
    def _fuse(text_vector: Vector | None, image_vector: Vector | None) -> Vector | None:
        """The vector the ANN index is built on.

        Products without a photo fall back to their text vector so they stay
        retrievable; the alternative (a NULL fused vector) would hide them entirely.
        """
        weight = get_settings().default_text_weight
        if text_vector is not None and image_vector is not None:
            return weighted_merge([text_vector, image_vector], [weight, 1.0 - weight])
        return text_vector if text_vector is not None else image_vector
