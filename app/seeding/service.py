"""Seeding the demo catalogue, callable from both the CLI and the admin API."""

import random
from collections.abc import Awaitable, Callable
from decimal import Decimal
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import PROJECT_ROOT, get_settings
from app.core.logging import get_logger
from app.embeddings import get_provider
from app.models import Interaction, InteractionType, Product
from app.repositories import EmbeddingRepository, ProductRepository
from app.schemas import ProductCreate
from app.seeding.catalog_data import CATALOG, CatalogEntry
from app.seeding.images import render
from app.services.catalog import CatalogService
from app.services.catalog import ProgressHook as CatalogProgressHook

log = get_logger(__name__)

OVERRIDE_DIR = PROJECT_ROOT / "data" / "images" / "seed"

ProgressHook = Callable[[int, int, str], Awaitable[None]]

#: Two personas so /users/{id}/recommendations has something to work with.
DEMO_USERS: dict[str, list[str]] = {
    "demo-outdoors": ["SNK-003", "JKT-001", "BAG-002", "MUG-003", "JKT-002"],
    "demo-desk": ["CHR-002", "AUD-001", "LMP-003", "BAG-001", "WCH-003"],
}


def load_image(entry: CatalogEntry) -> bytes:
    """Prefer a real photo named after the SKU; otherwise draw a placeholder."""
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = OVERRIDE_DIR / f"{entry['sku']}{suffix}"
        if candidate.is_file():
            return candidate.read_bytes()
    return render(entry)


class SeedingService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.products = ProductRepository(session)

    async def clear_catalog(self) -> int:
        """Remove every product, its vectors (via cascade), interactions and image files."""
        removed = await self.products.count()
        await self.session.execute(delete(Interaction))
        await self.session.execute(delete(Product))
        await self.session.flush()

        for stale in Path(get_settings().image_dir).glob("*"):
            if stale.is_file() and stale.name != ".gitkeep":
                stale.unlink(missing_ok=True)
        log.info("seed.cleared", products=removed)
        return removed

    async def seed(
        self,
        *,
        reset: bool = False,
        with_images: bool = True,
        providers: list[str] | None = None,
        progress: ProgressHook | None = None,
    ) -> dict[str, int]:
        """Create the demo catalogue and index it with each requested provider.

        The first provider does the ingest; the rest are layered on afterwards so their
        vectors sit alongside rather than replacing, which is what makes a side-by-side
        provider comparison possible.
        """
        names = providers or [get_settings().embedding_provider]
        primary = get_provider(names[0])

        if reset:
            await self.clear_catalog()

        service = CatalogService(self.session, primary)
        created = 0
        skipped = 0
        total = len(CATALOG)

        for index, entry in enumerate(CATALOG, start=1):
            existing = await self.products.get_by_sku(entry["sku"])
            if existing is not None:
                skipped += 1
            else:
                payload = ProductCreate(
                    sku=entry["sku"],
                    name=entry["name"],
                    description=entry["description"],
                    category=entry["category"],
                    brand=entry["brand"],
                    price=Decimal(str(entry["price"])),
                    attributes={**entry["attributes"], "color": entry["color"]},
                )
                await service.create(payload, image=load_image(entry) if with_images else None)
                created += 1

            if progress is not None:
                await progress(index, total, f"seeding with {primary.name}")

        await self.session.flush()

        for name in names[1:]:
            extra = CatalogService(self.session, get_provider(name))
            await extra.reembed_all(progress=self._relay(progress, f"indexing with {name}"))

        events = await self._seed_interactions()
        log.info("seed.completed", created=created, skipped=skipped, providers=names)
        return {"created": created, "skipped": skipped, "interactions": events}

    @staticmethod
    def _relay(progress: ProgressHook | None, label: str) -> "CatalogProgressHook | None":
        """Adapt the seeder's (done, total, detail) hook to the catalog service's."""
        if progress is None:
            return None

        async def relay(done: int, total: int) -> None:
            await progress(done, total, label)

        return relay

    async def _seed_interactions(self) -> int:
        """Give the demo personas a history, but only once."""
        existing = await self.session.execute(
            select(func.count(Interaction.id)).where(Interaction.user_id.in_(DEMO_USERS))
        )
        if int(existing.scalar_one()) > 0:
            return 0

        rng = random.Random(42)
        events = 0
        for user_id, skus in DEMO_USERS.items():
            for sku in skus:
                product = await self.products.get_by_sku(sku)
                if product is None:
                    continue
                event = rng.choice(list(InteractionType))
                self.session.add(
                    Interaction(user_id=user_id, product_id=product.id, event=event.value)
                )
                events += 1
        await self.session.flush()
        return events

    async def clear_provider(self, provider: str) -> int:
        removed = await EmbeddingRepository(self.session).clear_provider(provider)
        log.info("seed.provider_cleared", provider=provider, rows=removed)
        return removed
