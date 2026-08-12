"""Admin operations: catalogue status, seeding, re-indexing, clearing.

Every mutating action runs as a background job with its own database session, because
the request's session closes as soon as the endpoint returns.
"""

from sqlalchemy import func, select

from app.core.config import ALL_PROVIDERS, get_settings
from app.core.db import get_session_factory
from app.core.errors import ValidationError
from app.core.logging import get_logger
from app.embeddings import available_providers, get_provider, provider_capabilities
from app.models import Interaction
from app.repositories import EmbeddingRepository, ProductRepository
from app.repositories.embedding import ProviderCoverage
from app.schemas.admin import AdminStatus, JobRead, ProviderStatus
from app.seeding import SeedingService
from app.services.catalog import CatalogService
from app.services.jobs import Job, JobRegistry

log = get_logger(__name__)


def validate_providers(names: list[str]) -> list[str]:
    """Resolve a requested provider list, defaulting to the configured provider."""
    settings = get_settings()
    if not names:
        return [settings.embedding_provider]

    usable = set(available_providers())
    unknown = [n for n in names if n not in usable]
    if unknown:
        raise ValidationError(
            f"provider(s) not configured: {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(usable))}"
        )
    # Preserve caller order but drop duplicates.
    return list(dict.fromkeys(names))


class AdminService:
    """Read-side status. Mutations are submitted through `AdminJobs`."""

    def __init__(self, session, registry: JobRegistry) -> None:  # type: ignore[no-untyped-def]
        self.session = session
        self.registry = registry
        self.products = ProductRepository(session)
        self.embeddings = EmbeddingRepository(session)

    async def status(self) -> AdminStatus:
        settings = get_settings()
        coverage = await self.embeddings.coverage()
        configured = set(available_providers())

        interactions = await self.session.execute(select(func.count(Interaction.id)))

        # Every known provider is listed, including ones with no API key, so the panel
        # shows what could be enabled rather than hiding it. `configured` carries the
        # distinction.
        providers: list[ProviderStatus] = []
        for name in sorted(set(ALL_PROVIDERS) | set(coverage) | configured):
            stats: ProviderCoverage = coverage.get(
                name, ProviderCoverage(products=0, with_images=0, model="")
            )
            caps = provider_capabilities(name)
            providers.append(
                ProviderStatus(
                    name=name,
                    configured=name in configured,
                    indexed=name in coverage,
                    supports_images=caps.supports_images,
                    cross_modal=caps.supports_cross_modal,
                    is_default=name == settings.embedding_provider,
                    products=stats["products"],
                    with_images=stats["with_images"],
                    model=stats["model"],
                )
            )

        active = self.registry.active
        return AdminStatus(
            products=await self.products.count(),
            products_with_images=await self.products.count_with_images(),
            interactions=int(interactions.scalar_one()),
            embedding_dim=settings.embedding_dim,
            default_text_weight=settings.default_text_weight,
            default_min_score_ratio=settings.default_min_score_ratio,
            providers=providers,
            active_job=None if active is None else JobRead.model_validate(active),
            recent_jobs=[JobRead.model_validate(j) for j in self.registry.recent()],
        )


class AdminJobs:
    """Builds the background runners for each mutating admin action."""

    def __init__(self, registry: JobRegistry) -> None:
        self.registry = registry

    def seed(self, *, reset: bool, with_images: bool, providers: list[str]) -> Job:
        names = validate_providers(providers)

        async def runner(job: Job) -> str:
            async def progress(done: int, total: int, detail: str) -> None:
                await job.report(done, total, detail)

            async with get_session_factory()() as session:
                result = await SeedingService(session).seed(
                    reset=reset,
                    with_images=with_images,
                    providers=names,
                    progress=progress,
                )
                await session.commit()
            return (
                f"created {result['created']}, skipped {result['skipped']}, "
                f"{result['interactions']} interactions · providers: {', '.join(names)}"
            )

        return self.registry.start("seed", runner)

    def reindex(self, *, providers: list[str], skip_existing: bool = False) -> Job:
        names = validate_providers(providers)

        async def runner(job: Job) -> str:
            counts = []
            for name in names:
                provider = get_provider(name)

                async def progress(done: int, total: int, label: str = name) -> None:
                    await job.report(done, total, f"re-indexing with {label}")

                # Each batch commits on its own, so a rate-limit failure part-way keeps
                # the work already paid for.
                async with get_session_factory()() as session:
                    count = await CatalogService(session, provider).reembed_all(
                        progress=progress,
                        skip_existing=skip_existing,
                        commit_each_batch=True,
                    )
                counts.append(f"{name}: {count}")
            return "re-indexed " + ", ".join(counts)

        return self.registry.start("reindex", runner)

    def clear_catalog(self) -> Job:
        async def runner(job: Job) -> str:
            async with get_session_factory()() as session:
                removed = await SeedingService(session).clear_catalog()
                await session.commit()
            return f"removed {removed} products and their vectors"

        return self.registry.start("clear_catalog", runner)

    def clear_provider(self, provider: str) -> Job:
        names = validate_providers([provider])

        async def runner(job: Job) -> str:
            async with get_session_factory()() as session:
                removed = await SeedingService(session).clear_provider(names[0])
                await session.commit()
            return f"removed {removed} {names[0]} vectors"

        return self.registry.start("clear_provider", runner)
