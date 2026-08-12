"""Operational CLI: `uv run recsys <command>`."""

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from app.core.config import get_settings
from app.core.db import dispose_engine, get_session_factory
from app.embeddings import available_providers, close_providers, get_provider
from app.repositories import EmbeddingRepository, ProductRepository
from app.services.admin import validate_providers
from app.services.catalog import CatalogService

app = typer.Typer(help="Multimodal product recommender utilities.")
console = Console()


@app.command()
def info() -> None:
    """Show the active configuration and per-provider index coverage."""

    async def _run() -> None:
        settings = get_settings()
        table = Table(title="vector-recsys", show_header=False)
        table.add_row("default provider", settings.embedding_provider)
        table.add_row("configured providers", ", ".join(available_providers()))
        table.add_row("embedding dim", str(settings.embedding_dim))
        table.add_row("database", settings.database_url.split("@")[-1])
        table.add_row("api port", str(settings.api_port))

        try:
            async with get_session_factory()() as session:
                products = ProductRepository(session)
                table.add_row("products", str(await products.count()))
                table.add_row("with image files", str(await products.count_with_images()))
                for name, stats in (await EmbeddingRepository(session).coverage()).items():
                    table.add_row(
                        f"indexed: {name}",
                        f"{stats['products']} products, {stats['with_images']} with images",
                    )
        except Exception as exc:
            table.add_row("products", f"[red]unavailable: {exc}[/red]")
        finally:
            await close_providers()
            await dispose_engine()

        console.print(table)

    asyncio.run(_run())


@app.command()
def reindex(
    provider: list[str] = typer.Option(
        [], "--provider", "-p", help="Provider to rebuild (repeatable). Defaults to the default."
    ),
    batch_size: int = typer.Option(50, help="Products embedded per flush."),
) -> None:
    """Re-embed the catalogue for the named providers.

    Other providers' vectors are left alone, so this is also how you add a second index
    for side-by-side comparison rather than replacing what you already had.
    """

    async def _run() -> None:
        names = validate_providers(provider)
        try:
            for name in names:
                instance = get_provider(name)
                console.print(f"re-embedding with [bold]{name}[/bold] ({instance.dim}d)…")
                async with get_session_factory()() as session:
                    count = await CatalogService(session, instance).reembed_all(
                        batch_size=batch_size
                    )
                    await session.commit()
                console.print(f"  [green]done[/green] {count} products")
        finally:
            await close_providers()
            await dispose_engine()

    asyncio.run(_run())


if __name__ == "__main__":
    app()
