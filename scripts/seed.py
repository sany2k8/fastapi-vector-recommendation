"""CLI wrapper around the seeding service.

    uv run python -m scripts.seed                          # products + images
    uv run python -m scripts.seed --reset                  # wipe first
    uv run python -m scripts.seed --no-images              # text vectors only
    uv run python -m scripts.seed -p local_hash -p jina    # index with both

The same routine backs the admin UI (POST /api/v1/admin/seed), so the two cannot drift.
Real photos: drop files named after the SKU (e.g. SNK-001.jpg) into data/images/seed/.
"""

import asyncio

import typer
from rich.console import Console

from app.core.db import dispose_engine, get_session_factory
from app.embeddings import close_providers, get_provider
from app.seeding import SeedingService
from app.services.admin import validate_providers

console = Console()
cli = typer.Typer(help="Seed the demo catalogue.")


async def run(*, reset: bool, with_images: bool, providers: list[str]) -> None:
    names = validate_providers(providers)
    for name in names:
        provider = get_provider(name)
        console.print(
            f"[bold]{provider.name}[/bold] dim={provider.dim} "
            f"cross-modal={'yes' if provider.supports_cross_modal else 'no'}"
        )
        if not provider.supports_cross_modal:
            console.print(
                f"  [yellow]note:[/yellow] {provider.name} does not share a space between "
                "text and images — text→image search will be noise there."
            )

    async def progress(done: int, total: int, detail: str) -> None:
        if done == total or done % 10 == 0:
            console.print(f"  [dim]{detail}: {done}/{total}[/dim]")

    async with get_session_factory()() as session:
        result = await SeedingService(session).seed(
            reset=reset, with_images=with_images, providers=names, progress=progress
        )
        await session.commit()

    console.print(
        f"[green]seeded[/green] created={result['created']} skipped={result['skipped']} "
        f"interactions={result['interactions']} providers={', '.join(names)}"
    )
    await close_providers()
    await dispose_engine()


@cli.command()
def main(
    reset: bool = typer.Option(False, "--reset", help="Delete existing products first."),
    images: bool = typer.Option(True, "--images/--no-images", help="Embed product images."),
    provider: list[str] = typer.Option(
        [], "--provider", "-p", help="Index with this provider (repeatable)."
    ),
) -> None:
    asyncio.run(run(reset=reset, with_images=images, providers=provider))


if __name__ == "__main__":
    cli()
