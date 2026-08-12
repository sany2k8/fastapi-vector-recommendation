"""FastAPI application factory and app-edge error handling."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1 import api_router
from app.api.v1.health import router as health_router
from app.core.config import PROJECT_ROOT, get_settings
from app.core.db import dispose_engine
from app.core.errors import DomainError
from app.core.logging import configure_logging, get_logger
from app.embeddings import available_providers, close_providers, get_provider

log = get_logger(__name__)

STATIC_DIR = PROJECT_ROOT / "static"

DESCRIPTION = """
A product recommendation API where **text and images share one vector space**.

Each product carries three vectors: one for its text, one for its photo, and a fused
one that the pgvector HNSW index rides on. Queries take the same shape, so you can
search with a sentence, with a photo, or with both blended together.

Set `EMBEDDING_PROVIDER` to `jina` or `cohere` for true cross-modal retrieval; the
default `local_hash` provider runs offline with no API key and no model download,
but only supports text-to-text and image-to-image matching.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_settings()
    provider = get_provider()
    log.info(
        "app.startup",
        default_provider=provider.name,
        configured=available_providers(),
        dim=provider.dim,
        cross_modal=provider.supports_cross_modal,
        port=settings.api_port,
    )
    if not provider.supports_cross_modal:
        log.warning(
            "app.provider_not_cross_modal",
            detail=(
                "local_hash embeds text and images into unrelated spaces; text-to-image "
                "search will return noise. Set EMBEDDING_PROVIDER=jina or =cohere for "
                "real multimodal retrieval."
            ),
        )
    try:
        yield
    finally:
        await close_providers()
        await dispose_engine()
        log.info("app.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Multimodal Product Recommender",
        description=DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(DomainError)
    async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
        log.warning("api.domain_error", code=exc.code, path=request.url.path, detail=exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    app.include_router(health_router)
    app.include_router(api_router, prefix=settings.api_prefix)

    settings.image_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/images", StaticFiles(directory=settings.image_dir), name="images")
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(Path(STATIC_DIR) / "index.html")

    return app


app = create_app()
