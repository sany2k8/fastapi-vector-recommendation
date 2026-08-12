"""Integration fixtures: a real Postgres, but every test rolled back.

Each test runs inside an outer transaction with the session joined via SAVEPOINT, so
the service layer can commit normally and the whole thing still unwinds afterwards —
the developer's seeded catalogue survives a test run untouched.
"""

import io
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image, ImageDraw
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.core.config import get_settings
from app.core.db import Base, dispose_engine, get_engine, get_session
from app.main import create_app

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session", autouse=True)
def isolated_image_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Keep test uploads out of the real data/images directory."""
    settings = get_settings()
    target = tmp_path_factory.mktemp("images")
    settings.image_dir = target
    return target


#: The schema only needs creating once per run, but the engine is rebuilt per test
#: (below) so no pooled connection is ever reused across event loops.
_schema_ready = False


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    global _schema_ready
    eng = get_engine()
    try:
        if not _schema_ready:
            async with eng.begin() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.run_sync(Base.metadata.create_all)
            _schema_ready = True
        else:
            async with eng.connect() as conn:
                await conn.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - environment problem, not a test failure
        pytest.skip(f"Postgres with pgvector is not reachable: {exc}")
    try:
        yield eng
    finally:
        await dispose_engine()


@pytest_asyncio.fixture
async def session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    connection = await engine.connect()
    transaction = await connection.begin()
    db = AsyncSession(
        bind=connection,
        join_transaction_mode="create_savepoint",
        expire_on_commit=False,
    )
    try:
        yield db
    finally:
        await db.close()
        await transaction.rollback()
        await connection.close()


@pytest_asyncio.fixture
async def client(session: AsyncSession) -> AsyncIterator[AsyncClient]:
    app = create_app()

    async def override_session() -> AsyncIterator[AsyncSession]:
        yield session

    app.dependency_overrides[get_session] = override_session
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        yield http
    app.dependency_overrides.clear()


@pytest.fixture
def product_image() -> "ImageFactory":
    return ImageFactory()


class ImageFactory:
    """Builds small, visually distinct JPEGs for upload tests."""

    def __call__(self, color: tuple[int, int, int] = (200, 40, 50), *, shape: str = "box") -> bytes:
        img = Image.new("RGB", (160, 160), (247, 247, 247))
        draw = ImageDraw.Draw(img)
        if shape == "box":
            draw.rectangle([36, 36, 124, 124], fill=color)
        else:
            draw.ellipse([24, 56, 136, 104], fill=color)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()
