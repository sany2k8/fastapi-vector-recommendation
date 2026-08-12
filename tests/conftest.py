"""Global test guards.

Two things are forced for the whole session, both to stop tests touching anything real:

* **Offline embeddings.** The developer's `.env` may hold live Jina/Cohere keys, and
  nothing here should spend their quota. Hosted providers are covered separately in
  `tests/unit/test_hosted_providers.py` via a mock transport.
* **A dedicated database.** Background admin jobs open their *own* sessions and commit,
  so they escape the per-test transaction rollback. Pointed at the dev database they
  would rewrite the seeded catalogue; pointed at `recsys_test` they cannot.
"""

import re
from collections.abc import Iterator

import psycopg
import pytest

from app.core.config import get_settings

TEST_DB_NAME = "recsys_test"


@pytest.fixture(scope="session", autouse=True)
def offline_embeddings_only() -> Iterator[None]:
    """Blank every provider credential for the whole session.

    Derived from the settings fields rather than a hand-written list: naming providers
    explicitly meant that adding Voyage and Gemini silently re-opened the hole, and a
    test that spends a 3-requests-per-minute quota is worse than one that fails.
    """
    settings = get_settings()
    key_fields = [name for name in type(settings).model_fields if name.endswith("_api_key")]
    assert key_fields, "expected provider API key settings to exist"

    original = {name: getattr(settings, name) for name in key_fields}
    original_provider = settings.embedding_provider

    for name in key_fields:
        setattr(settings, name, "")
    settings.embedding_provider = "local_hash"
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(settings, name, value)
        settings.embedding_provider = original_provider


@pytest.fixture(scope="session", autouse=True)
def test_database() -> Iterator[None]:
    """Point every engine at `recsys_test`, creating it if needed."""
    settings = get_settings()
    original = settings.database_url
    admin_dsn = original.replace("postgresql+psycopg://", "postgresql://")

    try:
        with psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB_NAME,)
            ).fetchone()
            if not exists:
                conn.execute(f'CREATE DATABASE "{TEST_DB_NAME}"')
    except psycopg.Error as exc:  # pragma: no cover - environment problem, not a failure
        pytest.skip(f"Postgres is not reachable: {exc}")

    settings.database_url = re.sub(r"/[^/]+$", f"/{TEST_DB_NAME}", original)
    try:
        yield
    finally:
        settings.database_url = original
