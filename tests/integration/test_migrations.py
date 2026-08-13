"""Run the real migration chain against a throwaway database.

This exists because of a failure the rest of the suite structurally could not catch: the
tests build their schema with `Base.metadata.create_all`, so nothing ever executed
`alembic upgrade head` and a broken migration only surfaced in CI.

The specific bug was migration 0002 iterating the live `ALL_PROVIDERS` constant to create
its partial indexes. Adding two providers to that Literal retroactively changed what an
old migration did, so a fresh database created five indexes and 0003 then died with
"relation ix_product_embeddings_fused_gemini already exists". Existing databases were
fine, which is exactly why only a from-scratch run finds it.
"""

import ast
import os
import subprocess
import sys

import psycopg
import pytest

from app.core.config import PROJECT_ROOT, get_settings

pytestmark = pytest.mark.integration

SCRATCH_DB = "recsys_migration_check"

EXPECTED_INDEXES = {
    "ix_product_embeddings_fused_local_hash",
    "ix_product_embeddings_fused_jina",
    "ix_product_embeddings_fused_cohere",
    "ix_product_embeddings_fused_gemini",
    "ix_product_embeddings_fused_voyage",
}


def _admin_dsn() -> str:
    return get_settings().database_url.replace("postgresql+psycopg://", "postgresql://")


def _scratch_url(driver: bool) -> str:
    base = get_settings().database_url if driver else _admin_dsn()
    return base.rsplit("/", 1)[0] + "/" + SCRATCH_DB


def _alembic(command: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "DATABASE_URL": _scratch_url(driver=True)}
    return subprocess.run(
        [sys.executable, "-m", "alembic", command, *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.fixture
def scratch_database() -> str:
    try:
        with psycopg.connect(_admin_dsn(), autocommit=True, connect_timeout=5) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)')
            conn.execute(f'CREATE DATABASE "{SCRATCH_DB}"')
    except psycopg.Error as exc:  # pragma: no cover - environment problem
        pytest.skip(f"Postgres is not reachable: {exc}")

    yield _scratch_url(driver=False)

    with psycopg.connect(_admin_dsn(), autocommit=True, connect_timeout=5) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)')


class TestMigrationChain:
    def test_upgrade_head_succeeds_on_a_fresh_database(self, scratch_database: str) -> None:
        result = _alembic("upgrade", "head")
        assert result.returncode == 0, f"alembic upgrade head failed:\n{result.stderr}"

        with psycopg.connect(scratch_database, connect_timeout=5) as conn:
            rows = conn.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'product_embeddings'"
            ).fetchall()
        names = [r[0] for r in rows]

        assert set(names) >= EXPECTED_INDEXES
        # One index per provider — the duplicate-creation bug showed up as 0003 failing,
        # but a lenient IF NOT EXISTS would have shown up as duplicates instead.
        assert len(names) == len(set(names)), f"duplicate indexes: {names}"

    def test_full_downgrade_then_upgrade_round_trips(self, scratch_database: str) -> None:
        assert _alembic("upgrade", "head").returncode == 0

        down = _alembic("downgrade", "base")
        assert down.returncode == 0, f"downgrade failed:\n{down.stderr}"

        with psycopg.connect(scratch_database, connect_timeout=5) as conn:
            remaining = conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            ).fetchall()
        assert {"products", "product_embeddings", "interactions"}.isdisjoint(
            {r[0] for r in remaining}
        )

        again = _alembic("upgrade", "head")
        assert again.returncode == 0, f"second upgrade failed:\n{again.stderr}"

    def test_migrations_do_not_read_the_live_provider_list(self) -> None:
        """A migration must be a snapshot, not a function of today's configuration.

        Deriving DDL from `ALL_PROVIDERS` means editing an enum silently rewrites history
        for anyone who has not migrated yet.
        """
        offenders = []
        for path in (PROJECT_ROOT / "migrations" / "versions").glob("*.py"):
            tree = ast.parse(path.read_text())
            # Parse rather than grep: these files *discuss* ALL_PROVIDERS in comments
            # explaining why they must not use it, and a substring check flags the prose.
            used = any(
                (isinstance(node, ast.Name) and node.id == "ALL_PROVIDERS")
                or (
                    isinstance(node, ast.ImportFrom)
                    and any(alias.name == "ALL_PROVIDERS" for alias in node.names)
                )
                for node in ast.walk(tree)
            )
            if used:
                offenders.append(path.name)

        assert not offenders, (
            f"{offenders} reference the live provider list; freeze the names in the "
            f"migration instead so adding a provider cannot change an old migration"
        )
