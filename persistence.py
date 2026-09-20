"""Storage bootstrap for chat history: schema plus a stable auth secret."""

import asyncio
import logging
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

logger = logging.getLogger("aiEmployees.persistence")

DATA_DIR = Path(__file__).resolve().parent / "data"
SQLITE_PATH = DATA_DIR / "chainlit.db"
AUTH_SECRET_FILE = DATA_DIR / "auth_secret"
AUTH_SECRET_KEY = "chainlit_auth_secret"
BOOTSTRAP_TIMEOUT_SECONDS = 20

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    "id" TEXT PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" TEXT NOT NULL,
    "createdAt" TEXT
);

CREATE TABLE IF NOT EXISTS threads (
    "id" TEXT PRIMARY KEY,
    "createdAt" TEXT,
    "name" TEXT,
    "userId" TEXT,
    "userIdentifier" TEXT,
    "tags" TEXT,
    "metadata" TEXT,
    FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS steps (
    "id" TEXT PRIMARY KEY,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "threadId" TEXT NOT NULL,
    "parentId" TEXT,
    "streaming" BOOLEAN NOT NULL,
    "waitForAnswer" BOOLEAN,
    "isError" BOOLEAN,
    "metadata" TEXT,
    "tags" TEXT,
    "input" TEXT,
    "output" TEXT,
    "createdAt" TEXT,
    "command" TEXT,
    "start" TEXT,
    "end" TEXT,
    "generation" TEXT,
    "showInput" TEXT,
    "language" TEXT,
    "indent" INT,
    "defaultOpen" BOOLEAN,
    "modes" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS elements (
    "id" TEXT PRIMARY KEY,
    "threadId" TEXT,
    "type" TEXT,
    "url" TEXT,
    "chainlitKey" TEXT,
    "name" TEXT NOT NULL,
    "display" TEXT,
    "objectKey" TEXT,
    "size" TEXT,
    "page" INT,
    "language" TEXT,
    "forId" TEXT,
    "mime" TEXT,
    "props" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS feedbacks (
    "id" TEXT PRIMARY KEY,
    "forId" TEXT NOT NULL,
    "threadId" TEXT NOT NULL,
    "value" INT NOT NULL,
    "comment" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS app_settings (
    "key" TEXT PRIMARY KEY,
    "value" TEXT NOT NULL
);
"""

POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    "id" UUID PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" JSONB NOT NULL,
    "createdAt" TEXT
);

CREATE TABLE IF NOT EXISTS threads (
    "id" UUID PRIMARY KEY,
    "createdAt" TEXT,
    "name" TEXT,
    "userId" UUID,
    "userIdentifier" TEXT,
    "tags" TEXT[],
    "metadata" JSONB,
    FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS steps (
    "id" UUID PRIMARY KEY,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "threadId" UUID NOT NULL,
    "parentId" UUID,
    "streaming" BOOLEAN NOT NULL,
    "waitForAnswer" BOOLEAN,
    "isError" BOOLEAN,
    "metadata" JSONB,
    "tags" TEXT[],
    "input" TEXT,
    "output" TEXT,
    "createdAt" TEXT,
    "command" TEXT,
    "start" TEXT,
    "end" TEXT,
    "generation" JSONB,
    "showInput" TEXT,
    "language" TEXT,
    "indent" INT,
    "defaultOpen" BOOLEAN,
    "modes" JSONB,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS elements (
    "id" UUID PRIMARY KEY,
    "threadId" UUID,
    "type" TEXT,
    "url" TEXT,
    "chainlitKey" TEXT,
    "name" TEXT NOT NULL,
    "display" TEXT,
    "objectKey" TEXT,
    "size" TEXT,
    "page" INT,
    "language" TEXT,
    "forId" UUID,
    "mime" TEXT,
    "props" JSONB,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS feedbacks (
    "id" UUID PRIMARY KEY,
    "forId" UUID NOT NULL,
    "threadId" UUID NOT NULL,
    "value" INT NOT NULL,
    "comment" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS app_settings (
    "key" TEXT PRIMARY KEY,
    "value" TEXT NOT NULL
);
"""


def database_url() -> str:
    """Async SQLAlchemy URL. Falls back to a local sqlite file."""
    raw = os.getenv("DATABASE_URL", "").strip()
    if raw:
        for prefix in ("postgresql+asyncpg://", "postgres://", "postgresql://"):
            if raw.startswith(prefix):
                return "postgresql+asyncpg://" + raw[len(prefix) :]
        return raw
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{SQLITE_PATH}"


def uses_postgres() -> bool:
    return database_url().startswith("postgresql+asyncpg://")


def history_is_durable() -> bool:
    """Sqlite lives on an ephemeral disk, so only Postgres survives a restart."""
    return uses_postgres()


def postgres_ssl_required() -> bool:
    mode = os.getenv("DATABASE_SSL", "auto").strip().lower()
    if mode in {"off", "false", "0", "disable"}:
        return False
    if mode in {"on", "true", "1", "require"}:
        return True
    return uses_postgres()


def connect_args() -> dict:
    return {"ssl": True} if uses_postgres() and postgres_ssl_required() else {}


async def _prepare_storage() -> str:
    engine = create_async_engine(database_url(), connect_args=connect_args())
    schema = POSTGRES_SCHEMA if uses_postgres() else SQLITE_SCHEMA
    try:
        async with engine.begin() as conn:
            for statement in schema.split(";"):
                if statement.strip():
                    await conn.execute(text(statement))
            await conn.execute(
                text(
                    'INSERT INTO app_settings ("key", "value") VALUES (:key, :value) '
                    'ON CONFLICT ("key") DO NOTHING'
                ),
                {"key": AUTH_SECRET_KEY, "value": secrets.token_urlsafe(48)},
            )
            stored = await conn.execute(
                text('SELECT "value" FROM app_settings WHERE "key" = :key'),
                {"key": AUTH_SECRET_KEY},
            )
            return stored.scalar_one()
    finally:
        await engine.dispose()


def _file_secret() -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if AUTH_SECRET_FILE.exists():
        existing = AUTH_SECRET_FILE.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    secret = secrets.token_urlsafe(48)
    AUTH_SECRET_FILE.write_text(secret, encoding="utf-8")
    return secret


def bootstrap() -> None:
    """Create the history tables and pin an auth secret before Chainlit loads."""
    load_dotenv()
    try:
        secret = asyncio.run(
            asyncio.wait_for(_prepare_storage(), timeout=BOOTSTRAP_TIMEOUT_SECONDS)
        )
    except Exception as exc:
        logger.warning("Storage bootstrap failed (%s); chat history may not persist.", exc)
        secret = None

    if not os.getenv("CHAINLIT_AUTH_SECRET"):
        # Reusing one secret keeps people logged in across restarts and deploys.
        os.environ["CHAINLIT_AUTH_SECRET"] = secret or _file_secret()
