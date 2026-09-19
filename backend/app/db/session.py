import os
from pathlib import Path
from typing import AsyncGenerator
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.app.core.config import settings
from backend.app.core.logger import logger


def _get_database_url() -> str:
    """
    Kalıcı veritabanı bağlantı URL'sini belirler.
    Öncelik: PostgreSQL (ortam değişkenleri doğrulanmışsa) veya kalıcı SQLite veritabanı.
    """
    env_db_url = os.getenv("DATABASE_URL")
    if env_db_url and "postgresql" in env_db_url:
        return env_db_url

    # Varsayılan kalıcı SQLite veritabanı yolu (fhdss.db)
    db_file = settings.BASE_DIR / "fhdss.db"
    return f"sqlite+aiosqlite:///{db_file.as_posix()}"


DATABASE_URL = _get_database_url()

# SQLAlchemy Async Engine
engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,
)

if "sqlite" in DATABASE_URL:
    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=10000")
            cursor.close()
        except Exception as ex:
            logger.warning(f"SQLite PRAGMA ayarlanırken hata: {ex}")

# Async Session Fabrikası
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI Dependency Injection için asenkron veritabanı oturumu sağlayıcısı."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
