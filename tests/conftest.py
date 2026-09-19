import os
os.environ["ENVIRONMENT"] = "test"
import asyncio
import pytest
from backend.app.core.config import settings
settings.ENVIRONMENT = "test"
from backend.app.db.base import Base
from backend.app.db.session import engine
from backend.app.api.v1.endpoints.cases import _ORCHESTRATORS
import backend.app.models.entities  # noqa: F401


@pytest.fixture(autouse=True)
def setup_and_clean_test_database():
    """Her test öncesi tekil veritabanı tablolarını sıfırlayıp temiz bir durum sağlar."""
    async def _reset():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_reset())
    _ORCHESTRATORS.clear()
    yield
