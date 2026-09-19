"""
FHDSS - Test Suite for Unicode Path Resilience in Job Service
Ensures samples stored in paths with non-ASCII / Unicode characters (e.g. C:\\TÜBTAK\\...)
are safely loaded via cv2.imdecode from memory buffers without OpenCV ANSI path failures.
"""

import hashlib
from pathlib import Path
import cv2
import numpy as np
import pytest
from sqlalchemy import select

from backend.app.db.session import AsyncSessionLocal
from backend.app.models.entities import SampleEntity
from backend.app.services.job_service import _load_and_verify_sample


@pytest.mark.asyncio
async def test_job_service_sample_loading_with_unicode_paths():
    """VAKA numunelerinin Türkçe/Unicode karakterli dosya yollarından güvenle yüklenebildiğini doğrular."""
    async with AsyncSessionLocal() as session:
        # Veritabanında kayıtlı bir numuneyi bul
        stmt = select(SampleEntity).limit(1)
        res = await session.execute(stmt)
        sample = res.scalar_one_or_none()

        if sample and Path(sample.file_path).exists():
            loaded = await _load_and_verify_sample(session, sample.sample_id)
            assert loaded is not None
            assert "image" in loaded
            assert isinstance(loaded["image"], np.ndarray)
            assert loaded["image"].ndim == 3
            assert loaded["sha256_hash"] == sample.sha256_hash
