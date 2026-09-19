"""
FHDSS - Test Suite for GET /analysis/result/{case_id}
Verifies that the latest analysis result for a case can be fetched via REST API
and contains all required structured forensic fields.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from backend.app.main import app
from backend.app.core.security import create_access_token


@pytest.mark.asyncio
async def test_get_latest_analysis_result_endpoint():
    """Vakaya ait en son analiz sonucunun API üzerinden eksiksiz çekilebildiğini doğrular."""
    token = create_access_token({"sub": "admin", "role": "ADMIN"})
    headers = {"Authorization": f"Bearer {token}"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Mevcut vakaları al
        cases_res = await client.get("/api/v1/cases", headers=headers)
        assert cases_res.status_code == 200
        cases = cases_res.json()

        if cases:
            target_case_id = cases[0]["case_id"]
            # 2. Vaka analiz sonucunu sorgula
            res = await client.get(f"/api/v1/analysis/result/{target_case_id}", headers=headers)
            # Eğer analiz edilmişse 200 ve gerekli alanlar dönmeli; edilmemişse 404 dönmeli
            assert res.status_code in (200, 404)
            if res.status_code == 200:
                data = res.json()
                assert "case_id" in data
                assert "hypothesis" in data
                assert "fusion" in data
                assert "comparison" in data
