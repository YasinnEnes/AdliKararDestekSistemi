"""
FHDSS Sprint 15 - Adli Güvenlik, RBAC, Dosya ve Arşiv Bütünlük Test Paketi.
"""

import io
import zipfile
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.security import create_access_token
from tests.test_api_endpoints import create_synthetic_png_bytes

client = TestClient(app)


def test_unauthorized_case_deletion_rejected():
    """Standart bilirkişi (EXAMINER) rolünün vaka silme yetkisinin olmadığını (403) doğrular."""
    case_id = "VAKA_RBAC_TEST_01"
    examiner_token = create_access_token("examiner_01", role="EXAMINER")
    examiner_headers = {"Authorization": f"Bearer {examiner_token}"}

    # Vaka oluştur
    create_res = client.post(
        "/api/v1/cases",
        json={"case_id": case_id, "name": "RBAC Test Vakası"},
        headers=examiner_headers,
    )
    assert create_res.status_code == 201

    # EXAMINER silmeye çalışır -> 403 Forbidden
    del_res = client.delete(f"/api/v1/cases/{case_id}", headers=examiner_headers)
    assert del_res.status_code == 403
    assert "yetkiniz yetersiz" in del_res.json()["detail"].lower()


def test_admin_case_soft_delete():
    """ADMIN rolünün vakayı soft-delete olarak silebildiğini ve listenin güncellendiğini doğrular."""
    case_id = "VAKA_RBAC_ADMIN_01"
    admin_token = create_access_token("admin_user", role="ADMIN")
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # Vaka oluştur
    client.post(
        "/api/v1/cases",
        json={"case_id": case_id, "name": "Admin Soft Delete Vakası"},
        headers=admin_headers,
    )

    # ADMIN siler -> 200 OK
    del_res = client.delete(f"/api/v1/cases/{case_id}", headers=admin_headers)
    assert del_res.status_code == 200
    assert "soft-deleted" in del_res.json()["message"]

    # Silinen vaka sorgulanamaz (404)
    get_res = client.get(f"/api/v1/cases/{case_id}", headers=admin_headers)
    assert get_res.status_code == 404


def test_oversized_file_upload_rejected():
    """50 MB limitini aşan dosyaların 413 Payload Too Large ile reddedildiğini doğrular."""
    case_id = "VAKA_FILE_LIMIT_01"
    client.post("/api/v1/cases", json={"case_id": case_id, "name": "Dosya Limit Vakası"})

    # 51 MB sanal bayt
    oversized_data = b"\xFF\xD8\xFF" + b"\x00" * (51 * 1024 * 1024)
    res = client.post(
        f"/api/v1/cases/{case_id}/samples",
        files={"file": ("huge.jpg", io.BytesIO(oversized_data), "image/jpeg")},
        data={"sample_type": "questioned"},
    )
    assert res.status_code == 413
    assert "çok büyük" in res.json()["detail"]


def test_fake_mime_and_malicious_file_rejected():
    """Uzantısı .png olan ancak magic-byte'ı script/exe olan dosyaların 415 ile reddedildiğini doğrular."""
    case_id = "VAKA_MIME_TEST_01"
    client.post("/api/v1/cases", json={"case_id": case_id, "name": "MIME Güvenlik Vakası"})

    fake_png_script = b"<?php echo 'malicious payload'; ?>"
    res = client.post(
        f"/api/v1/cases/{case_id}/samples",
        files={"file": ("exploit.png", io.BytesIO(fake_png_script), "image/png")},
        data={"sample_type": "questioned"},
    )
    assert res.status_code == 415
    assert "Güvenlik Koruması" in res.json()["detail"]


def test_zip_path_traversal_rejected():
    """ZIP içinde path traversal (../) içeren arşivlerin doğrulama servisi tarafından reddedildiğini doğrular."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("../../../etc/shadow", "malicious_content")
        zf.writestr("manifest.json", '{"files": []}')

    zip_bytes = zip_buffer.getvalue()
    res = client.post(
        "/api/v1/archive/verify",
        files={"file": ("tampered.zip", io.BytesIO(zip_bytes), "application/zip")},
    )
    assert res.status_code == 200
    report = res.json()
    assert report["is_valid"] is False
    assert "Path Traversal" in report["error"]


def test_zip_bomb_protection():
    """Açılmış boyutu 200 MB'ı aşan Decompression Bomb arşivlerinin engellendiğini doğrular."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 210 MB boş içerik sıkıştırıldığında birkaç KB olur (klasik zip bomb)
        zf.writestr("huge_decompressed.txt", b"\x00" * (210 * 1024 * 1024))
        zf.writestr("manifest.json", '{"files": []}')

    zip_bytes = zip_buffer.getvalue()
    res = client.post(
        "/api/v1/archive/verify",
        files={"file": ("zipbomb.zip", io.BytesIO(zip_bytes), "application/zip")},
    )
    assert res.status_code == 200
    report = res.json()
    assert report["is_valid"] is False
    assert "Decompression Bomb" in report["error"]


def test_analysis_job_entity_drift_resolved():
    """AnalysisJob modelinde current_step ve step_label senkronizasyonunu doğrular."""
    from backend.app.models.entities import AnalysisJobEntity
    job = AnalysisJobEntity(case_id="VAKA_JOB_TEST_01", current_step="Adli Ön İşleme")
    assert job.step_label == "Adli Ön İşleme"
    assert job.current_step == "Adli Ön İşleme"
