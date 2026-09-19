import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.security import create_access_token

client = TestClient(app)
admin_headers = {"Authorization": f"Bearer {create_access_token('admin_test', role='ADMIN')}"}


def create_synthetic_png_bytes(text: str = "Adli Test Belgesi") -> bytes:
    """Bellekte sentetik PNG görseli üretip bayt dizisi döner."""
    img = np.full((300, 300), 255, dtype=np.uint8)
    cv2.putText(img, text, (30, 80), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.9, 0, 2, cv2.LINE_AA)
    cv2.circle(img, (150, 180), 35, 0, 2, cv2.LINE_AA)
    cv2.line(img, (30, 240), (270, 240), 0, 2, cv2.LINE_AA)

    success, encoded = cv2.imencode(".png", img)
    assert success
    return encoded.tobytes()


# ==============================================================================
# 1. Sistem Sağlık ve Vaka Yönetim Testleri
# ==============================================================================

def test_health_check_endpoint():
    """GET /health uç noktasının 200 OK ve sağlıklı durum döndüğünü test eder."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_create_case_and_query():
    """POST /api/v1/cases, GET /api/v1/cases/{case_id}, GET /api/v1/cases ve PUT /api/v1/cases/{case_id} uç noktalarını test eder."""
    case_id = "VAKA_API_2026_01"
    payload = {
        "case_id": case_id,
        "name": "Şüpheli İmza Aidiyet İncelemesi",
        "description": "Senet metni ve keşideci imzası inceleme talebi",
        "examination_type": "SIGNATURE",
        "authority": "Ankara 2. Asliye Ceza Mahkemesi",
        "examiner_id": "UZMAN_AHMET",
        "notes": "İmza aidiyeti tespiti talebi",
    }

    # 1. Vaka Oluştur
    response = client.post("/api/v1/cases", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["case_id"] == case_id
    assert data["name"] == "Şüpheli İmza Aidiyet İncelemesi"
    assert data["state"] == "PHASE_A_INIT"

    # 2. Vaka Durumu Sorgula
    get_res = client.get(f"/api/v1/cases/{case_id}")
    assert get_res.status_code == 200
    assert get_res.json()["case_id"] == case_id

    # 3. Tüm Vakaları Listele
    list_res = client.get("/api/v1/cases")
    assert list_res.status_code == 200
    cases_list = list_res.json()
    assert isinstance(cases_list, list)
    assert any(c["case_id"] == case_id for c in cases_list)

    # 4. Vaka Güncelle
    update_res = client.put(f"/api/v1/cases/{case_id}", json={"current_stage": "FEATURE_EXTRACTION", "status": "IN_PROGRESS"})
    assert update_res.status_code == 200
    assert update_res.json()["current_stage"] == "FEATURE_EXTRACTION"


def test_sample_upload_and_phase_a_transition():
    """Belge yükleme ve Bilişsel İzolasyon faz geçişini test eder."""
    case_id = "VAKA_API_2026_02"
    client.post("/api/v1/cases", json={"case_id": case_id})

    # 1. Sorgulanan (Questioned) Belge Yükle
    q_bytes = create_synthetic_png_bytes("Sorgulanan Imza Asli")
    files = {"file": ("questioned_01.png", io.BytesIO(q_bytes), "image/png")}
    data = {"sample_type": "questioned", "sample_id": "Q_DOC_01"}

    upload_q_res = client.post(f"/api/v1/cases/{case_id}/samples", files=files, data=data)
    assert upload_q_res.status_code == 201
    q_data = upload_q_res.json()
    assert q_data["sample_id"] == "Q_DOC_01"
    assert len(q_data["sha256_hash"]) == 64

    # Vakanın durumu PHASE_A_ANALYSIS olmalı
    case_state_res = client.get(f"/api/v1/cases/{case_id}")
    assert case_state_res.json()["state"] == "PHASE_A_ANALYSIS"

    # 2. Referans (Reference) Belgeleri Yükle
    for i in range(1, 3):
        r_bytes = create_synthetic_png_bytes(f"Referans Imza {i}")
        r_files = {"file": (f"ref_{i}.png", io.BytesIO(r_bytes), "image/png")}
        r_data = {"sample_type": "reference", "sample_id": f"REF_DOC_0{i}"}
        r_res = client.post(f"/api/v1/cases/{case_id}/samples", files=r_files, data=r_data)
        assert r_res.status_code == 201


# ==============================================================================
# 2. Uygunluk ve Analiz Pipeline Testleri
# ==============================================================================

def test_suitability_endpoint_with_file():
    """POST /api/v1/analysis/suitability uç noktasını dosya yüklemesi ile test eder."""
    img_bytes = create_synthetic_png_bytes("Net Yazi Ornegi")
    files = {"file": ("sample.png", io.BytesIO(img_bytes), "image/png")}

    response = client.post("/api/v1/analysis/suitability", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "suitability" in data
    assert data["suitability"]["status"] in ("SUITABLE", "PARTIALLY_SUITABLE")
    assert data["suitability"]["overall_score"] > 0.70


def test_full_pipeline_orchestration_flow():
    """
    Uçtan Uca Tam Senaryo:
    Vaka Aç -> Q & Ref Yükle -> Phase A Dondur -> Full Pipeline Çalıştır -> Raporu İndir
    """
    case_id = "VAKA_PIPELINE_END_TO_END"
    client.post("/api/v1/cases", json={"case_id": case_id, "examiner_id": "BILIRKISI_AI"})

    # 1. Q Yükle
    q_bytes = create_synthetic_png_bytes("Sorgulanan Senet Imzasi")
    client.post(
        f"/api/v1/cases/{case_id}/samples",
        files={"file": ("q.png", io.BytesIO(q_bytes), "image/png")},
        data={"sample_type": "questioned", "sample_id": "Q_PIPE_01"},
    )

    # 2. Ref Yükle (3 adet)
    ref_ids = ["REF_PIPE_01", "REF_PIPE_02", "REF_PIPE_03"]
    for ref_id in ref_ids:
        r_bytes = create_synthetic_png_bytes("Mukayese Imza")
        client.post(
            f"/api/v1/cases/{case_id}/samples",
            files={"file": (f"{ref_id}.png", io.BytesIO(r_bytes), "image/png")},
            data={"sample_type": "reference", "sample_id": ref_id},
        )

    # 3. Phase A Kilitle / Dondur (Bilişsel İzolasyon)
    isolate_payload = {
        "case_id": case_id,
        "questioned_sample_id": "Q_PIPE_01",
        "examiner_notes": "Phase A analizi yapıldı, öznitelikler kilitlendi.",
    }
    isolate_res = client.post("/api/v1/analysis/isolate-phase-a", json=isolate_payload)
    assert isolate_res.status_code == 200
    assert len(isolate_res.json()["integrity_hash_sha256"]) == 64

    # 4. Tam Pipeline Orkestrasyonunu Başlat
    pipeline_payload = {
        "case_id": case_id,
        "questioned_sample_id": "Q_PIPE_01",
        "reference_sample_ids": ref_ids,
        "suspect_name": "Mehmet Demir",
        "generate_pdf": True,
    }
    pipeline_res = client.post("/api/v1/analysis/full-pipeline", json=pipeline_payload)
    assert pipeline_res.status_code == 200
    report_data = pipeline_res.json()

    # Çıktıların bütünlüğünü doğrula
    assert report_data["case_id"] == case_id
    assert "suitability" in report_data
    assert "complexity" in report_data
    assert "comparison" in report_data
    assert "disguise" in report_data
    assert "fusion" in report_data
    assert "xai" in report_data
    assert "hypothesis" in report_data
    assert report_data["fusion"]["hybrid_similarity_score"] >= 0.0
    assert len(report_data["audit_checksum_sha256"]) == 64

    # 5. Mühürlü Raporu İndir
    report_dl_res = client.get(f"/api/v1/analysis/report/pdf/{case_id}")
    assert report_dl_res.status_code == 200
    assert len(report_dl_res.content) > 100


def test_delete_case_and_cascade_cleanup():
    """DELETE /api/v1/cases/{case_id} uç noktasını, kalıcı silme ve 404 doğrulamasını test eder."""
    case_id = "VAKA_DELETE_TEST_01"
    
    # 1. Yeni Vaka Oluştur
    create_res = client.post("/api/v1/cases", json={"case_id": case_id, "name": "Silinecek Adli Vaka"})
    assert create_res.status_code == 201

    # 2. Numune Yükle
    q_bytes = create_synthetic_png_bytes("Silinecek Numune")
    upload_res = client.post(
        f"/api/v1/cases/{case_id}/samples",
        files={"file": ("del_sample.png", io.BytesIO(q_bytes), "image/png")},
        data={"sample_type": "questioned", "sample_id": "DEL_Q_01"},
    )
    assert upload_res.status_code == 201

    # 3. Vakayı Sil
    del_res = client.delete(f"/api/v1/cases/{case_id}", headers=admin_headers)
    assert del_res.status_code == 200
    del_data = del_res.json()
    assert del_data["status"] == "success"
    assert del_data["case_id"] == case_id

    # 4. Silinen Vakanın 404 Döndürdüğünü Doğrula
    get_res = client.get(f"/api/v1/cases/{case_id}")
    assert get_res.status_code == 404

    # 5. Listede Yer Almadığını Doğrula
    list_res = client.get("/api/v1/cases")
    assert list_res.status_code == 200
    assert not any(c["case_id"] == case_id for c in list_res.json())

    # 6. Tekrar Silmeye Çalışıldığında 404 Döndüğünü Doğrula
    del_again_res = client.delete(f"/api/v1/cases/{case_id}", headers=admin_headers)
    assert del_again_res.status_code == 404


def test_delete_sample_and_reanalyze_flow():
    """
    Mukayese örneği silinip yenisi eklendiğinde ve analiz tekrar başlatıldığında
    Bilişsel İzolasyon İhlali olmadan analizin sorunsuz tamamlandığını doğrular.
    """
    case_id = "VAKA_REANALYZE_TEST_01"
    client.post("/api/v1/cases", json={"case_id": case_id, "name": "Yeniden Analiz Test Vakası"})

    # 1. Şüpheli Yükle
    q_bytes = create_synthetic_png_bytes("Sorgulanan Imza V1")
    client.post(
        f"/api/v1/cases/{case_id}/samples",
        files={"file": ("q.png", io.BytesIO(q_bytes), "image/png")},
        data={"sample_type": "questioned", "sample_id": "Q_RE_01"},
    )

    # 2. Mukayese 1 ve 2 Yükle
    for i in range(1, 3):
        r_bytes = create_synthetic_png_bytes(f"Mukayese Imza {i}")
        client.post(
            f"/api/v1/cases/{case_id}/samples",
            files={"file": (f"ref_{i}.png", io.BytesIO(r_bytes), "image/png")},
            data={"sample_type": "reference", "sample_id": f"REF_RE_0{i}"},
        )

    # 3. İlk Analizi Çalıştır
    pipe1_res = client.post(
        "/api/v1/analysis/full-pipeline",
        json={
            "case_id": case_id,
            "questioned_sample_id": "Q_RE_01",
            "reference_sample_ids": ["REF_RE_01", "REF_RE_02"],
        },
    )
    assert pipe1_res.status_code == 200

    # Vaka durumu EVALUATION_COMPLETED olmalıdır
    case_info = client.get(f"/api/v1/cases/{case_id}").json()
    assert case_info["state"] == "EVALUATION_COMPLETED"

    # 4. Mukayese 2'yi Sil
    del_ref_res = client.delete(f"/api/v1/cases/{case_id}/samples/REF_RE_02", headers=admin_headers)
    assert del_ref_res.status_code == 200

    # 5. Yeni Mukayese 3 Yükle
    r3_bytes = create_synthetic_png_bytes("Yeni Mukayese Imza 3")
    client.post(
        f"/api/v1/cases/{case_id}/samples",
        files={"file": ("ref_3.png", io.BytesIO(r3_bytes), "image/png")},
        data={"sample_type": "reference", "sample_id": "REF_RE_03"},
    )

    # 6. Analizi Yeni Mukayese Seti ile Tekrar Başlat
    pipe2_res = client.post(
        "/api/v1/analysis/full-pipeline",
        json={
            "case_id": case_id,
            "questioned_sample_id": "Q_RE_01",
            "reference_sample_ids": ["REF_RE_01", "REF_RE_03"],
        },
    )
    assert pipe2_res.status_code == 200
    report = pipe2_res.json()
    assert report["case_id"] == case_id
    assert report["comparison"]["reference_count"] == 2


