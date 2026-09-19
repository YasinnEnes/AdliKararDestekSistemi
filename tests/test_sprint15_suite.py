"""
FHDSS Sprint 15: Kurumsal Adli Karar Destek Sistemi Seviyesi Kapsamlı Entegrasyon ve Doğrulama Test Paketi.
Comprehensive Integration Test Suite for Sprint 15 Features.
"""

import hashlib
import io
import json
import zipfile
import pytest
import numpy as np
import torch
from fastapi.testclient import TestClient

from backend.app.core.security import create_access_token, get_password_hash, verify_password
from backend.app.main import app
from backend.app.services.audit_service import (
    compute_audit_hash,
    record_audit_trail,
    verify_audit_hash_chain,
)
from backend.app.services.legal_signature_service import (
    MockElectronicSignatureProvider,
    MockTimestampProvider,
)
from backend.app.services.multi_sample_service import multi_sample_clustering_service
from backend.app.services.onnx_inference import ForensicONNXInferenceEngine
from cv.biometrics.dynamic_features import extract_dynamic_features
from cv.biometrics.hardware_providers import (
    BrowserPointerProvider,
    MockSignatureProvider,
    StylusProvider,
)
from ml.models.siamese import ForensicConvNeXtEncoder

client = TestClient(app)


# ==============================================================================
# 1. KİMLİK DOĞRULAMA VE RBAC TESTLERİ
# ==============================================================================

def test_auth_password_hashing_and_jwt():
    """Şifre karma ve JWT token üretimi / doğrulaması."""
    raw_pwd = "GuvenliAdliSifre2026!"
    hashed = get_password_hash(raw_pwd)
    assert verify_password(raw_pwd, hashed)
    assert not verify_password("YanlisSifre", hashed)

    token = create_access_token(subject="adli_uzman", role="SENIOR_REVIEWER")
    assert isinstance(token, str)
    assert len(token) > 20


def test_auth_login_api():
    """Yeni kullanıcı kaydı ve /api/v1/auth/login endpointi testi."""
    reg_res = client.post(
        "/api/v1/auth/register",
        json={
            "username": "test_sprint15_user",
            "password": "GuvenliSifre2026!",
            "full_name": "Test Uzman",
            "role": "SENIOR_REVIEWER",
        },
    )
    assert reg_res.status_code in (200, 201)

    res = client.post(
        "/api/v1/auth/login",
        json={"username": "test_sprint15_user", "password": "GuvenliSifre2026!"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["user"]["role"] == "SENIOR_REVIEWER"
    assert data["token_type"] == "bearer"


def test_auth_get_me_with_jwt():
    """Geçerli Bearer token ile /api/v1/auth/me sorgusu."""
    token = create_access_token(subject="test_examiner", role="EXAMINER")
    res = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    user = res.json()
    assert user["username"] == "test_examiner"
    assert user["role"] == "EXAMINER"


# ==============================================================================
# 2. CANLI İMZA PEDİ VE DİNAMİK BİYOMETRİ (HAL) TESTLERİ
# ==============================================================================

def test_dynamic_biometric_feature_extraction():
    """Kinematik öznitelik çıkarıcı ve normalize zaman serisi testi."""
    raw_points = [
        {"x": 100, "y": 100, "pressure": 0.5, "time": 0.0, "is_pen_down": True},
        {"x": 120, "y": 110, "pressure": 0.7, "time": 0.05, "is_pen_down": True},
        {"x": 150, "y": 130, "pressure": 0.8, "time": 0.10, "is_pen_down": True},
        {"x": 190, "y": 140, "pressure": 0.0, "time": 0.15, "is_pen_down": False},
        {"x": 220, "y": 150, "pressure": 0.6, "time": 0.20, "is_pen_down": True},
    ]

    features = extract_dynamic_features(raw_points)
    assert features.stroke_count >= 1
    assert features.mean_velocity >= 0.0
    assert features.mean_pressure > 0.0
    assert features.total_duration_sec >= 0.0


def test_hardware_abstraction_providers():
    """Tarayıcı, Stylus ve Mock donanım sağlayıcılarının çıktısı."""
    mock_prov = MockSignatureProvider()
    pts = mock_prov.capture_sample()
    assert len(pts) > 0
    assert mock_prov.get_capabilities()["supports_pressure"] is True

    browser_prov = BrowserPointerProvider()
    assert browser_prov.get_capabilities()["provider"] == "BrowserPointerProvider"

    stylus_prov = StylusProvider()
    assert stylus_prov.get_capabilities()["supports_tilt"] is True


def test_capture_live_api_endpoint():
    """POST /api/v1/biometrics/capture-live endpointi doğrulaması."""
    payload = {
        "points": [
            {"x": 50, "y": 50, "pressure": 0.5, "timeStamp": 0, "pointerType": "pen"},
            {"x": 75, "y": 60, "pressure": 0.6, "timeStamp": 30, "pointerType": "pen"},
            {"x": 110, "y": 80, "pressure": 0.7, "timeStamp": 60, "pointerType": "pen"},
        ],
        "provider": "stylus",
        "device_name": "Wacom Cintiq Pro 24",
    }
    res = client.post("/api/v1/biometrics/capture-live", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["point_count"] == 3
    assert "features" in data
    assert data["features"]["mean_velocity"] >= 0.0
    assert data["provider_capabilities"]["supports_pressure"] is True


# ==============================================================================
# 3. ONNX CPU VE PYTORCH FALLBACK ÇIKARIM TESTLERİ
# ==============================================================================

def test_onnx_engine_and_pytorch_fallback():
    """ConvNeXt modelinin ONNX veya PyTorch Fallback çıkarımı ve L2 norm testi."""
    encoder = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    engine = ForensicONNXInferenceEngine(pytorch_model=encoder)

    status = engine.get_status()
    assert status["device"] == "CPU"
    assert status["active_engine"] in ("ONNX_RUNTIME_CPU", "PYTORCH_FALLBACK")

    dummy_tensor = torch.randn(1, 3, 224, 224)
    emb = engine.encode(dummy_tensor)
    assert emb.shape == (1, 128)
    norm = np.linalg.norm(emb[0])
    assert pytest.approx(norm, abs=1e-3) == 1.0


def test_onnx_inference_status_api():
    """GET /api/v1/analysis/inference-engine-status endpointi testi."""
    token = create_access_token({"sub": "admin_user", "role": "ADMIN"})
    res = client.get(
        "/api/v1/analysis/inference-engine-status",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "active_engine" in data
    assert "device" in data
    assert data["device"] == "CPU"


# ==============================================================================
# 4. ÇOKLU NUMUNE VE PCA 2D KÜMELEME TESTLERİ
# ==============================================================================

def test_multi_sample_pca_clustering():
    """PCA 2D projeksiyonu, yazar içi tolerans elipsi ve anomali tespiti."""
    np.random.seed(42)
    # Referans yazarın embedding kümesi (birbirine yakın)
    base_ref = np.random.randn(128).astype(np.float32)
    base_ref /= np.linalg.norm(base_ref)

    ref_samples = []
    for i in range(5):
        noise = np.random.randn(128).astype(np.float32) * 0.05
        ref_emb = base_ref + noise
        ref_emb /= np.linalg.norm(ref_emb)
        ref_samples.append({"sample_id": f"REF_00{i+1}", "embedding": ref_emb})

    # 1. Doğal varyasyon içi sorgulanan imza
    q_inside = base_ref + np.random.randn(128).astype(np.float32) * 0.04
    q_inside /= np.linalg.norm(q_inside)

    res_inside = multi_sample_clustering_service.analyze_clusters(
        questioned_sample_id="Q_NATURAL",
        questioned_embedding=q_inside,
        reference_samples=ref_samples,
    )
    assert res_inside["questioned"]["is_inside_tolerance_ellipse"] is True
    assert res_inside["intra_writer_variability"]["reference_count"] == 5

    # 2. Farklı yazar / tahrifatlı imza (uzak embedding)
    q_outside = np.random.randn(128).astype(np.float32)
    q_outside /= np.linalg.norm(q_outside)

    res_outside = multi_sample_clustering_service.analyze_clusters(
        questioned_sample_id="Q_ANOMALY",
        questioned_embedding=q_outside,
        reference_samples=ref_samples,
    )
    assert res_outside["questioned"]["is_inside_tolerance_ellipse"] is False
    assert res_outside["questioned"]["forensic_status"] == "VARYASYON_DIŞINDA_ANOMALİ"


# ==============================================================================
# 5. KRİPTOGRAFİK HASH-CHAINED AUDIT TRAIL TESTLERİ
# ==============================================================================

def test_audit_hash_computation():
    """Deterministik audit hash hesaplama testi."""
    h1 = compute_audit_hash(
        sequence_number=1,
        previous_hash="0" * 64,
        action="TEST_ACTION",
        case_id="CASE_001",
        user_id="UZMAN_1",
        timestamp_utc="2026-09-11T12:00:00Z",
        details_str="{}",
    )
    assert len(h1) == 64

    # Herhangi bir parametre değiştiğinde hash'in değiştiğini doğrula
    h2 = compute_audit_hash(
        sequence_number=1,
        previous_hash="0" * 64,
        action="TEST_ACTION_MODIFIED",
        case_id="CASE_001",
        user_id="UZMAN_1",
        timestamp_utc="2026-09-11T12:00:00Z",
        details_str="{}",
    )
    assert h1 != h2


@pytest.mark.asyncio
async def test_audit_chain_verification_tampering():
    """Sahte denetim zinciri üzerinde bütünlük ve tahrifat tespiti doğrulaması."""
    from backend.app.models.entities import AuditLogEntity

    # İki sıralı geçerli blok simüle et
    rec1 = AuditLogEntity(
        id="audit_1",
        sequence_number=1,
        previous_hash="0" * 64,
        action="PHASE_A_FROZEN",
        case_id="VAKA_99",
        examiner_id="EXAMINER",
        created_at_utc="2026-09-11T12:00:00Z",
        details_json={},
        event_type="AUDIT_CHAIN_RECORD",
        sha256_hash="",
    )
    rec1.sha256_hash = compute_audit_hash(
        sequence_number=rec1.sequence_number,
        previous_hash=rec1.previous_hash,
        action=rec1.action,
        case_id=rec1.case_id,
        user_id=rec1.examiner_id,
        timestamp_utc=rec1.created_at_utc,
        details_str="{}",
    )

    rec2 = AuditLogEntity(
        id="audit_2",
        sequence_number=2,
        previous_hash=rec1.sha256_hash,
        action="REPORT_SIGNED",
        case_id="VAKA_99",
        examiner_id="EXAMINER",
        created_at_utc="2026-09-11T12:05:00Z",
        details_json={},
        event_type="AUDIT_CHAIN_RECORD",
        sha256_hash="",
    )
    rec2.sha256_hash = compute_audit_hash(
        sequence_number=rec2.sequence_number,
        previous_hash=rec2.previous_hash,
        action=rec2.action,
        case_id=rec2.case_id,
        user_id=rec2.examiner_id,
        timestamp_utc=rec2.created_at_utc,
        details_str="{}",
    )

    # Tahrifat simülasyonu: rec1 action değiştirilirse hash uyuşmazlığı oluşmalı
    tampered_rec1_action = "PHASE_A_TAMPERED"
    recomputed = compute_audit_hash(
        sequence_number=rec1.sequence_number,
        previous_hash=rec1.previous_hash,
        action=tampered_rec1_action,
        case_id=rec1.case_id,
        user_id=rec1.examiner_id,
        timestamp_utc=rec1.created_at_utc,
        details_str="{}",
    )
    assert rec1.sha256_hash != recomputed


# ==============================================================================
# 6. 5070 E-İMZA VE ZAMAN DAMGASI SOYUTLAMA KATMANI TESTLERİ
# ==============================================================================

def test_legal_electronic_signature_provider():
    """5070 Mock E-İmza üretimi, yasal uyarı ve doğrulama testi."""
    provider = MockElectronicSignatureProvider()
    doc_bytes = b"ADLI_BILIRKISI_RAPORU_RESMI_METIN_2026"

    bundle = provider.sign_document(
        data_bytes=doc_bytes,
        signer_name="Uzm. Dr. Yasin Adli",
        tc_kimlik_no="12345678901",
    )

    assert bundle["is_mock"] is True
    assert "legal_disclaimer" in bundle
    assert "5070" in bundle["legal_disclaimer"]
    assert bundle["tc_kimlik_masked"] == "123*****01"
    assert provider.verify_signature(doc_bytes, bundle) is True
    assert provider.verify_signature(b"DEGISTIRILMIS_METIN", bundle) is False


def test_timestamp_provider():
    """RFC 3161 Mock Zaman Damgası üretimi ve doğrulaması."""
    ts_provider = MockTimestampProvider()
    data = b"HASH_CHAIN_AUDIT_RECORD_BLOCK"

    ts_bundle = ts_provider.create_timestamp(data)
    assert ts_bundle["is_mock"] is True
    assert "RFC 3161" in ts_bundle["protocol"]
    assert ts_provider.verify_timestamp(data, ts_bundle) is True
    assert ts_provider.verify_timestamp(b"TAMPERED", ts_bundle) is False


# ==============================================================================
# 7. GÜVENLİ ADLİ ARŞİV PAKETİ (.ZIP) VE MANİFEST DOĞRULAMA TESTLERİ
# ==============================================================================

def test_forensic_archive_verification_integrity():
    """ZIP adli arşiv paketi oluşturma, manifest denetimi ve tahrifat algılama testi."""
    from backend.app.services.archive_service import forensic_archive_service

    # Bellek içinde yapay bir adli ZIP paketi oluştur
    zip_buffer = io.BytesIO()
    test_content_1 = b"ORIJINAL_DELIL_GORSELI_BAYTLARI"
    test_content_2 = b'{"case_id": "VAKA_TEST", "notes": "Adli Aciklama"}'

    sha1 = hashlib.sha256(test_content_1).hexdigest()
    sha2 = hashlib.sha256(test_content_2).hexdigest()

    manifest = {
        "archive_format_version": "FHDSS-FORENSIC-ZIP-V1",
        "case_id": "VAKA_TEST",
        "files": [
            {"path": "evidence/sample1.png", "sha256": sha1},
            {"path": "case/metadata.json", "sha256": sha2},
        ],
    }

    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("evidence/sample1.png", test_content_1)
        zf.writestr("case/metadata.json", test_content_2)
        zf.writestr("manifest.json", json.dumps(manifest).encode("utf-8"))

    zip_bytes = zip_buffer.getvalue()

    # 1. Bütünlüğü tam olan arşivin doğrulanması
    report_valid = forensic_archive_service.verify_case_archive(zip_bytes)
    assert report_valid["is_valid"] is True
    assert report_valid["verified_files_count"] == 2
    assert len(report_valid["tampered_files"]) == 0

    # 2. Tahrifat yapılmış arşiv testi (sample1.png değiştirilmiş)
    tampered_buffer = io.BytesIO()
    with zipfile.ZipFile(tampered_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("evidence/sample1.png", b"TAHRIFATLI_DEGISTIRILMIS_ICERIK")
        zf.writestr("case/metadata.json", test_content_2)
        zf.writestr("manifest.json", json.dumps(manifest).encode("utf-8"))

    tampered_bytes = tampered_buffer.getvalue()
    report_tampered = forensic_archive_service.verify_case_archive(tampered_bytes)
    assert report_tampered["is_valid"] is False
    assert len(report_tampered["tampered_files"]) == 1
    assert "sample1.png" in report_tampered["tampered_files"][0]["path"]


# ==============================================================================
# 8. ASENKRON İŞ KUYRUĞU VE SSE BROADCASTER TESTİ
# ==============================================================================

@pytest.mark.asyncio
async def test_job_event_broadcaster():
    """JobEventBroadcaster subscribe ve publish mekanizması testi."""
    from backend.app.services.job_service import broadcaster

    job_id = "test_job_123"
    queue = broadcaster.subscribe(job_id)

    event_data = {"status": "RUNNING", "progress": 50, "step": "Morfolojik Analiz"}
    await broadcaster.publish(job_id, event_data)

    received = await queue.get()
    assert received["progress"] == 50
    assert received["step"] == "Morfolojik Analiz"

    broadcaster.unsubscribe(job_id, queue)
